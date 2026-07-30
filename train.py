"""3種類の学習目的でマスク拡散LMを学習する.

  random : 完全ランダムマスク (論文 Eq.2 の J_full)
  semiar : ブロック単位の半自己回帰マスク (論文 Eq. J_semi)
  trace  : 事前学習済みモデル自身の推論トレースに沿ったマスク (論文 3.2 節)
"""

import copy
import time

import torch
import torch.nn.functional as F

from data import L, MASK, PROMPT_LEN, make_batch
from model import TinyDLM


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _device()


def random_masking_loss(model, batch, t_min=0.02):
    B = batch.shape[0]
    t = torch.rand(B, 1, device=batch.device).clamp_min(t_min)
    resp = batch[:, PROMPT_LEN:]
    m = torch.rand(B, L, device=batch.device) < t
    m[m.sum(dim=1) == 0, 0] = True                 # 最低1つはマスクする
    x = batch.clone()
    x[:, PROMPT_LEN:] = torch.where(m, torch.full_like(resp, MASK), resp)
    logits = model(x)[:, PROMPT_LEN:]
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                         resp.reshape(-1), reduction="none").view(B, L)
    return ((ce * m).sum(dim=1) / (t.squeeze(1) * L)).mean()


def semiar_masking_loss(model, batch, block_size, t_min=0.02):
    """ブロック k をランダムに選び、k より前は正解、k より後は全マスクにする."""
    B = batch.shape[0]
    n_blocks = (L + block_size - 1) // block_size
    resp = batch[:, PROMPT_LEN:]
    k = torch.randint(0, n_blocks, (B,), device=batch.device)
    pos = torch.arange(L, device=batch.device).unsqueeze(0)
    blk_id = pos // block_size
    future = blk_id > k.unsqueeze(1)
    cur = blk_id == k.unsqueeze(1)

    t = torch.rand(B, 1, device=batch.device).clamp_min(t_min)
    m_cur = (torch.rand(B, L, device=batch.device) < t) & cur
    empty = (m_cur.sum(dim=1) == 0)
    if empty.any():                                # 最低1つはマスクする
        first = (k * block_size)[empty]
        m_cur[empty.nonzero(as_tuple=True)[0], first] = True

    x = batch.clone()
    x[:, PROMPT_LEN:] = torch.where(m_cur | future, torch.full_like(resp, MASK), resp)
    logits = model(x)[:, PROMPT_LEN:]
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                         resp.reshape(-1), reduction="none").view(B, L)
    n_cur = cur.sum(dim=1).clamp_min(1)
    return ((ce * m_cur).sum(dim=1) / (t.squeeze(1) * n_cur)).mean()


@torch.no_grad()
def collect_trace(base_model, batch, k_per_step=2):
    """正解を教師として、base_model が「自信を持って埋める順序」を求める.

    論文 B.1 と同じ手続き: 全マスクから始めて信頼度上位 k 個を正解で埋める、を繰り返す。
    戻り値: (B, L) の step_map (1始まり)
    """
    base_model.eval()
    B = batch.shape[0]
    resp = batch[:, PROMPT_LEN:]
    x = batch.clone()
    x[:, PROMPT_LEN:] = MASK
    step_map = torch.zeros(B, L, dtype=torch.long, device=batch.device)
    step = 0
    while (step_map == 0).any():
        step += 1
        logits = base_model(x)[:, PROMPT_LEN:]
        p = F.softmax(logits, dim=-1)
        conf = p.gather(-1, resp.unsqueeze(-1)).squeeze(-1)     # 正解トークンの確率
        conf = torch.where(step_map == 0, conf, torch.full_like(conf, -1.0))
        k = min(k_per_step, int((step_map == 0).sum(dim=1).max()))
        idx = conf.topk(k, dim=-1).indices
        sel = torch.zeros_like(step_map, dtype=torch.bool)
        sel.scatter_(1, idx, True)
        sel &= step_map == 0
        step_map[sel] = step
        cur = x[:, PROMPT_LEN:]
        cur[sel] = resp[sel]
        x[:, PROMPT_LEN:] = cur
    return step_map


def trace_loss(model, base_model, batch):
    B = batch.shape[0]
    resp = batch[:, PROMPT_LEN:]
    step_map = collect_trace(base_model, batch)
    n_steps = step_map.max(dim=1).values
    pick = (torch.rand(B, device=batch.device) * n_steps).long() + 1
    target = step_map == pick.unsqueeze(1)
    future = step_map > pick.unsqueeze(1)
    x = batch.clone()
    x[:, PROMPT_LEN:] = torch.where(target | future, torch.full_like(resp, MASK), resp)
    logits = model(x)[:, PROMPT_LEN:]
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                         resp.reshape(-1), reduction="none").view(B, L)
    return ((ce * target).sum(dim=1) / target.sum(dim=1).clamp_min(1)).mean()


def train(task: str, objective: str = "random", steps: int = 4000, batch_size: int = 128,
          lr: float = 3e-4, seed: int = 0, block_size: int = 4, base_model=None,
          log_every: int = 200, verbose: bool = True):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1000)
    model = TinyDLM().to(DEVICE) if base_model is None or objective != "trace" \
        else copy.deepcopy(base_model).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    history = []
    t0 = time.time()
    for it in range(1, steps + 1):
        batch = make_batch(task, batch_size, gen).to(DEVICE)
        if objective == "random":
            loss = random_masking_loss(model, batch)
        elif objective == "semiar":
            loss = semiar_masking_loss(model, batch, block_size)
        elif objective == "trace":
            loss = trace_loss(model, base_model, batch)
        else:
            raise ValueError(objective)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % log_every == 0 or it == 1:
            history.append((it, float(loss.item())))
            if verbose:
                print(f"  [{task}/{objective}] it={it:5d} loss={loss.item():.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    return model, history
