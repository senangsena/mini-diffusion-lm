"""ブロック単位の半自己回帰サンプリング.

dLLM-RL の generate.py:block_diffusion_generate と同じ構造にしてある。
違いは (1) full attention モデルなので未生成ブロックは MASK のまま入力に残す、
(2) KV-cache を使わない、の2点。
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from data import L, MASK, PROMPT_LEN, SEQ_LEN


def num_transfer_tokens(block_length: int, steps: int) -> torch.Tensor:
    """各デノイジングステップで確定させるトークン数 (公式実装と同じ配分)."""
    base = block_length // steps
    remainder = block_length % steps
    n = torch.zeros(steps, dtype=torch.long) + base
    n[:remainder] += 1
    return n


@dataclass
class GenOutput:
    tokens: torch.Tensor            # (B, SEQ_LEN)
    nfe: int                        # モデル forward 回数 (= 実際の総デノイジングステップ数)
    first_unmask: torch.Tensor      # (B, L) 各応答位置が何ステップ目で確定したか (1始まり)
    step_sizes: list = field(default_factory=list)


@torch.no_grad()
def generate(model, prompt_batch: torch.Tensor, block_length: int = 4,
             denoising_steps: int = 4, strategy: str = "low_confidence_static",
             confidence_threshold: float = 0.9, temperature: float = 0.0,
             confidence_mode: str = "max") -> GenOutput:
    """prompt_batch: (B, SEQ_LEN)。応答部は中身を無視して MASK で埋め直す。

    strategy: "low_confidence_static" / "low_confidence_dynamic" / "sequential"
    confidence_mode: "max"    = 論文の定義 max_v p(v)
                     "sampled"= 公式実装の定義 (サンプリングされたトークンの確率)
    """
    model.eval()
    device = next(model.parameters()).device
    x = prompt_batch.clone().to(device)
    x[:, PROMPT_LEN:] = MASK
    B = x.shape[0]

    first_unmask = torch.zeros(B, L, dtype=torch.long, device=device)
    nfe = 0
    step_sizes = []
    n_blocks = (L + block_length - 1) // block_length

    for b in range(n_blocks):
        lo = PROMPT_LEN + b * block_length
        hi = min(PROMPT_LEN + (b + 1) * block_length, SEQ_LEN)
        blk = hi - lo
        n_transfer = num_transfer_tokens(blk, denoising_steps)

        for step in range(denoising_steps):
            mask_index = x[:, lo:hi] == MASK
            if not mask_index.any():
                break
            logits = model(x)[:, lo:hi]
            nfe += 1
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                flat = probs.reshape(-1, probs.shape[-1])
                tok = torch.multinomial(flat, 1).view(B, blk)
                if confidence_mode == "sampled":
                    conf = flat.gather(-1, tok.reshape(-1, 1)).view(B, blk)
                else:
                    conf = probs.max(dim=-1).values
            else:
                probs = F.softmax(logits, dim=-1)
                conf, tok = probs.max(dim=-1)

            conf = torch.where(mask_index, conf, torch.full_like(conf, -1.0))

            if strategy == "low_confidence_static":
                transfer = torch.zeros_like(mask_index)
                k = min(int(n_transfer[step]), blk)
                idx = conf.topk(k, dim=-1).indices
                transfer.scatter_(1, idx, True)
                transfer &= mask_index
            elif strategy == "low_confidence_dynamic":
                transfer = conf > confidence_threshold
                k = min(int(n_transfer[step]), blk)
                fallback = torch.zeros_like(mask_index)
                idx = conf.topk(k, dim=-1).indices
                fallback.scatter_(1, idx, True)
                # 公式実装と同じく、閾値超えが static の本数に満たない行は topk で埋める
                short = transfer.sum(dim=1) < k
                transfer[short] = fallback[short]
                transfer &= mask_index
            elif strategy == "sequential":
                transfer = torch.zeros_like(mask_index)
                k = min(int(n_transfer[step]), blk)
                for j in range(B):
                    pos = mask_index[j].nonzero(as_tuple=True)[0]
                    if pos.numel():
                        transfer[j, pos[:k]] = True
            else:
                raise ValueError(strategy)

            step_sizes.append(int(transfer.sum().item()) / B)
            blk_x = x[:, lo:hi]
            blk_x[transfer] = tok[transfer]
            x[:, lo:hi] = blk_x
            fu = first_unmask[:, lo - PROMPT_LEN: hi - PROMPT_LEN]
            fu[transfer & (fu == 0)] = nfe
            first_unmask[:, lo - PROMPT_LEN: hi - PROMPT_LEN] = fu

        # 残ったマスクは最後にまとめて確定させる (公式実装の denoising_steps+1 回目に相当)
        mask_index = x[:, lo:hi] == MASK
        if mask_index.any():
            logits = model(x)[:, lo:hi]
            nfe += 1
            tok = logits.argmax(dim=-1)
            blk_x = x[:, lo:hi]
            blk_x[mask_index] = tok[mask_index]
            x[:, lo:hi] = blk_x
            fu = first_unmask[:, lo - PROMPT_LEN: hi - PROMPT_LEN]
            fu[mask_index & (fu == 0)] = nfe
            first_unmask[:, lo - PROMPT_LEN: hi - PROMPT_LEN] = fu

    return GenOutput(tokens=x.cpu(), nfe=nfe, first_unmask=first_unmask.cpu(),
                     step_sizes=step_sizes)


def accuracy(out_tokens: torch.Tensor, ref: torch.Tensor) -> float:
    """応答部の完全一致率."""
    pred = out_tokens[:, PROMPT_LEN:]
    gold = ref[:, PROMPT_LEN:]
    return (pred == gold).all(dim=1).float().mean().item()


def token_accuracy(out_tokens: torch.Tensor, ref: torch.Tensor) -> float:
    pred = out_tokens[:, PROMPT_LEN:]
    gold = ref[:, PROMPT_LEN:]
    return (pred == gold).float().mean().item()
