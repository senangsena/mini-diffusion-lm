"""トイタスクのデータ生成.

系列は  [x_1 ... x_L] [SEP] [y_1 ... y_L]  という固定長の並びで表す。
先頭の L+1 トークン (入力 + SEP) がプロンプト部で、常に非マスク。
後半の L トークンが応答部で、拡散(マスク復元)の対象になる。

タスク:
  sort   : 入力の L 桁を昇順に並べ替える。正解が一意に決まる。
  branch : 正解が2通りある。y = x か、y_i = (x_i + 5) mod 10 のどちらか。
           どちらを選んでもよいが、系列全体で同じ側に揃っていないと不正解。
           位置ごとの分布は 1/2 ずつなので、各位置を独立に選ぶと必ず壊れる。
"""

import torch

L = 12                      # 入力長 = 応答長
SEP = 10                    # 区切りトークン
MASK = 11                   # マスクトークン
VOCAB_SIZE = 12             # 0-9, SEP, MASK
PROMPT_LEN = L + 1
SEQ_LEN = 2 * L + 1
TASKS = ("sort", "branch")
SHIFT = 5                   # branch タスクのずらし幅


def make_batch(task: str, n: int, generator: torch.Generator) -> torch.Tensor:
    """(n, SEQ_LEN) の int64 テンソルを返す。"""
    x = torch.randint(0, 10, (n, L), generator=generator, dtype=torch.long)
    if task == "sort":
        y, _ = torch.sort(x, dim=1)
    elif task == "branch":
        b = torch.randint(0, 2, (n, 1), generator=generator, dtype=torch.long)
        y = (x + SHIFT * b) % 10
    else:
        raise ValueError(f"unknown task: {task}")
    sep = torch.full((n, 1), SEP, dtype=torch.long)
    return torch.cat([x, sep, y], dim=1)


def make_test_set(task: str, n: int = 512, seed: int = 20260730) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return make_batch(task, n, g)


def is_correct(task: str, seq: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """seq: (B, SEQ_LEN) 入力を含む正解系列, pred: (B, L) 生成された応答。"""
    x = seq[:, :L]
    if task == "sort":
        y, _ = torch.sort(x, dim=1)
        return (pred == y).all(dim=1)
    if task == "branch":
        ok0 = (pred == x).all(dim=1)
        ok1 = (pred == (x + SHIFT) % 10).all(dim=1)
        return ok0 | ok1
    raise ValueError(task)


def local_valid_rate(task: str, seq: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """位置ごとに見て「ありうるトークン」になっている割合 (B,)。

    branch では「局所的には正しいが全体として整合していない」状態を検出するために使う。
    """
    x = seq[:, :L]
    if task == "sort":
        y, _ = torch.sort(x, dim=1)
        return (pred == y).float().mean(dim=1)
    if task == "branch":
        ok = (pred == x) | (pred == (x + SHIFT) % 10)
        return ok.float().mean(dim=1)
    raise ValueError(task)
