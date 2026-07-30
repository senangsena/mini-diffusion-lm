"""極小のマスク拡散言語モデル (full attention)."""

import torch
import torch.nn as nn

from data import VOCAB_SIZE, SEQ_LEN


class TinyDLM(nn.Module):
    def __init__(self, d_model: int = 128, n_layers: int = 4, n_heads: int = 4,
                 d_ff: int = 512, dropout: float = 0.0):
        super().__init__()
        self.tok = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos = nn.Embedding(SEQ_LEN, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB_SIZE)
        self.register_buffer("pos_ids", torch.arange(SEQ_LEN).unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tok(x) + self.pos(self.pos_ids[:, : x.shape[1]])
        h = self.enc(h)
        return self.head(self.norm(h))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
