# models/head.py
import torch
import torch.nn as nn

class TokenHead(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x: torch.Tensor):
        # x: [B, N, D] -> logits: [B, N]
        return self.mlp(x).squeeze(-1)