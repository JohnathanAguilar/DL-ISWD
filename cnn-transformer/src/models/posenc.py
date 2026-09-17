# models/posenc.py
import math
import torch
import torch.nn as nn

class Learned2DPosEnc(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # We’ll generate a [Bh, Bw, D] map on the fly using a small conv
        self.gen = nn.Sequential(
            nn.Conv2d(1, embed_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
        )

    def forward(self, Bh: int, Bw: int, device=None):
        base = torch.ones(1, 1, Bh, Bw, device=device)         # [1,1,Bh,Bw]
        pe = self.gen(base).permute(0, 2, 3, 1).reshape(1, Bh * Bw, self.embed_dim)  # [1,N,D]
        return pe

def build_sincos_2d(Bh: int, Bw: int, dim: int, device=None):
    # Standard 2D sin-cos (no params), returns [1, N, D]
    assert dim % 4 == 0, "sincos_2d requires dim divisible by 4"
    y = torch.arange(Bh, device=device).float().unsqueeze(1).repeat(1, Bw)
    x = torch.arange(Bw, device=device).float().unsqueeze(0).repeat(Bh, 1)
    omega = torch.arange(dim // 4, device=device).float()
    omega = 1.0 / (10000 ** (omega / (dim // 4)))
    pos_y = y.reshape(-1).unsqueeze(-1) * omega
    pos_x = x.reshape(-1).unsqueeze(-1) * omega
    pe = torch.cat([pos_y.sin(), pos_y.cos(), pos_x.sin(), pos_x.cos()], dim=-1)
    return pe.unsqueeze(0)  # [1, N, D]