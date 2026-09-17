# models/transformer.py
import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0, drop: float = 0.1, attn_drop: float = 0.1, sra: bool = False, sra_ratio: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True)
        self.sra = sra
        self.sra_ratio = sra_ratio

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        # x: [B, N, D], key_padding_mask: [B, N] (True = pad)
        x1 = self.norm1(x)
        # Optional SRA: reduce K/V length by pooling tokens (simple average over windows)
        if self.sra and x1.shape[1] % self.sra_ratio == 0:
            B, N, D = x1.shape
            r = self.sra_ratio
            # reshape to [B, N//r, r, D] and average along r
            kv = x1.reshape(B, N // r, r, D).mean(dim=2)  # [B, N//r, D]
            # Build a reduced mask
            if key_padding_mask is not None:
                kpm = key_padding_mask.reshape(B, N // r, r).logical_not().any(dim=2).logical_not()  # “pad if all were pad”
            else:
                kpm = None
            # Attend with queries = x1, keys/values = kv (cross-attn emulated via MultiheadAttention by batching)
            # Trick: split batches; use loop for clarity
            outs = []
            for b in range(B):
                out_b, _ = self.attn(x1[b:b+1], kv[b:b+1], kv[b:b+1], key_padding_mask=kpm[b:b+1] if kpm is not None else None, need_weights=False)
                outs.append(out_b)
            attn_out = torch.cat(outs, dim=0)
        else:
            attn_out, _ = self.attn(x1, x1, x1, key_padding_mask=key_padding_mask, need_weights=False)

        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, dim: int = 128, depth: int = 3, num_heads: int = 4, mlp_ratio: float = 2.0, drop: float = 0.1, attn_drop: float = 0.1, sra: bool = False, sra_ratio: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([
            EncoderBlock(dim, num_heads, mlp_ratio, drop, attn_drop, sra, sra_ratio) for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        for blk in self.blocks:
            x = blk(x, key_padding_mask)
        return x