# models/block_tokenizer.py
import torch
import torch.nn as nn

class BlockTokenizer(nn.Module):
    def __init__(self, block_size: int = 10):
        super().__init__()
        self.block = block_size
        self.pool = nn.AvgPool2d(kernel_size=block_size, stride=block_size)

    def forward(self, feats: torch.Tensor, block_mask: torch.Tensor | None = None):
        """
        feats: [B, C, H, W]
        block_mask: [B, Bh, Bw] (bool) or None
        returns tokens: [B, N, C], token_mask: [B, N] (bool), Bh, Bw
        """
        B, C, H, W = feats.shape
        pooled = self.pool(feats)           # [B, C, Bh, Bw]
        B, C, Bh, Bw = pooled.shape
        tokens = pooled.permute(0, 2, 3, 1).reshape(B, Bh * Bw, C)  # [B, N, C]
        if block_mask is None:
            token_mask = torch.ones(B, Bh * Bw, dtype=torch.bool, device=feats.device)
        else:
            token_mask = block_mask.reshape(B, Bh * Bw)
        return tokens, token_mask, Bh, Bw