# models/cnn_stem.py
import torch
import torch.nn as nn

class CNNStem(nn.Module):
    def __init__(self, in_channels: int = 2, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, H, W] -> features: [B, C, H, W]
        return self.net(x)