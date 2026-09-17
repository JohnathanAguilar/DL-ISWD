# models/dense_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseRefinementHead(nn.Module):
    """
    Image-conditioned dense refinement head.

    Converts one confidence per block token into one confidence per pixel by
    combining broadcast transformer context with the full-resolution CNN
    features that still carry within-block spatial detail.

    Each transformer token is broadcast only across its own block_size x
    block_size pixel footprint, so the head never invents context from
    neighbouring blocks; the spatial variation inside a block comes entirely
    from the CNN features.

    token_upsample_mode controls how a token is spread over its footprint.
    "nearest" (the original behaviour) makes the context branch a literal
    step function at every block seam; only the two 3x3 convs below can blend
    across it afterward, which is why the learned map's gradient still
    concentrates on block seams far more than a plain bilinear/bicubic
    upsample of the same patch scores does. "bilinear" (the default) removes
    that discontinuity at its source by blending each token smoothly toward
    its neighbours before concatenation, while leaving the CNN features
    (unaffected either way) as the source of genuine within-block detail.
    "nearest" is kept available to exactly reproduce pre-fix ablations.

    Set use_cnn_features=False for the transformer-only ablation, where the
    head can only reproduce a smoothed version of the token grid.
    """

    _UPSAMPLE_MODES = ("nearest", "bilinear", "bicubic")

    def __init__(
        self,
        dim: int,
        hidden: int = 64,
        use_cnn_features: bool = True,
        token_upsample_mode: str = "bilinear",
    ):
        super().__init__()
        self.use_cnn_features = use_cnn_features

        if token_upsample_mode not in self._UPSAMPLE_MODES:
            raise ValueError(
                f"token_upsample_mode must be one of {self._UPSAMPLE_MODES}; "
                f"got {token_upsample_mode!r}"
            )
        self.token_upsample_mode = token_upsample_mode

        in_ch = dim * 2 if use_cnn_features else dim
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        feats: torch.Tensor,
        Bh: int,
        Bw: int,
        block_size: int,
    ) -> torch.Tensor:
        """
        tokens: [B, N, D] contextual transformer features (N = Bh * Bw)
        feats:  [B, D, Ht, Wt] full-resolution CNN features, already cropped to
                the tokenized region (Ht = Bh * block_size, Wt = Bw * block_size)
        returns dense logits: [B, 1, Ht, Wt]
        """
        B, N, D = tokens.shape
        if N != Bh * Bw:
            raise ValueError(f"tokens has {N} tokens but grid is {Bh}x{Bw} = {Bh * Bw}")

        Ht = Bh * block_size
        Wt = Bw * block_size
        if feats.shape[-2:] != (Ht, Wt):
            raise ValueError(
                f"feats spatial shape {tuple(feats.shape[-2:])} does not match the "
                f"tokenized region {(Ht, Wt)}; crop feats before calling the dense head."
            )

        # [B, N, D] -> [B, D, Bh, Bw] -> broadcast each token over its own block
        grid = tokens.reshape(B, Bh, Bw, D).permute(0, 3, 1, 2).contiguous()
        if self.token_upsample_mode == "nearest":
            broadcast = F.interpolate(grid, scale_factor=block_size, mode="nearest")
        else:
            broadcast = F.interpolate(
                grid, scale_factor=block_size, mode=self.token_upsample_mode, align_corners=False
            )

        if self.use_cnn_features:
            x = torch.cat([broadcast, feats], dim=1)
        else:
            x = broadcast

        return self.net(x)
