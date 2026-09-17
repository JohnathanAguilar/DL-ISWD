# models/token_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_config import ModelConfig
from .cnn_stem import CNNStem
from .block_tokenizer import BlockTokenizer
from .posenc import Learned2DPosEnc, build_sincos_2d
from .transformer import TransformerEncoder
from .head import TokenHead
from .dense_head import DenseRefinementHead

class BlocksTokenModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.stem = CNNStem(cfg.in_channels, width=cfg.embed_dim)   # keep C = embed_dim to simplify
        self.tokenizer = BlockTokenizer(cfg.block_size)

        # Positional encoding
        if cfg.positional == "learned2d":
            self.posenc = Learned2DPosEnc(cfg.embed_dim)
        elif cfg.positional == "sincos2d":
            self.posenc = None  # we'll generate in forward
        else:
            raise ValueError(f"Unknown positional: {cfg.positional}")

        self.encoder = TransformerEncoder(
            dim=cfg.embed_dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            drop=cfg.drop,
            attn_drop=cfg.attn_drop,
            sra=cfg.sra,
            sra_ratio=cfg.sra_ratio,
        )
        self.head = TokenHead(cfg.embed_dim)

        if getattr(cfg, "dense_head", False):
            self.dense_head = DenseRefinementHead(
                dim=cfg.embed_dim,
                hidden=cfg.dense_hidden,
                use_cnn_features=cfg.dense_use_cnn_features,
                token_upsample_mode=cfg.dense_token_upsample,
            )
        else:
            self.dense_head = None

    # ------------------------------------------------------------------
    # Backbone freezing, used by Stage 1 of the dense-refinement training
    # ------------------------------------------------------------------

    def backbone_modules(self):
        """Everything that defines the original patch-level detector."""
        mods = [self.stem, self.tokenizer, self.encoder, self.head]
        if self.posenc is not None:
            mods.append(self.posenc)
        return mods

    def freeze_backbone(self, freeze: bool = True):
        """
        Freeze/unfreeze the original detector so only the dense head trains.

        Call set_train_mode() rather than train() afterwards: the CNN stem uses
        BatchNorm, whose running statistics would otherwise keep updating and
        silently change the frozen detector's predictions.
        """
        for module in self.backbone_modules():
            for p in module.parameters():
                p.requires_grad_(not freeze)
        self._backbone_frozen = freeze
        return self

    @property
    def backbone_frozen(self) -> bool:
        return getattr(self, "_backbone_frozen", False)

    def set_train_mode(self, training: bool = True):
        """train()/eval() that keeps a frozen backbone in eval mode."""
        self.train(training)
        if training and self.backbone_frozen:
            for module in self.backbone_modules():
                module.eval()
        return self

    def forward(
        self,
        images: torch.Tensor,
        block_mask: torch.Tensor | None = None,
        return_heatmap: bool = False,
        return_dense: bool = False,
    ):
        """
        images: [B, 2, H, W]
        block_mask: [B, Bh, Bw] (bool), True = valid
        """
        feats = self.stem(images)  # [B, D, H, W]
        tokens, token_mask, Bh, Bw = self.tokenizer(feats, block_mask)  # [B, N, D], [B, N]

        # Add 2-D positional encoding
        if isinstance(self.posenc, Learned2DPosEnc):
            pe = self.posenc(Bh, Bw, device=images.device)  # [1, N, D]
        else:
            pe = build_sincos_2d(Bh, Bw, self.cfg.embed_dim, device=images.device)  # [1, N, D]
        x = tokens + pe  # broadcast [B, N, D] + [1, N, D]

        # key_padding_mask expects True for pads; we have True for valid -> invert
        kpm = (~token_mask)  # [B, N]
        x = self.encoder(x, key_padding_mask=kpm)  # [B, N, D]
        logits = self.head(x)                      # [B, N]
        probs = torch.sigmoid(logits)              # [B, N]

        out = {
            "logits": logits,
            "probs": probs,
            "token_mask": token_mask,             # True = valid
            "Bh": Bh, "Bw": Bw
        }

        if return_heatmap:
            # reshape to [B, 1, Bh, Bw] and upsample to [B, 1, H, W]
            grid_map = probs.reshape(images.shape[0], 1, Bh, Bw)
            heat_up = F.interpolate(grid_map, scale_factor=self.cfg.block_size, mode="nearest")
            out["heatmap_up"] = heat_up

        if return_dense:
            if self.dense_head is None:
                raise RuntimeError(
                    "return_dense=True but this model has no dense head. "
                    "Build it with ModelConfig(dense_head=True)."
                )
            bs = self.cfg.block_size
            Ht, Wt = Bh * bs, Bw * bs
            # The tokenizer floor-tiles, so ragged bottom/right pixels have no
            # token; crop the CNN features to the tokenized region.
            dense_logits = self.dense_head(
                tokens=x,
                feats=feats[..., :Ht, :Wt],
                Bh=Bh, Bw=Bw, block_size=bs,
            )
            out["dense_logits"] = dense_logits              # [B, 1, Ht, Wt]
            out["dense_probs"] = torch.sigmoid(dense_logits)

        return out
