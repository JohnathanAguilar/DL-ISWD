# models/model_config.py
from dataclasses import dataclass

@dataclass
class ModelConfig:
    in_channels: int = 2          # collator stacks 2 channels by default
    block_size: int = 10          # 10x10 tiles, per proposal
    embed_dim: int = 128          # d
    depth: int = 3                # transformer layers
    num_heads: int = 4            # MHSA heads
    mlp_ratio: float = 2.0
    drop: float = 0.1
    attn_drop: float = 0.1
    sra: bool = False             # enable Spatial Reduction Attention when N is large
    sra_ratio: int = 2            # downsample factor for keys/values
    positional: str = "learned2d" # or "sincos2d"

    # --- Dense pixel-resolution refinement head ---
    dense_head: bool = False      # attach DenseRefinementHead; patch branch is unchanged
    dense_hidden: int = 64        # channels in the refinement conv stack
    dense_use_cnn_features: bool = True  # False = transformer-only ablation
    dense_token_upsample: str = "bilinear"  # token->pixel broadcast mode; "nearest" reproduces pre-fix ablations
