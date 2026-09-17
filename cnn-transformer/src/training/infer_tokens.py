# src/training/infer_tokens.py
import os
import torch
import matplotlib.pyplot as plt
from models import ModelConfig, BlocksTokenModel

def save_heatmap(model_ckpt: str, images: torch.Tensor, block_mask: torch.Tensor, out_dir: str, prefix: str = "viz"):
    os.makedirs(out_dir, exist_ok=True)
    ckpt = torch.load(model_ckpt, map_location="cpu")
    cfg = ModelConfig(**ckpt.get("config", {}))
    model = BlocksTokenModel(cfg)
    model.load_state_dict(ckpt["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    with torch.no_grad():
        out = model(images.to(device), block_mask=block_mask.to(device), return_heatmap=True)
        heat = out["heatmap_up"].cpu()
    for i in range(min(images.size(0), 4)):
        fig = plt.figure(figsize=(10,4))
        plt.subplot(1,2,1); plt.title("Input (ch0)"); plt.imshow(images[i,0].cpu(), cmap="viridis"); plt.colorbar()
        plt.subplot(1,2,2); plt.title("Token confidences"); plt.imshow(heat[i,0], cmap="magma"); plt.colorbar()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{prefix}_{i:02d}.png"))
        plt.close(fig)