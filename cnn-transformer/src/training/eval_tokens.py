# src/training/eval_tokens.py
import os
import torch
from dataio import DataCollatorBlocks, create_dataloaders_train_val, make_nc_block_dataset
from models import ModelConfig, BlocksTokenModel
from .engine import Trainer

def evaluate_checkpoint(ckpt_path: str, nc_dir: str, channel_vars, mask_var: str, block: int, test_ratio: float, seed: int = 123):
    test_items, valid_items = make_nc_block_dataset(
        nc_dir=nc_dir, channel_var_names=tuple(channel_vars), mask_var_name=mask_var,
        block_size=(block, block), test_ratio=test_ratio, seed=seed,
    )
    collator = DataCollatorBlocks(pad_images=True, labels_as_float=True, make_block_attention_mask=True)
    _, val_loader = create_dataloaders_train_val(valid_items, test_items, collate_fn=collator, train_batch_size=1, val_batch_size=16, num_workers=0, pin_memory=False, bucket_by_grid=True, seed=seed)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = ckpt.get("config", {})
    cfg = ModelConfig(**cfg_dict)
    model = BlocksTokenModel(cfg)
    model.load_state_dict(ckpt["model_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(model, device, alpha=0.7, threshold=0.5, save_dir=os.path.dirname(ckpt_path))
    val_loss, metrics = trainer.evaluate(val_loader)
    print(f"Loaded {ckpt_path} | val_loss={val_loss:.4f} | auc={metrics['auc']:.4f} | f1@0.5={metrics['f1']:.4f}")