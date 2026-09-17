# src/training/optim.py
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

def build_optim_and_sched(model, lr: float, weight_decay: float, epochs: int):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    return optimizer, scheduler