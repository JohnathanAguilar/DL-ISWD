# dataloader.py
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import math
import random
import torch
from torch.utils.data import Dataset, DataLoader, Sampler


# ---------- Dataset wrapper ----------
class BlockItemDataset(Dataset):
    def __init__(self, items: Sequence[Dict[str, Any]], transform: Optional[Callable] = None):
        self.items = list(items)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        return self.transform(item) if self.transform is not None else item


# ---------- Worker seeding ----------
def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    try:
        import numpy as np
        np.random.seed(worker_seed)
    except Exception:
        pass


# ---------- Bucketed batch sampler by (Bh, Bw) ----------
@dataclass
class GridShapeBucketBatchSampler(Sampler[List[int]]):
    dataset: BlockItemDataset
    batch_size: int
    shuffle: bool = True
    drop_last: bool = False
    generator: Optional[torch.Generator] = None

    def __iter__(self) -> Iterable[List[int]]:
        buckets: Dict[Tuple[int, int], List[int]] = {}

        for idx in range(len(self.dataset)):
            Bh, Bw = tuple(self.dataset[idx]["grid_shape"])
            buckets.setdefault((Bh, Bw), []).append(idx)

        bucket_keys = list(buckets.keys())

        if self.shuffle:
            if self.generator is None:
                random.shuffle(bucket_keys)
            else:
                perm = torch.randperm(len(bucket_keys), generator=self.generator).tolist()
                bucket_keys = [bucket_keys[i] for i in perm]

        B = self.batch_size

        for key in bucket_keys:
            idxs = buckets[key]

            if self.shuffle:
                if self.generator is None:
                    random.shuffle(idxs)
                else:
                    perm = torch.randperm(len(idxs), generator=self.generator).tolist()
                    idxs = [idxs[i] for i in perm]

            # Full batches.
            n_full = len(idxs) // B
            for b in range(n_full):
                yield idxs[b * B : (b + 1) * B]

            # Remainder batch.
            rem = len(idxs) % B
            if rem and not self.drop_last:
                yield idxs[-rem:]

    def __len__(self) -> int:
        total = 0
        B = self.batch_size

        for n in self._bucket_lengths():
            total += (n // B) if self.drop_last else math.ceil(n / B)

        return total

    def _bucket_lengths(self) -> List[int]:
        counts: Dict[Tuple[int, int], int] = {}

        for idx in range(len(self.dataset)):
            Bh, Bw = tuple(self.dataset[idx]["grid_shape"])
            counts[(Bh, Bw)] = counts.get((Bh, Bw), 0) + 1

        return list(counts.values())


# ---------- Factories ----------
def create_dataloader(
    items: Sequence[Dict[str, Any]],
    collate_fn: Callable,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    bucket_by_grid: bool = False,
    transform: Optional[Callable] = None,
    seed: Optional[int] = 42,
) -> DataLoader:
    dataset = BlockItemDataset(items, transform=transform)

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    if bucket_by_grid:
        sampler = GridShapeBucketBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            generator=generator,
        )

        # Do NOT pass batch_size/shuffle/drop_last when using batch_sampler.
        loader = DataLoader(
            dataset,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            batch_sampler=sampler,
            worker_init_fn=_seed_worker if num_workers > 0 else None,
            generator=generator,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            drop_last=drop_last,
            worker_init_fn=_seed_worker if num_workers > 0 else None,
            generator=generator,
        )

    return loader


def create_dataloaders_train_val(
    train_items: Sequence[Dict[str, Any]],
    val_items: Sequence[Dict[str, Any]],
    test_items: Sequence[Dict[str, Any]],
    collate_fn: Callable,
    train_batch_size: int = 16,
    val_batch_size: int = 16,
    test_batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = False,
    bucket_by_grid: bool = True,
    seed: Optional[int] = 42,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    test_transform: Optional[Callable] = None,
    drop_last_train: bool = False,
    drop_last_val: bool = False,
    drop_last_test: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.

    Function name is kept as create_dataloaders_train_val for compatibility
    with the original code, but this version always expects all three splits:

        train_items, val_items, test_items

    and always returns:

        train_loader, val_loader, test_loader
    """
    train_loader = create_dataloader(
        train_items,
        collate_fn=collate_fn,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_train,
        bucket_by_grid=bucket_by_grid,
        transform=train_transform,
        seed=seed,
    )

    val_loader = create_dataloader(
        val_items,
        collate_fn=collate_fn,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_val,
        bucket_by_grid=bucket_by_grid,
        transform=val_transform,
        seed=seed,
    )

    test_loader = create_dataloader(
        test_items,
        collate_fn=collate_fn,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_test,
        bucket_by_grid=bucket_by_grid,
        transform=test_transform,
        seed=seed,
    )

    return train_loader, val_loader, test_loader