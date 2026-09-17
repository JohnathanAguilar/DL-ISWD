"""Statically partition the 6 (variant, seed) training jobs for the new
yolo11s frgb/gauss models across the free GPU devices -- no shared queue, no
lock file (same design as ../batch_run_customaug/gen_static_assignments.py).

README Step 4: train `frgb` and `gauss` again at the small (`s`) model size,
3 seeds each, using the exact same datasets/seeds as the existing
yolo11n_frgb_*/yolo11n_gauss_* runs, no augmentation (same offline-augmented
convention as the original 30), writing into the default runs/ dir.
"""
from pathlib import Path

COMPOSITE_DIR = Path("/workspace/data/bob/composite_grouped_yolo_splits")
HIGHPASS_DIR = Path("/workspace/data/bob/highpass_grouped_yolo_splits")

DATASETS = {
    "frgb": (COMPOSITE_DIR, "scientific_rgb_composite"),
    "gauss": (HIGHPASS_DIR, "gaussian_filter_highpass_magma"),
}

SEEDS = [123, 456, 789]
NUM_DEVICES = 6  # only 6 jobs -- one per device, no queueing needed

jobs = []
for short, (base_dir, folder) in DATASETS.items():
    data_yaml = base_dir / folder / "data.yaml"
    for seed in SEEDS:
        name = f"yolo11s_{short}_s{seed}"
        jobs.append((name, str(data_yaml), seed))

out_dir = Path(__file__).resolve().parent
assignments = [[] for _ in range(NUM_DEVICES)]
for i, job in enumerate(jobs):
    assignments[i % NUM_DEVICES].append(job)

for device, device_jobs in enumerate(assignments):
    p = out_dir / f"device{device}.txt"
    lines = [f"{name}\t{data_yaml}\t{seed}" for name, data_yaml, seed in device_jobs]
    p.write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"device{device}: {len(device_jobs)} job(s) -> {[j[0] for j in device_jobs]}")

print(f"\ntotal jobs: {len(jobs)}")
