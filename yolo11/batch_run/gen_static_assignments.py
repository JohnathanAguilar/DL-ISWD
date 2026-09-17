"""Statically partition the remaining jobs across 7 GPU devices - no shared
queue, no lock file needed. This replaces the flock-based queue after
discovering flock does not provide real mutual exclusion on this network
filesystem (all 7 workers grabbed the same queue item simultaneously)."""
from pathlib import Path

COMPOSITE_DIR = Path("/workspace/data/bob/composite_grouped_yolo_splits")
HIGHPASS_DIR = Path("/workspace/data/bob/highpass_grouped_yolo_splits")

DATASETS = {
    "gradmag": (HIGHPASS_DIR, "gradient_magnitude_magma"),
    "ssha": (HIGHPASS_DIR, "normal_unfiltered_ssha_magma"),
    "rollmean": (HIGHPASS_DIR, "rolling_mean_highpass_magma"),
    "stepconv": (HIGHPASS_DIR, "stepped_convolution_highpass_magma"),
}

REMAINING = [
    ("gradmag", 456), ("gradmag", 789),
    ("ssha", 123), ("ssha", 456), ("ssha", 789),
    ("rollmean", 123), ("rollmean", 456), ("rollmean", 789),
    ("stepconv", 123), ("stepconv", 456), ("stepconv", 789),
]

NUM_DEVICES = 7
assignments = [[] for _ in range(NUM_DEVICES)]
for i, (short, seed) in enumerate(REMAINING):
    device = i % NUM_DEVICES
    base_dir, folder = DATASETS[short]
    data_yaml = base_dir / folder / "data.yaml"
    name = f"yolo11n_{short}_s{seed}"
    assignments[device].append((name, str(data_yaml), seed))

out_dir = Path("/workspace/isw-detector/yolo11/batch_run")
for device, jobs in enumerate(assignments):
    p = out_dir / f"device{device}.txt"
    lines = [f"{name}\t{data_yaml}\t{seed}" for name, data_yaml, seed in jobs]
    p.write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"device{device}: {len(jobs)} job(s) -> {[j[0] for j in jobs]}")
