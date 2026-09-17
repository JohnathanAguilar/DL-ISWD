"""Statically partition all 30 (variant, seed) training jobs for the
augmented sweep across the free GPU devices -- no shared queue, no lock file
(same design as ../batch_run/gen_static_assignments.py, adopted after
discovering flock does not provide real mutual exclusion on this network
filesystem).

This sweep retrains every one of the original 30 yolo11n models with the
temp-test augmentation (fliplr/flipud/translate/scale), at the same 100
epochs, writing into runs_customaug/ (a sister of runs/) instead of runs/
itself -- the original 30 models are never touched.
"""
from pathlib import Path

COMPOSITE_DIR = Path("/workspace/data/bob/composite_grouped_yolo_splits")
HIGHPASS_DIR = Path("/workspace/data/bob/highpass_grouped_yolo_splits")

# short_name -> (base_dir, folder_name) -- same table as ../batch_run/gen_queue.py
DATASETS = {
    "frgb": (COMPOSITE_DIR, "scientific_rgb_composite"),
    "nograd": (COMPOSITE_DIR, "no_gradient_composite"),
    "nohp": (COMPOSITE_DIR, "no_high_pass_composite"),
    "nossha": (COMPOSITE_DIR, "no_ssha_composite"),
    "fft": (HIGHPASS_DIR, "fft_highpass_magma"),
    "gauss": (HIGHPASS_DIR, "gaussian_filter_highpass_magma"),
    "gradmag": (HIGHPASS_DIR, "gradient_magnitude_magma"),
    "ssha": (HIGHPASS_DIR, "normal_unfiltered_ssha_magma"),
    "rollmean": (HIGHPASS_DIR, "rolling_mean_highpass_magma"),
    "stepconv": (HIGHPASS_DIR, "stepped_convolution_highpass_magma"),
}

SEEDS = [123, 456, 789]
NUM_DEVICES = 7  # all 7 MIG slices (2 GPUs have 2 slices each; CUDA under-reports
                 # this as 5 devices unless CUDA_VISIBLE_DEVICES lists all 7 UUIDs
                 # explicitly -- see worker_static.sh)

jobs = []
for short, (base_dir, folder) in DATASETS.items():
    data_yaml = base_dir / folder / "data.yaml"
    for seed in SEEDS:
        name = f"yolo11n_{short}_s{seed}"
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
