"""Statically partition all (run, split) cnn-transformer inference jobs
across the GPU devices NOT currently running the yolo11 custom-augmentation
training job (device 0) -- no shared queue, no lock file. Same design as
yolo11/batch_run_inference/gen_inference_assignments.py."""

from pathlib import Path

CNNTRANS_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = CNNTRANS_DIR / "runs"
OUT_DIR = Path(__file__).resolve().parent

SPLITS = ["test", "valid"]
DEVICES = [1, 2, 3, 4]  # device 0 is busy running yolo11n_gauss_customaug_s123

run_names = sorted(p.name for p in RUNS_DIR.iterdir() if (p / "best.pt").exists())

jobs = [(run_name, split) for run_name in run_names for split in SPLITS]

assignments = {d: [] for d in DEVICES}
for i, job in enumerate(jobs):
    device = DEVICES[i % len(DEVICES)]
    assignments[device].append(job)

for device, device_jobs in assignments.items():
    p = OUT_DIR / f"device{device}.txt"
    lines = [f"{run_name}\t{split}" for run_name, split in device_jobs]
    p.write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"device{device}: {len(device_jobs)} job(s) -> {[f'{r}/{s}' for r, s in device_jobs]}")

print(f"\ntotal runs found: {len(run_names)}, total jobs: {len(jobs)}")
