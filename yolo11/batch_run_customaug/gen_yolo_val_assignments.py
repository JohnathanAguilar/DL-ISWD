"""Statically partition all (run, split) yolo-val jobs for the augmented
sweep across the free GPU devices -- no shared queue, no lock file. Same
design as ../yolo_val/gen_yolo_val_assignments.py, pointed at
runs_customaug/ instead of runs/."""

from pathlib import Path

YOLO11_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = YOLO11_DIR / "runs_customaug"
OUT_DIR = Path(__file__).resolve().parent

SPLITS = ["test", "val"]
NUM_DEVICES = 7  # all 7 MIG slices -- see worker_static.sh for why this isn't 5

run_names = sorted(p.name for p in RUNS_DIR.iterdir() if (p / "weights" / "best.pt").exists())

jobs = [(run_name, split) for run_name in run_names for split in SPLITS]

assignments = [[] for _ in range(NUM_DEVICES)]
for i, job in enumerate(jobs):
    assignments[i % NUM_DEVICES].append(job)

for device, device_jobs in enumerate(assignments):
    p = OUT_DIR / f"device{device}_val.txt"
    lines = [f"{run_name}\t{split}" for run_name, split in device_jobs]
    p.write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"device{device}: {len(device_jobs)} job(s) -> {[f'{r}/{s}' for r, s in device_jobs]}")

print(f"\ntotal runs found: {len(run_names)}, total jobs: {len(jobs)}")
