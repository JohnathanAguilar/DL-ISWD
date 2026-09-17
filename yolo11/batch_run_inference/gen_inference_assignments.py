"""Statically partition all (run, split) inference jobs across 7 GPU devices
-- no shared queue, no lock file. Same design as
../batch_run/gen_static_assignments.py, adopted after discovering that flock
does not provide real mutual exclusion on this network filesystem."""

from pathlib import Path

YOLO11_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = YOLO11_DIR / "runs"
OUT_DIR = Path(__file__).resolve().parent

SPLITS = ["test", "valid"]
NUM_DEVICES = 7

run_names = sorted(p.name for p in RUNS_DIR.iterdir() if (p / "weights" / "best.pt").exists())

jobs = [(run_name, split) for run_name in run_names for split in SPLITS]

assignments = [[] for _ in range(NUM_DEVICES)]
for i, job in enumerate(jobs):
    assignments[i % NUM_DEVICES].append(job)

for device, device_jobs in enumerate(assignments):
    p = OUT_DIR / f"device{device}.txt"
    lines = [f"{run_name}\t{split}" for run_name, split in device_jobs]
    p.write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"device{device}: {len(device_jobs)} job(s) -> {[f'{r}/{s}' for r, s in device_jobs]}")

print(f"\ntotal runs found: {len(run_names)}, total jobs: {len(jobs)}")
