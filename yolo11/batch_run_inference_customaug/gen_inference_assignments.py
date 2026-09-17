"""Statically partition the (run, split) inference jobs for the 6 augmented
frgb/gauss runs across GPU devices -- no shared queue, no lock file. Same
design as ../batch_run_inference/gen_inference_assignments.py, but pointed at
runs_customaug/ and filtered to only the frgb and gauss variants (the two
best performers in the augmented sweep's Ultralytics eval)."""

from pathlib import Path

YOLO11_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = YOLO11_DIR / "runs_customaug"
OUT_DIR = Path(__file__).resolve().parent

VARIANTS = {"frgb", "gauss"}
SPLITS = ["test", "valid"]
NUM_DEVICES = 6


def variant_of(run_name):
    # yolo11n_<variant>_s<seed> -> <variant>
    parts = run_name.split("_")
    return "_".join(parts[1:-1])


run_names = sorted(
    p.name
    for p in RUNS_DIR.iterdir()
    if (p / "weights" / "best.pt").exists() and variant_of(p.name) in VARIANTS
)

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
