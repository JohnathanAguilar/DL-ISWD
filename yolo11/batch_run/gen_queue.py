"""Generate the job queue: 10 datasets x 3 seeds, minus jobs already running."""
from pathlib import Path

COMPOSITE_DIR = Path("/workspace/data/bob/composite_grouped_yolo_splits")
HIGHPASS_DIR = Path("/workspace/data/bob/highpass_grouped_yolo_splits")

# short_name -> (base_dir, folder_name)
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

# Jobs already running/completed outside the queue (name -> reason skipped).
SKIP = {"yolo11n_frgb_s456"}

lines = []
for short, (base_dir, folder) in DATASETS.items():
    data_yaml = base_dir / folder / "data.yaml"
    for seed in SEEDS:
        name = f"yolo11n_{short}_s{seed}"
        if name in SKIP:
            continue
        lines.append(f"{name}\t{data_yaml}\t{seed}")

out = Path("/workspace/isw-detector/yolo11/batch_run/queue.txt")
out.write_text("\n".join(lines) + "\n")
print(f"Wrote {len(lines)} jobs to {out}")
