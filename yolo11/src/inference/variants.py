"""Maps each of the 10 dataset-variant abbreviations (as used in run names
like yolo11n_frgb_s123) to the detector family + constructor kwargs needed
to run inference for it."""

VARIANTS = {
    "frgb": ("composite", {"zero_channel": None}),
    "nograd": ("composite", {"zero_channel": "B"}),
    "nohp": ("composite", {"zero_channel": "G"}),
    "nossha": ("composite", {"zero_channel": "R"}),
    "gauss": ("highpass", {"source_var": "gaussian_filter_highpass"}),
    "fft": ("highpass", {"source_var": "fft_highpass"}),
    "gradmag": ("highpass", {"source_var": "gradient_magnitude"}),
    "ssha": ("highpass", {"source_var": "ssha_quality_masked"}),
    "rollmean": ("highpass", {"source_var": "rolling_mean_highpass"}),
    "stepconv": ("highpass", {"source_var": "stepped_convolution_highpass"}),
}


def parse_run_name(run_name):
    """Parse 'yolo11n_frgb_s123' -> ('frgb', 123). Raises ValueError if the
    variant suffix isn't recognized."""
    parts = run_name.split("_")
    if len(parts) < 3 or not parts[-1].startswith("s"):
        raise ValueError(f"Run name {run_name!r} doesn't match '<model>_<variant>_s<seed>'")

    seed = int(parts[-1][1:])
    variant = "_".join(parts[1:-1])

    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r} parsed from run name {run_name!r}")

    return variant, seed


def build_detector(variant, weights_path, device="auto"):
    family, kwargs = VARIANTS[variant]

    if family == "composite":
        from .rgb_composite_yolo_detector import RGBCompositeYOLODetector

        return RGBCompositeYOLODetector(weights_path, device=device, **kwargs)
    elif family == "highpass":
        from .highpass_yolo_detector import HighpassYOLODetector

        return HighpassYOLODetector(weights_path, device=device, **kwargs)
    else:
        raise ValueError(f"Unknown family {family!r}")
