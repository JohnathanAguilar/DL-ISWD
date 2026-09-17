"""Apply the SWOT L3 preprocessing (6 derived variables) + the divisor-10 crop
to whole cycle folders, in place.

This module is self-contained: the filter / crop / save implementations that
used to live in ``src/preprocessing/pre_processing.ipynb`` are defined here
directly. ``CONFIG`` holds every filter parameter, ``process_data_array``
computes the six derived channels for one file, ``crop_to_divisor`` trims each
spatial dimension to a multiple of 10, and ``save_dataset_safely`` does the
temp-file + atomic-replace write.

Per file, matching the original preprocess-then-crop order:
  1. compute the 6 arrays on the full-width swath (``process_data_array``),
  2. attach them + the ``preprocessing_*`` dataset attrs,
  3. crop num_lines/num_pixels to a multiple of 10 (``crop_to_divisor``): 69 -> 60,
  4. write once, temp-file + atomic replace (``save_dataset_safely``).

Example
-------
    cd cnn-transformer
    python src/preprocessing/preprocess_cycles.py --cycles 01 02 03 04

Run from the ``cnn-transformer/`` directory.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import xarray as xr
from scipy.ndimage import (
    gaussian_filter,
    uniform_filter,
    convolve,
    sobel,
)

_PREPROC_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _PREPROC_DIR.parent.parent  # cnn-transformer/


# ============================================================
# Configuration
# ============================================================

CONFIG = SimpleNamespace(
    data_dir=Path("../../../data/SWOT_L3_Full_Dataset/data"),

    base_var="ssha_unfiltered",
    quality_var="quality_flag",

    # Create a backup before modifying each NetCDF.
    # Backup name:
    # original_filename.backup.nc
    create_backups=False,

    # Replace preprocessing variables if they already exist.
    overwrite_existing_variables=True,

    # Quality flags considered valid.
    good_quality_flags=(0, 5, 30, 50),

    # Mask SSHA values outside this absolute range.
    max_absolute_ssha=10.0,

    # ========================================================
    # Filter parameters
    # ========================================================

    # Approximate SWOT pixel spacing.
    pixel_size_km=2.0,

    # Gaussian high-pass.
    gaussian_sigma=8,

    # FFT high-pass cutoff.
    #
    # Wavelengths larger than approximately this value are suppressed.
    fft_cutoff_wavelength_km=30.0,

    # ========================================================
    # Stepped-convolution parameters
    # ========================================================

    inner_radius_km=5.0,
    middle_radius_km=15.0,
    outer_radius_km=30.0,

    inner_weight=5.0,
    middle_weight=0.5,
    outer_weight=0.2,
)


# ============================================================
# Derived parameters
# ============================================================

# Rolling-mean high-pass window.
CONFIG.rolling_window = CONFIG.gaussian_sigma * 2

# Stepped-convolution radii converted to pixels.
CONFIG.inner_radius_px = int(
    np.ceil(CONFIG.inner_radius_km / CONFIG.pixel_size_km)
)

CONFIG.middle_radius_px = int(
    np.ceil(CONFIG.middle_radius_km / CONFIG.pixel_size_km)
)

CONFIG.outer_radius_px = int(
    np.ceil(CONFIG.outer_radius_km / CONFIG.pixel_size_km)
)

# Crop the two spatial dims to a multiple of this. num_lines is along-track
# (vertical, top/bottom), num_pixels is cross-track (horizontal, left/right).
CROP_DIVISOR = 10
LINE_DIM = "num_lines"
PIXEL_DIM = "num_pixels"

DERIVED_VARS = [
    "ssha_quality_masked",
    "rolling_mean_highpass",
    "gaussian_filter_highpass",
    "stepped_convolution_highpass",
    "fft_highpass",
    "gradient_magnitude",
]


# ============================================================
# Quality and value masking
# ============================================================

def apply_quality_flag_mask(ssha, quality_flag):
    """
    Keep quality_flag values 0, 5, 30, and 50.

    All other pixels become NaN.
    """
    ssha = np.asarray(ssha, dtype=np.float32).copy()
    quality_flag = np.asarray(quality_flag)

    good_quality = np.isin(
        quality_flag,
        CONFIG.good_quality_flags,
    )

    ssha[~good_quality] = np.nan

    return ssha


def mask_unrealistic_values(arr):
    """
    Mask SSHA values outside:
        [-CONFIG.max_absolute_ssha, CONFIG.max_absolute_ssha]
    """
    arr = np.asarray(arr, dtype=np.float32).copy()

    arr[np.abs(arr) > CONFIG.max_absolute_ssha] = np.nan

    return arr


# ============================================================
# NaN-aware smoothing filters
# ============================================================

def nan_uniform_filter(arr, window):
    """
    Apply a uniform filter while ignoring NaN pixels.
    """
    arr = np.asarray(arr, dtype=np.float32)

    valid = np.isfinite(arr)

    arr_filled = np.where(
        valid,
        arr,
        0.0,
    ).astype(np.float32)

    valid_weights = valid.astype(np.float32)

    filtered_values = uniform_filter(
        arr_filled,
        size=window,
        mode="nearest",
    )

    filtered_weights = uniform_filter(
        valid_weights,
        size=window,
        mode="nearest",
    )

    smooth = filtered_values / np.maximum(
        filtered_weights,
        1e-6,
    )

    smooth[filtered_weights <= 1e-6] = np.nan

    return smooth.astype(np.float32)


def nan_gaussian_filter(arr, sigma):
    """
    Apply a Gaussian filter while ignoring NaN pixels.
    """
    arr = np.asarray(arr, dtype=np.float32)

    valid = np.isfinite(arr)

    arr_filled = np.where(
        valid,
        arr,
        0.0,
    ).astype(np.float32)

    valid_weights = valid.astype(np.float32)

    filtered_values = gaussian_filter(
        arr_filled,
        sigma=sigma,
        mode="nearest",
    )

    filtered_weights = gaussian_filter(
        valid_weights,
        sigma=sigma,
        mode="nearest",
    )

    smooth = filtered_values / np.maximum(
        filtered_weights,
        1e-6,
    )

    smooth[filtered_weights <= 1e-6] = np.nan

    return smooth.astype(np.float32)


def build_stepped_kernel():
    """
    Build a normalized stepped radial convolution kernel.
    """
    kernel_radius = CONFIG.outer_radius_px

    y, x = np.ogrid[
        -kernel_radius:kernel_radius + 1,
        -kernel_radius:kernel_radius + 1,
    ]

    radius = np.sqrt(x**2 + y**2)

    kernel = np.zeros_like(
        radius,
        dtype=np.float32,
    )

    kernel[
        radius <= CONFIG.inner_radius_px
    ] = CONFIG.inner_weight

    kernel[
        (radius > CONFIG.inner_radius_px)
        & (radius <= CONFIG.middle_radius_px)
    ] = CONFIG.middle_weight

    kernel[
        (radius > CONFIG.middle_radius_px)
        & (radius <= CONFIG.outer_radius_px)
    ] = CONFIG.outer_weight

    kernel_sum = kernel.sum()

    if kernel_sum <= 0:
        raise ValueError(
            "The stepped-convolution kernel has a zero sum."
        )

    kernel /= kernel_sum

    return kernel.astype(np.float32)


STEPPED_KERNEL = build_stepped_kernel()


def nan_stepped_convolution_filter(arr, kernel):
    """
    Apply stepped convolution while ignoring NaN pixels.
    """
    arr = np.asarray(arr, dtype=np.float32)

    valid = np.isfinite(arr)

    arr_filled = np.where(
        valid,
        arr,
        0.0,
    ).astype(np.float32)

    valid_weights = valid.astype(np.float32)

    filtered_values = convolve(
        arr_filled,
        weights=kernel,
        mode="nearest",
    )

    filtered_weights = convolve(
        valid_weights,
        weights=kernel,
        mode="nearest",
    )

    smooth = filtered_values / np.maximum(
        filtered_weights,
        1e-6,
    )

    smooth[filtered_weights <= 1e-6] = np.nan

    return smooth.astype(np.float32)


# ============================================================
# High-pass filters
# ============================================================

def rolling_highpass(arr, window):
    """
    Rolling high-pass:

        original - rolling mean
    """
    arr = np.asarray(arr, dtype=np.float32)

    valid = np.isfinite(arr)

    smooth = nan_uniform_filter(
        arr,
        window=window,
    )

    highpass = arr - smooth
    highpass[~valid] = np.nan

    return highpass.astype(np.float32)


def gaussian_highpass(arr, sigma):
    """
    Gaussian high-pass:

        original - Gaussian smooth
    """
    arr = np.asarray(arr, dtype=np.float32)

    valid = np.isfinite(arr)

    smooth = nan_gaussian_filter(
        arr,
        sigma=sigma,
    )

    highpass = arr - smooth
    highpass[~valid] = np.nan

    return highpass.astype(np.float32)


def stepped_convolution_highpass(arr, kernel):
    """
    Stepped-convolution high-pass:

        original - stepped convolution smooth
    """
    arr = np.asarray(arr, dtype=np.float32)

    valid = np.isfinite(arr)

    smooth = nan_stepped_convolution_filter(
        arr,
        kernel=kernel,
    )

    highpass = arr - smooth
    highpass[~valid] = np.nan

    return highpass.astype(np.float32)


# ============================================================
# FFT high-pass
# ============================================================

def fill_nan_for_filtering(arr):
    """
    Fill NaNs with the array median before FFT or Sobel processing.

    Returns
    -------
    filled : np.ndarray
        Array with NaNs filled.
    valid_mask : np.ndarray
        Original finite-value mask.
    """
    arr = np.asarray(arr, dtype=np.float32)

    valid_mask = np.isfinite(arr)

    if not valid_mask.any():
        return arr.copy(), valid_mask

    fill_value = np.nanmedian(arr)

    filled = arr.copy()
    filled[~valid_mask] = fill_value

    return filled.astype(np.float32), valid_mask


def fft_highpass(
    arr,
    pixel_size_km=2.0,
    cutoff_wavelength_km=30.0,
):
    """
    Apply a smooth radial FFT high-pass filter.

    Large-scale structures with wavelengths greater than the cutoff
    are increasingly suppressed.
    """
    if pixel_size_km <= 0:
        raise ValueError(
            "pixel_size_km must be greater than zero."
        )

    if cutoff_wavelength_km <= 0:
        raise ValueError(
            "cutoff_wavelength_km must be greater than zero."
        )

    filled, valid_mask = fill_nan_for_filtering(arr)

    if not valid_mask.any():
        return np.full_like(
            filled,
            np.nan,
            dtype=np.float32,
        )

    height, width = filled.shape

    frequency_y = np.fft.fftfreq(
        height,
        d=pixel_size_km,
    )

    frequency_x = np.fft.fftfreq(
        width,
        d=pixel_size_km,
    )

    frequency_x_grid, frequency_y_grid = np.meshgrid(
        frequency_x,
        frequency_y,
    )

    radial_frequency = np.sqrt(
        frequency_x_grid**2
        + frequency_y_grid**2
    )

    cutoff_frequency = 1.0 / cutoff_wavelength_km

    # Smooth Gaussian-style high-pass mask.
    highpass_mask = 1.0 - np.exp(
        -(radial_frequency**2)
        / (2.0 * cutoff_frequency**2)
    )

    fft_image = np.fft.fft2(filled)

    filtered_fft = fft_image * highpass_mask

    filtered = np.real(
        np.fft.ifft2(filtered_fft)
    ).astype(np.float32)

    filtered[~valid_mask] = np.nan

    return filtered


# ============================================================
# Sobel gradient
# ============================================================

def sobel_derivative(
    arr,
    axis,
    pixel_size_km=2.0,
):
    """
    Calculate a Sobel derivative.

    The result is approximately in SSHA units per kilometer.
    """
    filled, valid_mask = fill_nan_for_filtering(arr)

    if not valid_mask.any():
        return np.full_like(
            filled,
            np.nan,
            dtype=np.float32,
        )

    derivative = sobel(
        filled,
        axis=axis,
        mode="nearest",
    ).astype(np.float32)

    # scipy.ndimage.sobel has an effective scale factor of 8
    # for a two-dimensional array.
    derivative /= 8.0 * pixel_size_km

    derivative[~valid_mask] = np.nan

    return derivative.astype(np.float32)


def gradient_magnitude(
    arr,
    pixel_size_km=2.0,
):
    """
    Calculate Sobel gradient magnitude:

        sqrt(gradient_x^2 + gradient_y^2)
    """
    gradient_x = sobel_derivative(
        arr,
        axis=1,
        pixel_size_km=pixel_size_km,
    )

    gradient_y = sobel_derivative(
        arr,
        axis=0,
        pixel_size_km=pixel_size_km,
    )

    gradient = np.sqrt(
        gradient_x**2 + gradient_y**2
    ).astype(np.float32)

    gradient[~np.isfinite(arr)] = np.nan

    return gradient


# ============================================================
# Process one 2D swath
# ============================================================

def process_2d_swath(ssha, quality_flag):
    """
    Apply quality masking and all preprocessing methods to one
    two-dimensional SWOT swath.
    """
    ssha = np.asarray(ssha, dtype=np.float32)
    quality_flag = np.asarray(quality_flag)

    if ssha.shape != quality_flag.shape:
        raise ValueError(
            f"Shape mismatch: SSHA={ssha.shape}, "
            f"quality_flag={quality_flag.shape}"
        )

    masked_ssha = apply_quality_flag_mask(
        ssha,
        quality_flag,
    )

    masked_ssha = mask_unrealistic_values(
        masked_ssha
    )

    if not np.isfinite(masked_ssha).any():
        raise ValueError(
            "All SSHA pixels are NaN after quality masking."
        )

    results = {
        "ssha_quality_masked": masked_ssha.astype(
            np.float32
        ),

        "rolling_mean_highpass": rolling_highpass(
            masked_ssha,
            window=CONFIG.rolling_window,
        ),

        "gaussian_filter_highpass": gaussian_highpass(
            masked_ssha,
            sigma=CONFIG.gaussian_sigma,
        ),

        "stepped_convolution_highpass": (
            stepped_convolution_highpass(
                masked_ssha,
                kernel=STEPPED_KERNEL,
            )
        ),

        "fft_highpass": fft_highpass(
            masked_ssha,
            pixel_size_km=CONFIG.pixel_size_km,
            cutoff_wavelength_km=(
                CONFIG.fft_cutoff_wavelength_km
            ),
        ),

        "gradient_magnitude": gradient_magnitude(
            masked_ssha,
            pixel_size_km=CONFIG.pixel_size_km,
        ),
    }

    return results


# ============================================================
# Process DataArrays while retaining original dimensions
# ============================================================

def get_spatial_dimensions(data_array):
    """
    Identify the two non-singleton spatial dimensions.

    Example supported dimensions:
        (num_lines, num_pixels)
        (file, num_lines, num_pixels)

    In the second example, file must normally have size 1.
    """
    spatial_dimensions = [
        dimension
        for dimension in data_array.dims
        if data_array.sizes[dimension] > 1
    ]

    if len(spatial_dimensions) != 2:
        raise ValueError(
            f"Expected two non-singleton spatial dimensions for "
            f"{data_array.name!r}, but found "
            f"{spatial_dimensions}. Full shape: "
            f"{dict(data_array.sizes)}"
        )

    return tuple(spatial_dimensions)


def process_data_array(base_da, quality_da):
    """
    Process the complete SSHA DataArray while preserving its
    original dimensions.
    """
    # Ensure quality dimensions use the same ordering as SSHA.
    if quality_da.dims != base_da.dims:
        quality_da = quality_da.transpose(
            *base_da.dims
        )

    if quality_da.shape != base_da.shape:
        raise ValueError(
            f"{CONFIG.base_var} shape {base_da.shape} does not match "
            f"{CONFIG.quality_var} shape {quality_da.shape}."
        )

    spatial_dimensions = get_spatial_dimensions(
        base_da
    )

    extra_dimensions = [
        dimension
        for dimension in base_da.dims
        if dimension not in spatial_dimensions
    ]

    output_names = [
        "ssha_quality_masked",
        "rolling_mean_highpass",
        "gaussian_filter_highpass",
        "stepped_convolution_highpass",
        "fft_highpass",
        "gradient_magnitude",
    ]

    outputs = {
        output_name: np.full(
            base_da.shape,
            np.nan,
            dtype=np.float32,
        )
        for output_name in output_names
    }

    # No singleton or additional dimensions.
    if not extra_dimensions:
        calculated = process_2d_swath(
            base_da.values,
            quality_da.values,
        )

        for output_name, output_array in calculated.items():
            outputs[output_name][...] = output_array

        return outputs

    dimension_to_axis = {
        dimension: axis
        for axis, dimension in enumerate(base_da.dims)
    }

    extra_shapes = tuple(
        base_da.sizes[dimension]
        for dimension in extra_dimensions
    )

    for extra_index in np.ndindex(*extra_shapes):
        numpy_index = [slice(None)] * base_da.ndim

        for dimension, index_value in zip(
            extra_dimensions,
            extra_index,
        ):
            axis = dimension_to_axis[dimension]
            numpy_index[axis] = index_value

        numpy_index = tuple(numpy_index)

        ssha_slice = np.asarray(
            base_da.values[numpy_index],
            dtype=np.float32,
        )

        quality_slice = np.asarray(
            quality_da.values[numpy_index]
        )

        if ssha_slice.ndim != 2:
            raise ValueError(
                f"Expected a 2D SSHA slice, but got "
                f"shape {ssha_slice.shape}."
            )

        calculated = process_2d_swath(
            ssha_slice,
            quality_slice,
        )

        for output_name, output_array in calculated.items():
            outputs[output_name][numpy_index] = output_array

    return outputs


# ============================================================
# Variable metadata
# ============================================================

def get_variable_attributes(base_da):
    """
    Create metadata for the generated NetCDF variables.
    """
    base_units = base_da.attrs.get("units", "")

    gradient_units = (
        f"{base_units} km-1"
        if base_units
        else "SSHA units km-1"
    )

    common_attributes = {
        "source_variable": CONFIG.base_var,
        "quality_flag_variable": CONFIG.quality_var,
        "valid_quality_flags": ", ".join(
            str(flag) for flag in CONFIG.good_quality_flags
        ),
        "invalid_value_mask": (
            f"abs({CONFIG.base_var}) > {CONFIG.max_absolute_ssha}"
        ),
    }

    return {
        "ssha_quality_masked": {
            **common_attributes,
            "long_name": "Quality-masked SSHA",
            "description": (
                "Original SSHA after quality-flag and "
                "unrealistic-value masking."
            ),
            "units": base_units,
        },

        "rolling_mean_highpass": {
            **common_attributes,
            "long_name": (
                "Rolling-mean high-pass filtered SSHA"
            ),
            "description": (
                "Quality-masked SSHA minus a NaN-aware "
                "rolling mean."
            ),
            "rolling_window_pixels": CONFIG.rolling_window,
            "rolling_window_km": (
                CONFIG.rolling_window * CONFIG.pixel_size_km
            ),
            "pixel_size_km": CONFIG.pixel_size_km,
            "units": base_units,
        },

        "gaussian_filter_highpass": {
            **common_attributes,
            "long_name": (
                "Gaussian high-pass filtered SSHA"
            ),
            "description": (
                "Quality-masked SSHA minus a NaN-aware "
                "Gaussian-smoothed field."
            ),
            "gaussian_sigma_pixels": CONFIG.gaussian_sigma,
            "gaussian_sigma_km": (
                CONFIG.gaussian_sigma * CONFIG.pixel_size_km
            ),
            "pixel_size_km": CONFIG.pixel_size_km,
            "units": base_units,
        },

        "stepped_convolution_highpass": {
            **common_attributes,
            "long_name": (
                "Stepped-convolution high-pass filtered SSHA"
            ),
            "description": (
                "Quality-masked SSHA minus a NaN-aware "
                "stepped radial convolution."
            ),
            "inner_radius_km": CONFIG.inner_radius_km,
            "middle_radius_km": CONFIG.middle_radius_km,
            "outer_radius_km": CONFIG.outer_radius_km,
            "inner_weight": CONFIG.inner_weight,
            "middle_weight": CONFIG.middle_weight,
            "outer_weight": CONFIG.outer_weight,
            "pixel_size_km": CONFIG.pixel_size_km,
            "units": base_units,
        },

        "fft_highpass": {
            **common_attributes,
            "long_name": "FFT high-pass filtered SSHA",
            "description": (
                "Quality-masked SSHA filtered using a smooth "
                "radial FFT high-pass mask."
            ),
            "cutoff_wavelength_km": (
                CONFIG.fft_cutoff_wavelength_km
            ),
            "cutoff_frequency_cycles_per_km": (
                1.0 / CONFIG.fft_cutoff_wavelength_km
            ),
            "pixel_size_km": CONFIG.pixel_size_km,
            "units": base_units,
        },

        "gradient_magnitude": {
            **common_attributes,
            "long_name": "Sobel SSHA gradient magnitude",
            "description": (
                "Magnitude of the horizontal and vertical "
                "Sobel derivatives."
            ),
            "pixel_size_km": CONFIG.pixel_size_km,
            "units": gradient_units,
        },
    }


# ============================================================
# Crop pixels so width/height are divisible by CROP_DIVISOR
# ============================================================
#
# Opens a dataset and trims num_pixels (left/right) and num_lines
# (top/bottom) equally from both edges so each is a multiple of
# CROP_DIVISOR. Dataset.isel crops every variable and coordinate that
# shares either spatial dimension in one step, so the derived
# preprocessing variables, lat/lon, and raw SSHA stay in sync;
# variables on unrelated dimensions (e.g. num_nadir) are left
# untouched. Datasets already divisible by CROP_DIVISOR are returned
# unmodified.

def crop_to_divisor(dataset, divisor=CROP_DIVISOR):
    """
    Crop num_lines and num_pixels equally from both edges so each is
    divisible by `divisor`.

    Returns (cropped_dataset, crop_info). crop_info is None when the
    dataset was already divisible on both axes, in which case
    cropped_dataset is the input dataset, unmodified.
    """
    n_lines = dataset.sizes[LINE_DIM]
    n_pixels = dataset.sizes[PIXEL_DIM]

    line_remainder = n_lines % divisor
    pixel_remainder = n_pixels % divisor

    if line_remainder == 0 and pixel_remainder == 0:
        return dataset, None

    crop_top = line_remainder // 2
    crop_bottom = line_remainder - crop_top

    crop_left = pixel_remainder // 2
    crop_right = pixel_remainder - crop_left

    new_n_lines = n_lines - line_remainder
    new_n_pixels = n_pixels - pixel_remainder

    if new_n_lines <= 0 or new_n_pixels <= 0:
        raise ValueError(
            f"Cropping to a multiple of {divisor} would leave a "
            f"non-positive size: {LINE_DIM} {n_lines} -> {new_n_lines}, "
            f"{PIXEL_DIM} {n_pixels} -> {new_n_pixels}."
        )

    cropped = dataset.isel({
        LINE_DIM: slice(crop_top, n_lines - crop_bottom),
        PIXEL_DIM: slice(crop_left, n_pixels - crop_right),
    })

    crop_info = {
        "lines": (n_lines, crop_top, crop_bottom, new_n_lines),
        "pixels": (n_pixels, crop_left, crop_right, new_n_pixels),
    }

    return cropped, crop_info


# ============================================================
# Save modified NetCDF
# ============================================================

def save_dataset_safely(dataset, nc_file, generated_variables):
    """
    Write to a temporary file first.

    The original file is replaced only after the temporary output
    is successfully written and verified.

    generated_variables lists the variables that get the forced
    float32 / zlib / NaN-fill encoding and are checked for after the
    write; pass [] to write with no forced encoding (so integer and
    datetime variables keep their native dtype) and use only the
    safe temp-file-then-atomic-replace behavior.
    """
    nc_file = Path(nc_file)

    temporary_handle = tempfile.NamedTemporaryFile(
        prefix=f"{nc_file.stem}_temporary_",
        suffix=".nc",
        dir=nc_file.parent,
        delete=False,
    )

    temporary_path = Path(temporary_handle.name)
    temporary_handle.close()

    encoding = {
        variable_name: {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
            "_FillValue": np.float32(np.nan),
        }
        for variable_name in generated_variables
        if variable_name in dataset
    }

    try:
        try:
            dataset.to_netcdf(
                temporary_path,
                mode="w",
                engine="netcdf4",
                encoding=encoding,
            )

        except (ModuleNotFoundError, ImportError, ValueError):
            # Fall back to the default xarray engine if netCDF4
            # is not installed.
            print(
                "    netCDF4 compression unavailable. "
                "Using the default xarray writer."
            )

            dataset.to_netcdf(temporary_path, mode="w")

        # Verify the temporary file is readable and, if any variables
        # were explicitly encoded above, that they made it through.
        with xr.open_dataset(temporary_path) as check_ds:
            missing_variables = [
                variable_name
                for variable_name in generated_variables
                if variable_name not in check_ds
            ]

            if missing_variables:
                raise RuntimeError(
                    "Temporary NetCDF is missing variables: "
                    f"{missing_variables}"
                )

        os.replace(temporary_path, nc_file)

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()

        raise


# ============================================================
# Process one NetCDF file
# ============================================================

def preprocess_file(nc_file: Path) -> dict:
    """Add the derived variables, crop to a multiple of 10, write back in place."""
    with xr.open_dataset(nc_file, decode_cf=True, mask_and_scale=True) as opened:
        dataset = opened.load()

    if CONFIG.base_var not in dataset:
        raise KeyError(f"missing variable: {CONFIG.base_var}")
    if CONFIG.quality_var not in dataset:
        raise KeyError(f"missing variable: {CONFIG.quality_var}")

    base_da = dataset[CONFIG.base_var]
    quality_da = dataset[CONFIG.quality_var]

    results = process_data_array(base_da, quality_da)
    attributes = get_variable_attributes(base_da)

    coordinates = {
        dim: base_da.coords[dim] for dim in base_da.dims if dim in base_da.coords
    }
    for name, array in results.items():
        if name in dataset:
            dataset = dataset.drop_vars(name)
        dataset[name] = xr.DataArray(
            data=array, dims=base_da.dims, coords=coordinates, attrs=attributes[name],
        )

    dataset.attrs["preprocessing_source_variable"] = CONFIG.base_var
    dataset.attrs["preprocessing_quality_variable"] = CONFIG.quality_var
    dataset.attrs["preprocessing_pixel_size_km"] = CONFIG.pixel_size_km
    dataset.attrs["preprocessing_gaussian_sigma_pixels"] = CONFIG.gaussian_sigma
    dataset.attrs["preprocessing_rolling_window_pixels"] = CONFIG.rolling_window
    dataset.attrs["preprocessing_fft_cutoff_wavelength_km"] = CONFIG.fft_cutoff_wavelength_km

    cropped, crop_info = crop_to_divisor(dataset, divisor=CROP_DIVISOR)
    save_dataset_safely(cropped, nc_file, generated_variables=DERIVED_VARS)

    dataset.close()
    return {
        "sizes": dict(cropped.sizes),
        "crop": crop_info,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cycles", nargs="+", default=["01", "02", "03", "04"],
                   help="Cycle folder suffixes under --data-root (e.g. 01 02 03 04).")
    p.add_argument("--data-root", type=Path,
                   default=_MODEL_DIR.parent / "data/SWOT_L3_Full_Dataset",
                   help="Directory holding the cycle_<N> folders.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N files per cycle (debug).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(f"stepped kernel {STEPPED_KERNEL.shape}, cropping to /{CROP_DIVISOR}")

    grand_ok = grand_fail = 0

    for cycle in args.cycles:
        cdir = args.data_root / f"cycle_{cycle}"
        files = sorted(cdir.glob("*.nc"))
        if not files:
            print(f"\ncycle_{cycle}: no .nc files in {cdir.resolve()} -- skipped")
            continue
        if args.limit:
            files = files[: args.limit]

        print(f"\n{'=' * 60}\ncycle_{cycle}: {len(files)} files  ({cdir.resolve()})\n{'=' * 60}")
        n_ok = n_fail = 0

        for i, nc_file in enumerate(files, start=1):
            try:
                info = preprocess_file(nc_file)
                n_ok += 1
                if i % 25 == 0 or i == len(files):
                    px = info["crop"]["pixels"] if info["crop"] else None
                    print(f"[{i}/{len(files)}] {nc_file.name}: "
                          f"sizes={info['sizes']} pixels_crop={px}")
            except Exception as exc:  # noqa: BLE001 - report, leave raw file, continue
                n_fail += 1
                print(f"[{i}/{len(files)}] {nc_file.name}: FAILED: {exc}")
                traceback.print_exc()

        print(f"cycle_{cycle}: ok={n_ok} failed={n_fail}")
        grand_ok += n_ok
        grand_fail += n_fail

    print(f"\n{'=' * 60}\nall cycles: ok={grand_ok} failed={grand_fail}\n{'=' * 60}")
    return 1 if grand_fail else 0


if __name__ == "__main__":
    sys.exit(main())
