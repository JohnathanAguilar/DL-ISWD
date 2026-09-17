import os
import glob
from typing import List, Dict, Tuple, Sequence, Optional

import numpy as np
import xarray as xr


def _ensure_block_size(block_size) -> Tuple[int, int]:
    """
    Normalize block_size to (bh, bw).
    """
    if isinstance(block_size, int):
        return block_size, block_size

    if isinstance(block_size, (tuple, list)) and len(block_size) == 2:
        return int(block_size[0]), int(block_size[1])

    raise ValueError("block_size must be an int or a (bh, bw) tuple")


def _majority_block_labels(
    mask: np.ndarray,
    bh: int,
    bw: int,
):
    """
    Compute block-wise binary labels using majority vote over each tile.

    Parameters
    ----------
    mask : np.ndarray
        Binary or thresholdable mask with shape [H, W].
    bh : int
        Block height.
    bw : int
        Block width.

    Returns
    -------
    labels : np.ndarray
        Block labels with shape [Bh, Bw] and dtype uint8.
    valid : np.ndarray
        Boolean validity mask with shape [Bh, Bw].
    crop_h : int
        Cropped image height.
    crop_w : int
        Cropped image width.
    """
    H, W = mask.shape

    mask01 = (mask >= 0.5).astype(np.uint8)

    Bh = H // bh
    Bw = W // bw

    if Bh == 0 or Bw == 0:
        raise ValueError(
            f"Block size ({bh}, {bw}) is larger than mask shape ({H}, {W})."
        )

    crop_h = Bh * bh
    crop_w = Bw * bw

    mask_c = mask01[:crop_h, :crop_w]

    # [Bh, bh, Bw, bw]
    tiles = mask_c.reshape(Bh, bh, Bw, bw)

    # Sum pixels inside each block.
    sums = tiles.sum(axis=(1, 3))

    # Ties are labeled as positive.
    threshold = (bh * bw) / 2.0
    labels = (sums >= threshold).astype(np.uint8)

    # All blocks are complete because ragged edges were removed.
    valid = np.ones_like(labels, dtype=bool)

    return labels, valid, crop_h, crop_w


def _find_nc_files(directory: str, split_name: str) -> List[str]:
    """
    Find all NetCDF files inside a directory.
    """
    if not os.path.isdir(directory):
        raise NotADirectoryError(
            f"{split_name} directory does not exist or is not a directory: "
            f"{directory}"
        )

    files = sorted(glob.glob(os.path.join(directory, "*.nc")))

    if not files:
        raise FileNotFoundError(
            f"No .nc files found in the {split_name} directory: {directory}"
        )

    return files


def _load_nc_items(
    files: Sequence[str],
    channel_var_names: Sequence[str],
    mask_var_name: str,
    bh: int,
    bw: int,
) -> List[Dict]:
    """
    Load a list of NetCDF files and convert them into dataset items.
    """
    items: List[Dict] = []

    for path in files:
        filename = os.path.basename(path)

        try:
            with xr.open_dataset(path) as ds:
                channel_raws = [
                    np.asarray(ds[channel_name].values)
                    for channel_name in channel_var_names
                ]

                m_raw = np.asarray(ds[mask_var_name].values)

        except KeyError as error:
            raise KeyError(
                f"Missing variable in {filename}: {error}. "
                f"Expected channels {channel_var_names} "
                f"and mask '{mask_var_name}'."
            ) from error

        except Exception as error:
            raise RuntimeError(
                f"Failed to read {path}: {error}"
            ) from error

        # -------------------------------------------------------------
        # Validate dimensions
        # -------------------------------------------------------------
        for channel_name, channel_raw in zip(
            channel_var_names,
            channel_raws,
        ):
            if channel_raw.ndim != 2:
                raise ValueError(
                    f"Channel '{channel_name}' in {filename} must be 2-D; "
                    f"got shape {channel_raw.shape}"
                )

        if m_raw.ndim != 2:
            raise ValueError(
                f"Mask '{mask_var_name}' in {filename} must be 2-D; "
                f"got shape {m_raw.shape}"
            )

        # -------------------------------------------------------------
        # Validate spatial shapes
        # -------------------------------------------------------------
        ref_shape = channel_raws[0].shape

        bad_channel_shapes = [
            f"{channel_name}{channel_raw.shape}"
            for channel_name, channel_raw in zip(
                channel_var_names,
                channel_raws,
            )
            if channel_raw.shape != ref_shape
        ]

        if bad_channel_shapes or m_raw.shape != ref_shape:
            all_channel_shapes = ", ".join(
                f"{channel_name}{channel_raw.shape}"
                for channel_name, channel_raw in zip(
                    channel_var_names,
                    channel_raws,
                )
            )

            raise ValueError(
                f"Shape mismatch in {filename}: "
                f"{all_channel_shapes}, "
                f"{mask_var_name}{m_raw.shape}"
            )

        # -------------------------------------------------------------
        # Create block labels
        # -------------------------------------------------------------
        labels, block_valid, crop_h, crop_w = (
            _majority_block_labels(
                mask=m_raw,
                bh=bh,
                bw=bw,
            )
        )

        # -------------------------------------------------------------
        # Clean and stack input channels
        # -------------------------------------------------------------
        channels = [
            np.nan_to_num(
                channel_raw[:crop_h, :crop_w],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32, copy=False)
            for channel_raw in channel_raws
        ]

        image = np.stack(channels, axis=0)

        item = {
            "image": image,
            "labels": labels,
            "valid": block_valid,
            "id": filename,
            "block_size": (bh, bw),
            "grid_shape": labels.shape,
        }

        items.append(item)

    return items


def load_nc_item(
    nc_path: str,
    channel_var_names: Sequence[str],
    mask_var_name: str,
    block_size,
) -> Dict:
    """
    Load a single NetCDF file into one dataset item.

    Convenience wrapper around `_load_nc_items` for callers that want to
    inspect or visualize one specific pass (e.g. a qualitative figure)
    rather than build a full train/valid/test split.
    """
    bh, bw = _ensure_block_size(block_size)
    items = _load_nc_items([nc_path], channel_var_names, mask_var_name, bh, bw)
    return items[0]


def make_nc_block_dataset(
    nc_dir: Optional[str] = None,
    channel_var_names: Sequence[str] = (),
    mask_var_name: str = "",
    block_size=None,
    valid_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    train_dir: Optional[str] = None,
    valid_dir: Optional[str] = None,
    test_dir: Optional[str] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Build train, validation, and test datasets from NetCDF files.

    The function supports two mutually exclusive modes.

    Mode 1: Automatic splitting
    ---------------------------
    Provide `nc_dir`.

    All NetCDF files in that directory are shuffled and divided into
    train, validation, and test sets according to `valid_ratio` and
    `test_ratio`.

    Example:

        train_items, valid_items, test_items = make_nc_block_dataset(
            nc_dir="data/processed/block_labeled_data",
            channel_var_names=("ssha_unfiltered",),
            mask_var_name="label",
            block_size=32,
            valid_ratio=0.2,
            test_ratio=0.2,
            seed=42,
        )

    Mode 2: Existing split directories
    ----------------------------------
    Provide all three of:

        train_dir
        valid_dir
        test_dir

    Every NetCDF file in each directory is loaded directly into its
    corresponding dataset. No random splitting is performed.

    Example:

        train_items, valid_items, test_items = make_nc_block_dataset(
            train_dir="data/splits/train",
            valid_dir="data/splits/valid",
            test_dir="data/splits/test",
            channel_var_names=("ssha_unfiltered",),
            mask_var_name="label",
            block_size=32,
        )

    Parameters
    ----------
    nc_dir : str, optional
        Directory containing all NetCDF files when automatic splitting
        should be performed.
    channel_var_names : Sequence[str]
        NetCDF variable names to use as input channels.
    mask_var_name : str
        NetCDF variable name containing the binary label mask.
    block_size : int or tuple
        Block size as one integer or as `(height, width)`.
    valid_ratio : float
        Validation ratio used only in automatic split mode.
    test_ratio : float
        Test ratio used only in automatic split mode.
    seed : int
        Random seed used only in automatic split mode.
    train_dir : str, optional
        Directory containing existing training files.
    valid_dir : str, optional
        Directory containing existing validation files.
    test_dir : str, optional
        Directory containing existing test files.

    Returns
    -------
    train_items : list of dict
        Training dataset items.
    valid_items : list of dict
        Validation dataset items.
    test_items : list of dict
        Test dataset items.

    Each item contains:

        {
            "image": np.ndarray [C, Hc, Wc] float32,
            "labels": np.ndarray [Bh, Bw] uint8,
            "valid": np.ndarray [Bh, Bw] bool,
            "id": str,
            "block_size": (bh, bw),
            "grid_shape": (Bh, Bw),
        }
    """
    # -----------------------------------------------------------------
    # Validate common arguments
    # -----------------------------------------------------------------
    if block_size is None:
        raise ValueError("block_size must be provided")

    bh, bw = _ensure_block_size(block_size)

    if bh <= 0 or bw <= 0:
        raise ValueError(
            f"block_size values must be positive, got ({bh}, {bw})"
        )

    # Allow a single string or a sequence of strings.
    if isinstance(channel_var_names, str):
        channel_var_names = (channel_var_names,)
    else:
        channel_var_names = tuple(channel_var_names)

    if len(channel_var_names) == 0:
        raise ValueError(
            "channel_var_names must contain at least one channel name"
        )

    if not mask_var_name:
        raise ValueError("mask_var_name must be provided")

    # -----------------------------------------------------------------
    # Determine which loading mode is being used
    # -----------------------------------------------------------------
    using_single_directory = nc_dir is not None

    explicit_directories = {
        "train_dir": train_dir,
        "valid_dir": valid_dir,
        "test_dir": test_dir,
    }

    provided_explicit_directories = {
        name: path
        for name, path in explicit_directories.items()
        if path is not None
    }

    using_explicit_directories = bool(provided_explicit_directories)

    if using_single_directory and using_explicit_directories:
        raise ValueError(
            "Use either `nc_dir` for automatic splitting or "
            "`train_dir`, `valid_dir`, and `test_dir` for existing "
            "splits. Do not provide both modes."
        )

    if not using_single_directory and not using_explicit_directories:
        raise ValueError(
            "Provide either `nc_dir` or all three split directories: "
            "`train_dir`, `valid_dir`, and `test_dir`."
        )

    # -----------------------------------------------------------------
    # Mode 1: use existing split directories
    # -----------------------------------------------------------------
    if using_explicit_directories:
        missing_directories = [
            name
            for name, path in explicit_directories.items()
            if path is None
        ]

        if missing_directories:
            raise ValueError(
                "When using existing split directories, all three must "
                "be provided. Missing: "
                f"{missing_directories}"
            )

        train_files = _find_nc_files(
            directory=train_dir,
            split_name="training",
        )

        valid_files = _find_nc_files(
            directory=valid_dir,
            split_name="validation",
        )

        test_files = _find_nc_files(
            directory=test_dir,
            split_name="test",
        )

        train_items = _load_nc_items(
            files=train_files,
            channel_var_names=channel_var_names,
            mask_var_name=mask_var_name,
            bh=bh,
            bw=bw,
        )

        valid_items = _load_nc_items(
            files=valid_files,
            channel_var_names=channel_var_names,
            mask_var_name=mask_var_name,
            bh=bh,
            bw=bw,
        )

        test_items = _load_nc_items(
            files=test_files,
            channel_var_names=channel_var_names,
            mask_var_name=mask_var_name,
            bh=bh,
            bw=bw,
        )

        print("Loaded existing dataset splits:")
        print(f"  Train: {len(train_items)} files")
        print(f"  Valid: {len(valid_items)} files")
        print(f"  Test:  {len(test_items)} files")

        return train_items, valid_items, test_items

    # -----------------------------------------------------------------
    # Mode 2: automatically split one directory
    # -----------------------------------------------------------------
    valid_ratio = float(valid_ratio)
    test_ratio = float(test_ratio)

    if not 0.0 <= valid_ratio < 1.0:
        raise ValueError(
            f"valid_ratio must be in [0, 1), got {valid_ratio}"
        )

    if not 0.0 <= test_ratio < 1.0:
        raise ValueError(
            f"test_ratio must be in [0, 1), got {test_ratio}"
        )

    if valid_ratio + test_ratio >= 1.0:
        raise ValueError(
            "valid_ratio + test_ratio must be less than 1.0 so that "
            "some files remain for training. "
            f"Got valid_ratio={valid_ratio}, "
            f"test_ratio={test_ratio}, "
            f"sum={valid_ratio + test_ratio}."
        )

    files = _find_nc_files(
        directory=nc_dir,
        split_name="input",
    )

    rng = np.random.RandomState(seed)
    permutation = rng.permutation(len(files))

    n_files = len(files)
    n_test = int(round(test_ratio * n_files))
    n_valid = int(round(valid_ratio * n_files))

    if n_test + n_valid >= n_files:
        raise ValueError(
            "Not enough files for the requested split. "
            f"Found {n_files} files, requested "
            f"n_valid={n_valid}, n_test={n_test}. "
            "Reduce valid_ratio and/or test_ratio."
        )

    test_indices = permutation[:n_test]
    valid_indices = permutation[n_test:n_test + n_valid]
    train_indices = permutation[n_test + n_valid:]

    train_files = [files[index] for index in train_indices]
    valid_files = [files[index] for index in valid_indices]
    test_files = [files[index] for index in test_indices]

    train_items = _load_nc_items(
        files=train_files,
        channel_var_names=channel_var_names,
        mask_var_name=mask_var_name,
        bh=bh,
        bw=bw,
    )

    valid_items = _load_nc_items(
        files=valid_files,
        channel_var_names=channel_var_names,
        mask_var_name=mask_var_name,
        bh=bh,
        bw=bw,
    )

    test_items = _load_nc_items(
        files=test_files,
        channel_var_names=channel_var_names,
        mask_var_name=mask_var_name,
        bh=bh,
        bw=bw,
    )

    print("Created automatic dataset splits:")
    print(f"  Total: {n_files} files")
    print(f"  Train: {len(train_items)} files")
    print(f"  Valid: {len(valid_items)} files")
    print(f"  Test:  {len(test_items)} files")

    return train_items, valid_items, test_items