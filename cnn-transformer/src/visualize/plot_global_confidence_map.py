"""Plot CNN-Transformer internal-wave confidence on a global map.

Reads every slim NetCDF written by ``src/inference/infer.py`` /
``src/inference/infer_tiled.py`` -- each carries a pixel-resolution
``internal_wave_bbox_confidence`` variable plus 2-D ``latitude``/``longitude``.

Two modes:

* default -- bin the confidences onto a global lat/lon grid and draw the per-cell
  mean with Cartopy. One sample per 20x20 block is taken (``--block-stride``),
  which is lossless when the field is block-constant (the block-head models).
* ``--scatter`` -- no binning: scatter every pixel whose confidence is
  ``>= --scatter-min``, coloured by its own value. Use this for the dense-head
  model, whose confidence field varies pixel to pixel.

Example
-------
    cd cnn-transformer
    python src/inference/plot_global_confidence_map.py
    python src/inference/plot_global_confidence_map.py --resolution 0.5 --vmax 0.1
    python src/inference/plot_global_confidence_map.py --scatter --scatter-min 0.2 \\
      --netcdf-dir results/dense_head_rolling_mean_highpass_v0.1.0/SWOT_L3_Full_Dataset/netcdf \\
      --out results/dense_head_rolling_mean_highpass_v0.1.0/SWOT_L3_Full_Dataset/global_confidence_map_perpixel.png

Run from the ``cnn-transformer/`` directory (default paths are relative to it).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

_INFERENCE_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _INFERENCE_DIR.parent.parent  # cnn-transformer/

DEFAULT_NETCDF_DIR = (
    _MODEL_DIR
    / "results/rolling_mean_higpass_v0.2.0/SWOT_L3_Full_Dataset/netcdf"
)
DEFAULT_OUT = DEFAULT_NETCDF_DIR.parent / "global_confidence_map.png"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--netcdf-dir", type=Path, nargs="+", default=[DEFAULT_NETCDF_DIR],
                   help="One or more directories of slim inference NetCDFs to scan for "
                        "*.nc (pooled together, e.g. several cycles).")
    p.add_argument("--conf-var", default="internal_wave_bbox_confidence",
                   help="Name of the pixel-resolution confidence variable.")
    p.add_argument("--resolution", type=float, default=0.25,
                   help="Global bin size in degrees.")
    p.add_argument("--block-stride", type=int, default=20,
                   help="Pixel block size; one confidence sample is taken per block.")
    p.add_argument("--vmax", type=float, default=None,
                   help="Upper end of the colour scale. Default: max cell mean.")
    p.add_argument("--min-confidence", type=float, default=0.02,
                   help="Cells with a mean confidence below this are left unplotted. "
                        "Set to 0 to plot every non-empty cell.")
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)

    p.add_argument("--scatter", action="store_true",
                   help="Per-pixel scatter instead of binning (for dense-head "
                        "outputs).")
    p.add_argument("--scatter-min", type=float, default=0.2,
                   help="--scatter: only pixels with confidence >= this are drawn.")
    p.add_argument("--scatter-size", type=float, default=0.5,
                   help="--scatter: matplotlib marker size.")
    return p.parse_args(argv)


def wrap_longitude(lon: np.ndarray) -> np.ndarray:
    """0..360 (the SWOT L3 convention) -> [-180, 180)."""
    return ((lon + 180.0) % 360.0) - 180.0


def accumulate(nc_files, conf_var, stride, lat_edges, lon_edges):
    nlat = lat_edges.size - 1
    nlon = lon_edges.size - 1
    conf_sum = np.zeros((nlat, nlon), dtype=np.float64)
    count = np.zeros((nlat, nlon), dtype=np.float64)

    n_ok = n_fail = n_blocks = 0
    block_max = 0.0
    off = stride // 2

    for i, nc_path in enumerate(nc_files, start=1):
        try:
            with xr.open_dataset(nc_path) as ds:
                conf = np.asarray(ds[conf_var].values, dtype=np.float64)
                lat = np.asarray(ds["latitude"].values, dtype=np.float64)
                lon = np.asarray(ds["longitude"].values, dtype=np.float64)

            conf = conf[off::stride, off::stride].ravel()
            lat = lat[off::stride, off::stride].ravel()
            lon = wrap_longitude(lon[off::stride, off::stride].ravel())

            good = np.isfinite(conf) & np.isfinite(lat) & np.isfinite(lon)
            conf, lat, lon = conf[good], lat[good], lon[good]
            if conf.size == 0:
                n_ok += 1
                continue

            block_max = max(block_max, float(conf.max()))
            n_blocks += conf.size

            s, _, _ = np.histogram2d(lat, lon, bins=[lat_edges, lon_edges], weights=conf)
            c, _, _ = np.histogram2d(lat, lon, bins=[lat_edges, lon_edges])
            conf_sum += s
            count += c
            n_ok += 1
        except Exception as exc:  # noqa: BLE001 - report and keep going
            n_fail += 1
            print(f"[{i}/{len(nc_files)}] {nc_path.name}: FAILED: {exc}")

        if i % 100 == 0 or i == len(nc_files):
            print(f"  scanned {i}/{len(nc_files)} files")

    return conf_sum, count, n_ok, n_fail, n_blocks, block_max


def collect_points(nc_files, conf_var, scatter_min):
    """Gather (lon, lat, conf) for every pixel with conf >= scatter_min."""
    lons, lats, confs = [], [], []
    n_ok = n_fail = kept = 0
    conf_max = 0.0

    for i, nc_path in enumerate(nc_files, start=1):
        try:
            with xr.open_dataset(nc_path) as ds:
                conf = np.asarray(ds[conf_var].values, dtype=np.float32)
                lat = np.asarray(ds["latitude"].values, dtype=np.float32)
                lon = wrap_longitude(np.asarray(ds["longitude"].values, dtype=np.float32))

            keep = (np.isfinite(conf) & (conf >= scatter_min)
                    & np.isfinite(lat) & np.isfinite(lon))
            if keep.any():
                confs.append(conf[keep])
                lats.append(lat[keep])
                lons.append(lon[keep])
                kept += int(keep.sum())
                conf_max = max(conf_max, float(conf[keep].max()))
            n_ok += 1
        except Exception as exc:  # noqa: BLE001 - report and keep going
            n_fail += 1
            print(f"[{i}/{len(nc_files)}] {nc_path.name}: FAILED: {exc}")

        if i % 100 == 0 or i == len(nc_files):
            print(f"  scanned {i}/{len(nc_files)} files | kept {kept:,} px")

    if not confs:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty, empty, n_ok, n_fail, kept, conf_max
    return (np.concatenate(lons), np.concatenate(lats), np.concatenate(confs),
            n_ok, n_fail, kept, conf_max)


def render_scatter(args, nc_files):
    smin = float(args.scatter_min)
    print(f"netcdf_dir={', '.join(str(d.resolve()) for d in args.netcdf_dir)}")
    print(f"files={len(nc_files)} conf_var={args.conf_var} mode=scatter "
          f"scatter_min={smin} scatter_size={args.scatter_size}")

    lon, lat, conf, n_ok, n_fail, kept, conf_max = collect_points(
        nc_files, args.conf_var, smin
    )
    print(f"read ok={n_ok} failed={n_fail} | pixels kept (conf>={smin}): {kept:,} | "
          f"max conf={conf_max:.4f}")
    if conf.size == 0:
        raise SystemExit(f"no pixels with confidence >= {smin}")

    order = np.argsort(conf, kind="stable")  # strongest pixels drawn last / on top
    lon, lat, conf = lon[order], lat[order], conf[order]
    vmax = args.vmax if args.vmax is not None else float(conf[-1])

    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=0)
    ax.coastlines(linewidth=0.4)

    sc = ax.scatter(
        lon, lat, c=conf, s=args.scatter_size, marker="s", linewidths=0,
        cmap=args.cmap, vmin=smin, vmax=vmax,
        transform=ccrs.PlateCarree(), rasterized=True,
    )

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.6", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    fig.colorbar(sc, ax=ax, orientation="vertical", shrink=0.7, pad=0.02,
                 label="internal-wave confidence (per pixel)")
    ax.set_title(
        f"CNN-Transformer per-pixel confidence  |  {n_ok} files  |  "
        f"conf \N{GREATER-THAN OR EQUAL TO} {smin:g}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out.resolve()}")
    return 1 if n_fail else 0


def main(argv=None):
    args = parse_args(argv)

    nc_files = sorted(p for d in args.netcdf_dir for p in d.glob("*.nc"))
    if not nc_files:
        dirs = ", ".join(str(d.resolve()) for d in args.netcdf_dir)
        raise FileNotFoundError(f"No .nc files found in {dirs}")

    if args.scatter:
        return render_scatter(args, nc_files)

    res = float(args.resolution)
    lat_edges = np.arange(-90.0, 90.0 + res / 2, res)
    lon_edges = np.arange(-180.0, 180.0 + res / 2, res)

    print(f"netcdf_dir={', '.join(str(d.resolve()) for d in args.netcdf_dir)}")
    print(f"files={len(nc_files)} conf_var={args.conf_var} resolution={res} deg "
          f"grid={lat_edges.size - 1}x{lon_edges.size - 1}")

    conf_sum, count, n_ok, n_fail, n_blocks, block_max = accumulate(
        nc_files, args.conf_var, args.block_stride, lat_edges, lon_edges
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_grid = np.where(count > 0, conf_sum / count, np.nan)

    n_filled = int(np.isfinite(mean_grid).sum())

    vmin = max(0.0, float(args.min_confidence))
    if vmin > 0:
        mean_grid = np.where(mean_grid >= vmin, mean_grid, np.nan)

    n_cells = int(np.isfinite(mean_grid).sum())
    grid_max = float(np.nanmax(mean_grid)) if n_cells else 0.0
    vmax = args.vmax if args.vmax is not None else (grid_max if grid_max > vmin else vmin + 1.0)

    print(f"read ok={n_ok} failed={n_fail} | blocks binned={n_blocks:,} | "
          f"non-empty cells={n_filled:,}")
    print(f"per-block max confidence={block_max:.4f} | max cell mean={grid_max:.4f} | "
          f"min-confidence={vmin:.4f} -> plotted cells={n_cells:,} | vmax={vmax:.4f}")

    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=0)
    ax.coastlines(linewidth=0.4)

    mesh = ax.pcolormesh(
        lon_edges, lat_edges, np.ma.masked_invalid(mean_grid),
        transform=ccrs.PlateCarree(), cmap=args.cmap,
        vmin=vmin, vmax=vmax, shading="flat",
    )

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.6", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    fig.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.7, pad=0.02,
                 label="mean internal-wave confidence")
    title = (
        f"CNN-Transformer (rmh v0.2.0) mean confidence  |  "
        f"{n_ok} files  |  {res}\N{DEGREE SIGN} bins"
    )
    if vmin > 0:
        title += f"  |  cells \N{GREATER-THAN OR EQUAL TO} {vmin:g}"
    ax.set_title(title)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out.resolve()}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
