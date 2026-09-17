"""Global map of ISW detection_count, summed across every pass in a Zarr store.

Reads the ``iw[time, channel, latitude, longitude]`` array built by
``build_isw_store.py`` (e.g. ``stores/isw-detection-counts``, one time-slice
per SWOT pass) and sums the ``detection_count`` channel across the time axis
to build one global raster: for each grid cell, "how many times, across all
passes and all their repeat cycles, did an ISW get detected here."

The store is far too large to load whole (e.g. 584 passes x 9000 x 18000
float32 = ~180 GB for one channel), so the reduction is done chunk-by-chunk
via ``dask.array``, matching the store's own on-disk chunking. Cells no pass
ever covers stay NaN (transparent / NoData) rather than reading as a false
"0 detections".

Writes a PNG (matplotlib + Cartopy, same look as ``plot_detection_frequency.py``)
and/or a GeoTIFF (rasterio, same conventions as
``plot_detection_frequency_geotiff.py``).

Requires ``zarr``, ``dask``, and (for the PNG) ``cartopy``/``matplotlib`` or
(for the GeoTIFF) ``rasterio``. On this machine the ``gis-stuff`` conda env
has all of them: ``/Users/aguilarj/miniforge3/envs/gis-stuff/bin/python``.

Example
-------
    cd cnn-transformer/src/visualize
    /Users/aguilarj/miniforge3/envs/gis-stuff/bin/python \\
      plot_detection_count_global_sum.py --store stores/isw-detection-counts
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import zarr
import dask.array as da

_HERE = Path(__file__).resolve().parent


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--store", type=Path, default=_HERE / "stores" / "isw-detection-counts")
    p.add_argument("--channel", default="detection_count")
    p.add_argument("--cmap", default="inferno")
    p.add_argument("--vmax", type=float, default=None,
                   help="Colour-scale max (default: data max).")
    p.add_argument("--out-png", type=Path,
                   default=_HERE.parent.parent / "results" / "all_cycles"
                   / "detection_count_global_sum.png",
                   help="Set to '' / 'none' to skip the PNG.")
    p.add_argument("--out-tif", type=Path,
                   default=_HERE.parent.parent / "results" / "all_cycles"
                   / "detection_count_global_sum.tif",
                   help="Set to '' / 'none' to skip the GeoTIFF.")
    return p.parse_args(argv)


def load_and_sum(store: Path, channel: str):
    """Sum ``channel`` across the store's time axis, chunk-by-chunk.

    Returns (total, latitude, longitude) where ``total`` is a
    (ny, nx) float32 array, NaN where no pass ever covers that cell.
    """
    root = zarr.open_group(str(store), mode="r")
    channels = root.attrs["channels"]
    idx = next((c["index"] for c in channels if c["name"] == channel), None)
    if idx is None:
        names = [c["name"] for c in channels]
        raise SystemExit(f"channel {channel!r} not in store; available: {names}")

    g = root["0"]
    lat = np.asarray(g["latitude"][:], dtype="f4")
    lon = np.asarray(g["longitude"][:], dtype="f4")

    iw = g["iw"]
    n_time = iw.shape[0]
    print(f"store = {store.resolve()}")
    print(f"channel = {channel!r} (index {idx})  time slices = {n_time}  "
          f"grid = {lat.size}x{lon.size}")

    plane = da.from_array(iw, chunks=iw.chunks)[:, idx, :, :]
    total = da.nansum(plane, axis=0)
    coverage = da.any(np.isfinite(plane), axis=0)
    total = da.where(coverage, total, np.nan)

    print("reducing across time axis (chunked)...")
    total = total.compute().astype("f4")
    print(f"non-empty cells = {int(np.isfinite(total).sum()):,}  "
          f"max = {np.nanmax(total):.4g}  median(non-empty) = {np.nanmedian(total):.4g}")
    return total, lat, lon


def write_png(total, lat, lon, out_png: Path, cmap: str, vmax: float | None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    vmax = vmax if vmax is not None else float(np.nanmax(total))
    masked = np.ma.masked_invalid(total)
    masked = np.ma.masked_less_equal(masked, 0)

    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=0)
    ax.coastlines(linewidth=0.4)

    im = ax.imshow(
        masked, extent=[lon[0], lon[-1], lat[0], lat[-1]], origin="lower",
        transform=ccrs.PlateCarree(), cmap=cmap, vmin=0, vmax=vmax,
        interpolation="nearest", zorder=1,
    )

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.6", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.7, pad=0.02,
                 label="summed detection_count (all passes)")
    ax.set_title("SWOT internal-wave detection_count, summed across all passes")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png.resolve()}")


def write_geotiff(total, lat, lon, out_tif: Path, n_passes: int, store: Path):
    import rasterio
    from rasterio.transform import from_bounds

    # The store's lat/lon coordinate arrays are float32 pixel centres, so deriving
    # per-pixel resolution from adjacent-cell differences (e.g. lon[1]-lon[0]) picks
    # up float32 rounding error that, multiplied across 18000 columns, drifts the
    # east edge by ~0.08 deg -- a real misalignment once georeferenced. The store is
    # known to span the whole globe exactly, so anchor the transform to the true
    # -180/-90/180/90 bounds instead of an accumulated per-pixel resolution.
    ny, nx = total.shape

    # exclude zero-count cells (no detections) the same way the PNG does -- they
    # become NoData, not a real 0.0 value -- then rescale the remaining values to
    # [0, 1] by dividing by the max, so the raster is a normalized detection-count
    # surface rather than raw counts.
    zeroed = np.where(total > 0, total, np.nan).astype("f4")
    max_val = float(np.nanmax(zeroed))
    normalized = zeroed / max_val

    # array row 0 is the southernmost latitude; GeoTIFF row 0 must be north-up.
    band = np.flipud(normalized).astype("f4")
    transform = from_bounds(-180.0, -90.0, 180.0, 90.0, nx, ny)

    out_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_tif, "w", driver="COG",
        height=band.shape[0], width=band.shape[1], count=1,
        dtype="float32", crs="EPSG:4326", transform=transform,
        nodata=np.nan, compress="DEFLATE",
    ) as dst:
        dst.write(band, 1)
        dst.set_band_description(1, "normalized_sum_detection_count")
        dst.update_tags(
            n_passes=str(n_passes), source_store=str(store),
            normalization=f"raw_sum / max({max_val:g}); zero-count cells excluded (NoData)",
            max_raw_sum_detection_count=str(max_val),
        )

    print(f"wrote {out_tif.resolve()}")


def main(argv=None):
    args = parse_args(argv)

    total, lat, lon = load_and_sum(args.store, args.channel)

    root = zarr.open_group(str(args.store), mode="r")
    n_passes = len(root.attrs.get("slices", []))

    out_png = args.out_png if str(args.out_png).lower() not in ("", "none") else None
    out_tif = args.out_tif if str(args.out_tif).lower() not in ("", "none") else None

    if out_png is not None:
        write_png(total, lat, lon, out_png, args.cmap, args.vmax)
    if out_tif is not None:
        write_geotiff(total, lat, lon, out_tif, n_passes, args.store)

    if out_png is None and out_tif is None:
        print("both --out-png and --out-tif were skipped; nothing written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
