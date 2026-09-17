"""Global map of highpass_variance_mean, averaged across every pass in a Zarr store.

Reads the ``iw[time, channel, latitude, longitude]`` array built by
``build_isw_store.py`` (e.g. ``stores/isw-highpass-variance``, one time-slice
per SWOT pass) and takes the MEAN of the ``highpass_variance_mean`` channel
across the time axis to build one global raster. Mean (not max) is used
because ``plot_highpass_variance.py`` itself never reduces across passes at
all -- it scatters every kept block once, at its own value, since blocks
essentially never overlap between passes. Averaging reproduces that same
behavior almost everywhere (at a cell with a single contributing pass,
mean == max == that pass's value); the two only differ at rare crossover
cells with more than one contributing pass, where mean (unlike max) won't
inflate the cell with an outlier.

The store is far too large to load whole, so the reduction is done
chunk-by-chunk via ``dask.array``, matching the store's own on-disk
chunking. Cells no pass ever covers stay NaN (transparent / NoData).

Writes a PNG (matplotlib + Cartopy, same look as
``plot_detection_count_global_sum.py``) and/or a GeoTIFF (rasterio, same
conventions: COG, zero-value cells excluded, normalized to [0, 1], exact
global bounds).

Requires ``zarr``, ``dask``, and (for the PNG) ``cartopy``/``matplotlib`` or
(for the GeoTIFF) ``rasterio``. On this machine the ``gis-stuff`` conda env
has all of them: ``/Users/aguilarj/miniforge3/envs/gis-stuff/bin/python``.

Example
-------
    cd cnn-transformer/src/visualize
    /Users/aguilarj/miniforge3/envs/gis-stuff/bin/python \\
      plot_highpass_variance_global_mean.py --store stores/isw-highpass-variance
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import zarr
import dask.array as da

_HERE = Path(__file__).resolve().parent


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--store", type=Path, default=_HERE / "stores" / "isw-highpass-variance")
    p.add_argument("--channel", default="highpass_variance_mean")
    p.add_argument("--cmap", default="magma")
    p.add_argument("--vmin", type=float, default=None,
                   help="Bottom of the colour scale (raw m^2). Default: 0, or the "
                        "low --clim-quantile if given.")
    p.add_argument("--vmax", type=float, default=None,
                   help="Top of the colour scale (raw m^2). Default: 95th pct of "
                        "the plotted values, or the high --clim-quantile if given.")
    p.add_argument("--clim-quantile", type=float, nargs=2, default=(0.05, 0.95),
                   metavar=("LO", "HI"),
                   help="Set the colour limits from two quantiles of the plotted "
                        "values, fractions in [0,1]. --vmin/--vmax override the "
                        "matching end.")
    p.add_argument("--log", action="store_true", help="Log colour scale.")
    p.add_argument("--out-png", type=Path,
                   default=_HERE.parent.parent / "results" / "all_cycles"
                   / "highpass_variance_mean_global_mean.png",
                   help="Set to '' / 'none' to skip the PNG.")
    p.add_argument("--out-tif", type=Path,
                   default=_HERE.parent.parent / "results" / "all_cycles"
                   / "highpass_variance_mean_global_mean.tif",
                   help="Set to '' / 'none' to skip the GeoTIFF.")
    return p.parse_args(argv)


def load_and_mean(store: Path, channel: str):
    """Average ``channel`` across the store's time axis, chunk-by-chunk.

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

    print("reducing across time axis (chunked)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns, expected
        total = da.nanmean(plane, axis=0)
        total = total.compute().astype("f4")
    print(f"non-empty cells = {int(np.isfinite(total).sum()):,}  "
          f"max = {np.nanmax(total):.4g}  median(non-empty) = {np.nanmedian(total):.4g}")
    return total, lat, lon


def write_png(total, lat, lon, out_png: Path, cmap: str, vmin, vmax, clim_quantile, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    masked = np.ma.masked_invalid(total)
    masked = np.ma.masked_less_equal(masked, 0)
    finite = np.asarray(masked.compressed())
    if finite.size == 0:
        raise SystemExit("nothing to plot: no finite, positive cells")

    q_lo, q_hi = clim_quantile if clim_quantile else (None, None)
    if clim_quantile and not (0.0 <= q_lo < q_hi <= 1.0):
        raise SystemExit("--clim-quantile LO HI must satisfy 0 <= LO < HI <= 1")
    vmin = (vmin if vmin is not None
            else float(np.quantile(finite, q_lo)) if q_lo is not None else 0.0)
    vmax = (vmax if vmax is not None
            else float(np.quantile(finite, q_hi)) if q_hi is not None
            else float(np.quantile(finite, 0.98)))
    if vmax <= vmin:
        vmax = vmin + (abs(vmin) or 1.0) * 1e-3
    print(f"colour scale: vmin={vmin:.4g} vmax={vmax:.4g} (raw max={float(finite.max()):.4g}) log={log}")

    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=0)
    ax.coastlines(linewidth=0.4)

    imshow_kwargs = dict(
        extent=[lon[0], lon[-1], lat[0], lat[-1]], origin="lower",
        transform=ccrs.PlateCarree(), cmap=cmap, interpolation="nearest", zorder=1,
    )
    if log:
        vmin_log = vmin if vmin > 0 else max(float(np.quantile(finite, 0.02)), vmax * 1e-4)
        norm = LogNorm(vmin=vmin_log, vmax=max(vmax, vmin_log * 1.001))
        im = ax.imshow(masked, norm=norm, **imshow_kwargs)
    else:
        im = ax.imshow(masked, vmin=vmin, vmax=vmax, **imshow_kwargs)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.6", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.7, pad=0.02,
                 label="mean highpass_variance_mean across all passes (m\N{SUPERSCRIPT TWO})")
    ax.set_title("SWOT highpass SSHA variance (mean over cycles), averaged across all passes")

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

    # exclude zero-value cells the same way the detection-count map does -- they
    # become NoData, not a real 0.0 value (in practice an exact-zero block variance
    # essentially never occurs, so this is mostly a no-op here) -- then rescale the
    # remaining values to [0, 1] by dividing by the max.
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
        dst.set_band_description(1, "normalized_mean_highpass_variance_mean")
        dst.update_tags(
            n_passes=str(n_passes), source_store=str(store),
            normalization=f"raw_mean / max({max_val:g}); zero-value cells excluded (NoData)",
            mean_raw_highpass_variance_mean_peak=str(max_val),
        )

    print(f"wrote {out_tif.resolve()}")


def main(argv=None):
    args = parse_args(argv)

    total, lat, lon = load_and_mean(args.store, args.channel)

    root = zarr.open_group(str(args.store), mode="r")
    n_passes = len(root.attrs.get("slices", []))

    out_png = args.out_png if str(args.out_png).lower() not in ("", "none") else None
    out_tif = args.out_tif if str(args.out_tif).lower() not in ("", "none") else None

    if out_png is not None:
        write_png(total, lat, lon, out_png, args.cmap, args.vmin, args.vmax,
                 args.clim_quantile, args.log)
    if out_tif is not None:
        write_geotiff(total, lat, lon, out_tif, n_passes, args.store)

    if out_png is None and out_tif is None:
        print("both --out-png and --out-tif were skipped; nothing written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
