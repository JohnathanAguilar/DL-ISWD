"""Side-by-side check: a high-pass SSHA field straight from the raw swath NetCDF
(native curvilinear geolocation) vs. the same field pulled from a regridded
``build_isw_store.py`` zarr store (regular lat/lon grid).

One figure, two rows: a wide overview (``--lat-min``/``--lat-max``) on top and a
close-up (``--zoom-height`` degrees tall, centred at ``--zoom-center``) that shows
native-pixel-level detail on the bottom -- raw on the left, regridded on the right
in both rows, one shared colour scale.

This is the practical version of "did the quadmesh regrid change anything" -- run it
after ``build_isw_store.py`` to visually confirm the regrid matches the source.

Example
-------
    cd cnn-transformer/src/visualize
    /Users/aguilarj/miniforge3/envs/gis-stuff/bin/python plot_regrid_comparison.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from plot_detection_frequency import wrap_longitude
from plot_pass_section import _land_features

_HERE = Path(__file__).resolve().parent
_DATA_ROOT = _HERE.parent.parent.parent / "data"

_DEFAULT_RAW = (
    _DATA_ROOT / "SWOT_L3_Full_Dataset" / "cycle_15"
    / "SWOT_L3_LR_SSH_Expert_015_001_20240508T080454_20240508T085620_v3.0.nc"
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--raw", type=Path, default=_DEFAULT_RAW,
                   help="Raw swath NetCDF (native curvilinear lat/lon).")
    p.add_argument("--store", type=Path, default=_HERE / "stores" / "cycle15_pass001.zarr",
                   help="Regridded zarr store from build_isw_store.py.")
    p.add_argument("--level", type=int, default=0, help="Pyramid level to read.")
    p.add_argument("--time", type=int, default=0, help="Time-slice index in the store.")
    p.add_argument("--hp-var", default="gaussian_filter_highpass",
                   help="Variable/channel name, same in both the raw file and the store.")
    p.add_argument("--lat-min", type=float, default=-20.0)
    p.add_argument("--lat-max", type=float, default=20.0)
    p.add_argument("--zoom-center", type=float, default=0.0,
                   help="Latitude (deg) the close-up row is centred on.")
    p.add_argument("--zoom-height", type=float, default=1.0,
                   help="Height (deg latitude) of the close-up row.")
    p.add_argument("--clim", type=float, default=None,
                   help="Symmetric colour limit (m), shared by both rows. Default: 98th "
                        "pct of |value| over the overview crop (which contains the zoom crop).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output PNG. Default: <store>_raw_vs_regrid_lat{min}to{max}_"
                        "zoom{zoom_height}deg.png")
    return p.parse_args(argv)


def load_raw(path, hp_var, lat_min, lat_max):
    with xr.open_dataset(path) as ds:
        lat = np.asarray(ds["latitude"].values, dtype="f8")
        lon = wrap_longitude(np.asarray(ds["longitude"].values, dtype="f8"))
        hp = np.asarray(ds[hp_var].values, dtype="f4")

    inband = (lat >= lat_min) & (lat <= lat_max)
    rows = np.where(np.any(inband, axis=1))[0]
    if rows.size == 0:
        raise SystemExit(f"raw file has no rows with latitude in [{lat_min},{lat_max}]")
    r0, r1 = rows.min(), rows.max() + 1
    sl = (slice(r0, r1), slice(None))
    lat, lon, hp, inband = lat[sl], lon[sl], hp[sl], inband[sl]
    hp = np.where(inband, hp, np.nan)
    return lat, lon, hp


def load_store(store, level, time, hp_var, lat_min, lat_max):
    root = zarr.open_group(str(store), mode="r")
    channels = root.attrs["channels"]
    idx = next((c["index"] for c in channels if c["name"] == hp_var), None)
    if idx is None:
        names = [c["name"] for c in channels]
        raise SystemExit(f"channel {hp_var!r} not in store; available: {names}")

    g = root[str(level)]
    lat = np.asarray(g["latitude"])
    lon = np.asarray(g["longitude"])
    r0, r1 = np.searchsorted(lat, [lat_min, lat_max])
    hp = np.asarray(g["iw"][time, idx, r0:r1, :], dtype="f4")
    lat = lat[r0:r1]

    slices = root.attrs.get("slices", [])
    meta = slices[time] if time < len(slices) else {}
    return lat, lon, hp, meta


def raw_lon_bounds(lon, hp, margin_deg):
    """True (lo, hi) longitude span of hp's finite cells, from the full 2-D
    lon/hp footprint (NOT a single row -- the swath's longitude drifts
    substantially along-track over a wide latitude crop)."""
    mask = np.isfinite(hp) & np.isfinite(lon)
    if not mask.any():
        raise SystemExit("no finite data in the requested latitude band")
    lo = float(np.min(lon[mask])) - margin_deg
    hi = float(np.max(lon[mask])) + margin_deg
    return lo, hi


@dataclass
class Row:
    label: str
    lat_r: np.ndarray
    lon_r: np.ndarray
    hp_r: np.ndarray
    lat_s: np.ndarray
    lon_s: np.ndarray
    hp_s: np.ndarray
    extent: list
    n_raw: int
    n_store: int


def load_row(args, lat_lo, lat_hi, lon_margin, label, meta_out):
    lat_r, lon_r, hp_r = load_raw(args.raw, args.hp_var, lat_lo, lat_hi)
    lat_s, lon_s, hp_s, meta = load_store(
        args.store, args.level, args.time, args.hp_var, lat_lo, lat_hi
    )
    meta_out.update(meta)

    lon_lo, lon_hi = raw_lon_bounds(lon_r, hp_r, lon_margin)
    c0, c1 = np.searchsorted(lon_s, [lon_lo, lon_hi])
    lon_s, hp_s = lon_s[c0:c1], hp_s[:, c0:c1]

    n_raw, n_store = int(np.isfinite(hp_r).sum()), int(np.isfinite(hp_s).sum())
    print(f"[{label}] raw:   rows={hp_r.shape}  finite={n_raw:,}")
    print(f"[{label}] store: crop={hp_s.shape}  finite={n_store:,}")
    print(f"[{label}] extent=lon[{lon_lo:.4f},{lon_hi:.4f}] lat[{lat_lo:g},{lat_hi:g}]")
    return Row(label, lat_r, lon_r, hp_r, lat_s, lon_s, hp_s,
              [lon_lo, lon_hi, lat_lo, lat_hi], n_raw, n_store)


def plot_row(axes, row: Row, clim):
    ax = axes[0]
    ax.set_extent(row.extent, crs=ccrs.PlateCarree())
    _land_features(ax)
    mesh = ax.pcolormesh(row.lon_r, row.lat_r, np.ma.masked_invalid(row.hp_r), cmap="RdBu_r",
                         vmin=-clim, vmax=clim, shading="nearest",
                         transform=ccrs.PlateCarree(), zorder=2)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.6", alpha=0.5)
    gl.top_labels = gl.right_labels = False
    ax.set_title(f"raw swath (native geolocation)\n{row.label}  \N{BULLET}  "
                 f"finite px = {row.n_raw:,}", fontsize=10)

    ax = axes[1]
    ax.set_extent(row.extent, crs=ccrs.PlateCarree())
    _land_features(ax)
    ax.pcolormesh(row.lon_s, row.lat_s, np.ma.masked_invalid(row.hp_s), cmap="RdBu_r",
                  vmin=-clim, vmax=clim, shading="nearest",
                  transform=ccrs.PlateCarree(), zorder=2)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.6", alpha=0.5)
    gl.top_labels = gl.right_labels = False
    ax.set_title(f"regridded (0.02\N{DEGREE SIGN} quadmesh)\n{row.label}  \N{BULLET}  "
                 f"finite px = {row.n_store:,}", fontsize=10)
    return mesh


def main(argv=None):
    args = parse_args(argv)
    out = args.out or (
        args.store.parent
        / f"{args.store.stem}_raw_vs_regrid_lat{args.lat_min:g}to{args.lat_max:g}"
          f"_zoom{args.zoom_height:g}deg.png"
    )
    zlo = args.zoom_center - args.zoom_height / 2
    zhi = args.zoom_center + args.zoom_height / 2

    meta = {}
    overview = load_row(args, args.lat_min, args.lat_max, 0.5, "overview", meta)
    zoom = load_row(args, zlo, zhi, 0.1, f"zoom \N{PLUS-MINUS SIGN}{args.zoom_height/2:g}\N{DEGREE SIGN}", meta)

    finite_vals = np.concatenate([
        overview.hp_r[np.isfinite(overview.hp_r)].ravel(),
        overview.hp_s[np.isfinite(overview.hp_s)].ravel(),
    ])
    if finite_vals.size == 0:
        raise SystemExit("no finite values to plot")
    clim = args.clim if args.clim is not None else float(np.percentile(np.abs(finite_vals), 98))
    if clim <= 0:
        clim = 0.15
    print(f"shared clim = \N{PLUS-MINUS SIGN}{clim:.4g} m")

    ax_w = 3.2
    def row_h(row):
        lon_span = max(row.extent[1] - row.extent[0], 1e-6)
        lat_span = row.extent[3] - row.extent[2]
        return min(18.0, max(0.8, ax_w * lat_span / lon_span))
    h_ov, h_zm = row_h(overview), row_h(zoom)

    fig = plt.figure(figsize=(2 * ax_w + 2.5, h_ov + h_zm + 3.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[h_ov, h_zm], hspace=0.4, wspace=0.15)
    axes_ov = [fig.add_subplot(gs[0, c], projection=ccrs.PlateCarree()) for c in range(2)]
    axes_zm = [fig.add_subplot(gs[1, c], projection=ccrs.PlateCarree()) for c in range(2)]

    plot_row(axes_ov, overview, clim)
    mesh = plot_row(axes_zm, zoom, clim)

    fig.colorbar(mesh, ax=[*axes_ov, *axes_zm], orientation="horizontal", shrink=0.5,
                 pad=0.02, label=f"{args.hp_var} (m)")
    cyc, pas, date = meta.get("cycle", "?"), meta.get("pass", "?"), meta.get("date", "?")
    fig.suptitle(f"SWOT cycle {cyc} pass {pas}  \N{BULLET}  {date}", fontsize=13)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
