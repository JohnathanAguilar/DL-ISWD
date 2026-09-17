"""Plot a SWOT pass cropped to a lon/lat box: high-pass SSHA base + cyan ISW-detection mask.

Scans ``results/cycle_*/*.nc`` for the passes with the most ``internal_wave_bbox_confidence``
> threshold pixels inside the box, then for the top N (distinct pass numbers) draws the swath
section that falls in the box at native pixel geolocation:

  * base  -- ``gaussian_filter_highpass`` (high-pass SSHA, m) as a diverging pcolormesh;
  * overlay -- a translucent cyan mask wherever confidence > --threshold.

Example
-------
    cd cnn-transformer
    python src/inference/plot_pass_section.py \\
      --box -40 -25 -65 -55 --threshold 0.95 --n-passes 3 \\
      --out-dir results/all_cycles
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from plot_detection_frequency import wrap_longitude

_INFERENCE_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _INFERENCE_DIR.parent.parent  # cnn-transformer/

_FNAME_RE = re.compile(r"SWOT_L3_LR_SSH_Expert_(\d{3})_(\d{3})_(\d{8})T")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results-root", type=Path, default=_MODEL_DIR / "results")
    p.add_argument("--cycle-glob", default="cycle_*")
    p.add_argument("--box", type=float, nargs=4, default=[-40.0, -25.0, -65.0, -55.0],
                   metavar=("LON0", "LON1", "LAT0", "LAT1"))
    p.add_argument("--conf-var", default="internal_wave_bbox_confidence")
    p.add_argument("--hp-var", default="gaussian_filter_highpass")
    p.add_argument("--threshold", type=float, default=0.95)
    p.add_argument("--n-passes", type=int, default=3,
                   help="Number of distinct pass numbers to plot (top by in-box detections).")
    p.add_argument("--passes", nargs="+", default=None, metavar="C:P",
                   help="Explicit cycle:pass list; bypasses the scan.")
    p.add_argument("--clim", type=float, default=None,
                   help="Symmetric high-pass colour limit (m). Default: 98th pct of |hp| in crop.")
    p.add_argument("--cmap", default="gray",
                   help="Colormap for the high-pass SSHA base (grayscale by default).")
    p.add_argument("--mask-color", default="#00b8b8",
                   help="Colour of the confidence>threshold overlay (a slightly dark cyan).")
    p.add_argument("--mask-alpha", type=float, default=0.30)
    p.add_argument("--margin", type=float, default=0.5,
                   help="Degrees of margin around the box for the map extent + crop.")
    p.add_argument("--layout", choices=("swath", "map"), default="swath",
                   help="'swath' = vertical pcolormesh in array coords (along-track up); "
                        "'map' = cartopy PlateCarree at true geolocation.")
    p.add_argument("--out-dir", type=Path, default=_MODEL_DIR / "results/all_cycles")
    return p.parse_args(argv)


def _land_features(ax):
    for scale in ("50m", "110m"):
        try:
            ax.add_feature(cfeature.LAND.with_scale(scale), facecolor="0.85", zorder=0)
            ax.coastlines(scale, linewidth=0.5, zorder=4)
            return
        except Exception:  # noqa: BLE001 - offline / missing Natural Earth data
            continue


def select_passes(files, box, conf_var, threshold, n):
    L0, L1, A0, A1 = box
    best = {}  # pass number -> (ndet, cycle, path, date)
    for f in files:
        m = _FNAME_RE.search(Path(f).name)
        if not m:
            continue
        cyc, p, date = int(m.group(1)), int(m.group(2)), m.group(3)
        try:
            with xr.open_dataset(f) as ds:
                lat = ds["latitude"].values
                lon = wrap_longitude(ds["longitude"].values)
                conf = ds[conf_var].values
        except Exception as exc:  # noqa: BLE001
            print(f"  {Path(f).name}: FAILED: {exc}")
            continue
        inbox = (lon >= L0) & (lon <= L1) & (lat >= A0) & (lat <= A1)
        ndet = int(np.sum(inbox & np.isfinite(conf) & (conf > threshold)))
        if ndet and (p not in best or ndet > best[p][0]):
            best[p] = (ndet, cyc, f, date)
    ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)
    return [(cyc, p, path, date, nd) for p, (nd, cyc, path, date) in ranked[:n]]


def resolve_passes(spec, results_root, cycle_glob):
    out = []
    for s in spec:
        cyc, p = (int(x) for x in s.split(":"))
        cand = list(results_root.glob(f"cycle_{cyc:02d}/SWOT_L3_LR_SSH_Expert_"
                                      f"{cyc:03d}_{p:03d}_*.nc"))
        if not cand:
            raise SystemExit(f"no file for cycle {cyc} pass {p}")
        m = _FNAME_RE.search(cand[0].name)
        out.append((cyc, p, str(cand[0]), m.group(3), -1))
    return out


def plot_one(cyc, p, path, date, ndet, args):
    L0, L1, A0, A1 = args.box
    mg = args.margin
    with xr.open_dataset(path) as ds:
        lat = np.asarray(ds["latitude"].values, dtype="f8")
        lon = wrap_longitude(np.asarray(ds["longitude"].values, dtype="f8"))
        hp = np.asarray(ds[args.hp_var].values, dtype="f4")
        conf = np.asarray(ds[args.conf_var].values, dtype="f4")

    near = ((lon >= L0 - mg) & (lon <= L1 + mg)
            & (lat >= A0 - mg) & (lat <= A1 + mg))
    rows = np.where(np.any(near, axis=1))[0]
    if rows.size == 0:
        print(f"  c{cyc} p{p}: no rows in box, skipped")
        return None
    r0, r1 = rows.min(), rows.max() + 1
    sl = (slice(r0, r1), slice(None))
    lat, lon, hp, conf = lat[sl], lon[sl], hp[sl], conf[sl]

    # trim leading/trailing all-NaN rows (ice/land mask beyond the swath data)
    hpr = np.where(np.any(np.isfinite(hp), axis=1))[0]
    if hpr.size:
        k0, k1 = hpr.min(), hpr.max() + 1
        lat, lon, hp, conf = lat[k0:k1], lon[k0:k1], hp[k0:k1], conf[k0:k1]

    # extent = the swath's own bbox, clipped to the (margin-expanded) box, so the
    # section fills the frame instead of leaving the rest of the box blank
    fin = np.isfinite(lat) & np.isfinite(lon)
    ext = [max(L0 - mg, float(np.min(lon[fin]))), min(L1 + mg, float(np.max(lon[fin]))),
           max(A0 - mg, float(np.min(lat[fin]))), min(A1 + mg, float(np.max(lat[fin])))]

    clim = args.clim
    if clim is None:
        a = np.abs(hp[np.isfinite(hp)])
        clim = float(np.percentile(a, 98)) if a.size else 0.15
    nmask = int(np.sum(conf > args.threshold))
    mmask = np.ma.masked_where(~(conf > args.threshold), np.ones_like(conf, dtype="f4"))
    mask_cmap = ListedColormap([args.mask_color])
    title = (f"SWOT cycle {cyc}  pass {p}  \N{BULLET}  {date[:4]}-{date[4:6]}-{date[6:]}"
             f"   \N{BULLET}   {nmask:,} px  conf > {args.threshold:g}")
    mask_legend = Patch(facecolor=args.mask_color, alpha=args.mask_alpha,
                        label=f"confidence > {args.threshold:g}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.layout == "swath":
        ny, nx = hp.shape
        # each array cell rendered as a true square (aspect = 'equal'); size the
        # figure to that real ny:nx ratio so nothing is stretched
        ax_w = 3.0
        ax_h = min(24.0, ax_w * ny / nx)
        fig, ax = plt.subplots(figsize=(ax_w + 3.2, ax_h + 1.8))
        mesh = ax.pcolormesh(np.ma.masked_invalid(hp), cmap=args.cmap, vmin=-clim, vmax=clim)
        ax.pcolormesh(mmask, cmap=mask_cmap, vmin=0, vmax=1, alpha=args.mask_alpha)
        ax.set_aspect("equal")
        mid = nx // 2
        latcol = lat[:, mid]
        d = np.diff(latcol[np.isfinite(latcol)])
        if d.size and np.nanmean(d) < 0:      # row 0 is the northern end -> put north up
            ax.invert_yaxis()
        yt = np.unique(np.linspace(0, ny - 1, 8).round().astype(int))
        ax.set_yticks(yt + 0.5)
        ax.set_yticklabels([f"{latcol[i]:.1f}\N{DEGREE SIGN}" if np.isfinite(latcol[i]) else "—"
                            for i in yt])
        ax.set_xlabel("cross-track pixel")
        ax.set_ylabel("latitude (mid-swath)")
        fig.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.9, pad=0.04,
                     label=f"{args.hp_var} — high-pass SSHA (m)")
        ax.legend(handles=[mask_legend], loc="lower left", fontsize=8, framealpha=0.9)
        ax.set_title(title, fontsize=9)
        out = args.out_dir / f"pass_section_swath_c{cyc:03d}p{p:03d}_{date}.png"
    else:
        fig = plt.figure(figsize=(9, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent(ext, crs=ccrs.PlateCarree())
        _land_features(ax)
        mesh = ax.pcolormesh(lon, lat, np.ma.masked_invalid(hp), cmap=args.cmap,
                             vmin=-clim, vmax=clim, shading="nearest",
                             transform=ccrs.PlateCarree(), zorder=2)
        ax.pcolormesh(lon, lat, mmask, cmap=mask_cmap, vmin=0, vmax=1,
                      alpha=args.mask_alpha, shading="nearest",
                      transform=ccrs.PlateCarree(), zorder=3)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.6", alpha=0.5)
        gl.top_labels = gl.right_labels = False
        fig.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.75, pad=0.03,
                     label=f"{args.hp_var} — high-pass SSHA (m)")
        ax.legend(handles=[mask_legend], loc="lower left", fontsize=9, framealpha=0.9)
        ax.set_title(title, fontsize=11)
        out = args.out_dir / f"pass_section_c{cyc:03d}p{p:03d}_{date}.png"

    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"  c{cyc} p{p} {date}: crop {hp.shape}  clim=\N{PLUS-MINUS SIGN}{clim:.3g} m  "
          f"mask_px={nmask:,}  -> {out}")
    return out


def main(argv=None):
    args = parse_args(argv)
    L0, L1, A0, A1 = args.box
    print(f"box: lon[{L0:g},{L1:g}] lat[{A0:g},{A1:g}]  threshold={args.threshold:g}")

    if args.passes:
        chosen = resolve_passes(args.passes, args.results_root, args.cycle_glob)
    else:
        files = sorted(
            str(p) for d in args.results_root.glob(args.cycle_glob) if d.is_dir()
            for p in d.glob("*.nc")
        )
        if not files:
            raise FileNotFoundError(f"no .nc under {args.results_root.resolve()}")
        print(f"scanning {len(files)} files for the top {args.n_passes} passes ...")
        chosen = select_passes(files, args.box, args.conf_var, args.threshold, args.n_passes)
        if not chosen:
            raise SystemExit("no pass has a detection in the box")

    print("selected:")
    for cyc, p, _path, date, nd in chosen:
        print(f"  cycle {cyc:>3}  pass {p:>3}  {date}  "
              + (f"in-box conf>{args.threshold:g} px = {nd:,}" if nd >= 0 else "(explicit)"))
    print("plotting:")
    for rec in chosen:
        plot_one(*rec, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
