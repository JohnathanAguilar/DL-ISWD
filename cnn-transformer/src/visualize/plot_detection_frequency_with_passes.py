"""Global detection-frequency map plus one example SWOT pass per named region.

Combines two existing plots into one figure:

  * TOP    -- the ``plot_detection_frequency`` global per-block map (blocks colored by
    how many cycles clear the |lat|-banded confidence cutoff), with each named
    ``REGIONS`` box outlined and labeled by its share of total detection counts;
  * BOTTOM -- for every region, the single SWOT pass with the most in-box
    ``internal_wave_bbox_confidence`` > ``--section-threshold`` pixels (picked the same
    way as ``plot_pass_section.select_passes``), rendered as a small cropped map (high-pass
    SSHA + translucent confidence-mask overlay). A dashed line connects each region's box
    on the main map to its example-pass panel below.

Example
-------
    cd cnn-transformer
    python src/inference/plot_detection_frequency_with_passes.py \\
      --threshold 0.95 0.98 0.998 --band-edges 30 60 90 --workers 8 \\
      --section-threshold 0.95 \\
      --out results/all_cycles/detection_frequency_with_passes.png
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # parallelism is across processes

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from matplotlib.patches import ConnectionPatch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

from plot_detection_frequency import (
    REGIONS,
    wrap_longitude,
    band_labels,
    describe_thresholds,
    process_pass,
    _FNAME_RE as _FREQ_FNAME_RE,
)
from plot_pass_section import select_passes, _land_features

_INFERENCE_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _INFERENCE_DIR.parent.parent  # cnn-transformer/


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results-root", type=Path, default=_MODEL_DIR / "results")
    p.add_argument("--cycle-glob", default="cycle_*")
    p.add_argument("--conf-var", default="internal_wave_bbox_confidence")
    p.add_argument("--hp-var", default="gaussian_filter_highpass")

    # main frequency-map options (see plot_detection_frequency.py)
    p.add_argument("--threshold", type=float, nargs="+", default=[0.90, 0.95, 1.0],
                   metavar="CONF",
                   help="Block-mean confidence cutoff: one value, or one per |latitude| "
                        "band (see --band-edges), ordered from the equator outward.")
    p.add_argument("--band-edges", type=float, nargs="+", default=[30.0, 60.0, 90.0],
                   metavar="DEG")
    p.add_argument("--block", type=int, default=20)
    p.add_argument("--min-pixels", type=int, default=1)
    p.add_argument("--min-count", type=int, default=1)
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--cmap", default="inferno", help="Colormap for the main frequency map.")
    p.add_argument("--scatter-size", type=float, default=1.0)

    # example-pass panel options (see plot_pass_section.py)
    p.add_argument("--section-threshold", type=float, default=0.95,
                   help="Per-pixel confidence cutoff used to pick each region's top "
                        "pass and to draw its mask overlay.")
    p.add_argument("--margin", type=float, default=0.5,
                   help="Degrees of margin around each region box for the pass crop.")
    p.add_argument("--clim", type=float, default=None,
                   help="Symmetric high-pass colour limit (m) for panels. "
                        "Default: each panel's own 98th pct of |hp|.")
    p.add_argument("--panel-cmap", default="gray",
                   help="Colormap for the panel high-pass SSHA base.")
    p.add_argument("--mask-color", default="#00b8b8")
    p.add_argument("--mask-alpha", type=float, default=0.30)

    p.add_argument("--regions", nargs="+", default=None, metavar="NAME",
                   help="Subset of REGIONS to show a panel for (default: all).")
    p.add_argument("--out", type=Path,
                   default=_MODEL_DIR / "results/all_cycles/detection_frequency_with_passes.png")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, only the first N pass numbers (smoke run).")
    p.add_argument("--workers", type=int, default=0,
                   help="Parallel workers (0 = auto = min(8, cpu-2)).")
    return p.parse_args(argv)


def render_pass_panel(ax, cyc, p, path, date, box, margin, hp_var, conf_var,
                       threshold, clim, cmap, mask_color, mask_alpha):
    """Draw a cropped high-pass-SSHA + confidence-mask map for one pass onto ``ax``.

    Adapted from the ``"map"`` layout of ``plot_pass_section.plot_one``, but renders
    into an existing axes instead of creating/saving its own figure. Returns None if
    the pass has no data inside ``box ± margin``.
    """
    L0, L1, A0, A1 = box
    mg = margin
    with xr.open_dataset(path) as ds:
        lat = np.asarray(ds["latitude"].values, dtype="f8")
        lon = wrap_longitude(np.asarray(ds["longitude"].values, dtype="f8"))
        hp = np.asarray(ds[hp_var].values, dtype="f4")
        conf = np.asarray(ds[conf_var].values, dtype="f4")

    near = ((lon >= L0 - mg) & (lon <= L1 + mg)
            & (lat >= A0 - mg) & (lat <= A1 + mg))
    rows = np.where(np.any(near, axis=1))[0]
    if rows.size == 0:
        return None
    r0, r1 = rows.min(), rows.max() + 1
    sl = (slice(r0, r1), slice(None))
    lat, lon, hp, conf = lat[sl], lon[sl], hp[sl], conf[sl]

    hpr = np.where(np.any(np.isfinite(hp), axis=1))[0]
    if hpr.size:
        k0, k1 = hpr.min(), hpr.max() + 1
        lat, lon, hp, conf = lat[k0:k1], lon[k0:k1], hp[k0:k1], conf[k0:k1]

    fin = np.isfinite(lat) & np.isfinite(lon)
    if not fin.any():
        return None
    ext = [max(L0 - mg, float(np.min(lon[fin]))), min(L1 + mg, float(np.max(lon[fin]))),
           max(A0 - mg, float(np.min(lat[fin]))), min(A1 + mg, float(np.max(lat[fin])))]

    if clim is None:
        a = np.abs(hp[np.isfinite(hp)])
        panel_clim = float(np.percentile(a, 98)) if a.size else 0.15
    else:
        panel_clim = clim

    nmask = int(np.sum(conf > threshold))
    mmask = np.ma.masked_where(~(conf > threshold), np.ones_like(conf, dtype="f4"))
    mask_cmap = ListedColormap([mask_color])

    ax.set_extent(ext, crs=ccrs.PlateCarree())
    _land_features(ax)
    ax.pcolormesh(lon, lat, np.ma.masked_invalid(hp), cmap=cmap,
                  vmin=-panel_clim, vmax=panel_clim, shading="nearest",
                  transform=ccrs.PlateCarree(), zorder=2)
    ax.pcolormesh(lon, lat, mmask, cmap=mask_cmap, vmin=0, vmax=1,
                  alpha=mask_alpha, shading="nearest",
                  transform=ccrs.PlateCarree(), zorder=3)
    gl = ax.gridlines(draw_labels=False, linewidth=0.2, color="0.6", alpha=0.4)

    ax.set_title(f"c{cyc} p{p}  \N{BULLET}  {date[:4]}-{date[4:6]}-{date[6:]}\n"
                 f"{nmask:,} px conf>{threshold:g}", fontsize=8)
    return ext


def build_pass_region_membership(by_pass, passes, regions_dict, margin=0.0):
    """One lat/lon read per pass number -> {region name: set of pass numbers whose
    ground track enters that region's box}.

    SWOT repeats the same ground track for a given pass number every cycle (see the
    module docstring in plot_detection_frequency.py), so a single cycle's footprint
    stands in for all of that pass number's files. This turns the region -> best-pass
    scan from an O(files x regions) full read into an O(passes) footprint pass plus a
    much smaller O(candidate files) confidence scan per region.
    """
    membership = defaultdict(set)
    for p in passes:
        _cyc, path = sorted(by_pass[p])[0]
        try:
            with xr.open_dataset(path) as ds:
                lat = np.asarray(ds["latitude"].values, dtype="f4")
                lon = wrap_longitude(np.asarray(ds["longitude"].values, dtype="f4"))
        except Exception as exc:  # noqa: BLE001
            print(f"    pass {p}: footprint read FAILED: {exc}", flush=True)
            continue
        fin = np.isfinite(lat) & np.isfinite(lon)
        if not fin.any():
            continue
        lat_f, lon_f = lat[fin], lon[fin]
        for name, (lo, hi, la0, la1) in regions_dict.items():
            if np.any((lon_f >= lo - margin) & (lon_f <= hi + margin)
                      & (lat_f >= la0 - margin) & (lat_f <= la1 + margin)):
                membership[name].add(p)
    return membership


def main(argv=None):
    args = parse_args(argv)

    cycle_dirs = sorted(p for p in args.results_root.glob(args.cycle_glob) if p.is_dir())
    if not cycle_dirs:
        raise FileNotFoundError(
            f"No '{args.cycle_glob}' folders under {args.results_root.resolve()}"
        )

    band_edges = [float(e) for e in args.band_edges]
    if band_edges[0] <= 0 or any(b <= a for a, b in zip(band_edges, band_edges[1:])):
        raise SystemExit("--band-edges must be strictly increasing and positive")
    thresholds = [float(t) for t in args.threshold]
    if len(thresholds) == 1:
        thresholds = thresholds * len(band_edges)
    elif len(thresholds) != len(band_edges):
        raise SystemExit(
            f"got {len(thresholds)} --threshold value(s) for {len(band_edges)} "
            f"|latitude| band(s); pass 1 or {len(band_edges)}"
        )

    region_names = args.regions or list(REGIONS)
    unknown = [n for n in region_names if n not in REGIONS]
    if unknown:
        raise SystemExit(f"unknown region(s) {unknown}; known: {list(REGIONS)}")

    by_pass = defaultdict(list)
    n_files = 0
    for cdir in cycle_dirs:
        for nc in cdir.glob("*.nc"):
            m = _FREQ_FNAME_RE.search(nc.name)
            if not m:
                continue
            by_pass[int(m.group(2))].append((int(m.group(1)), str(nc)))
            n_files += 1
    passes = sorted(by_pass)
    if args.limit:
        passes = passes[: args.limit]
    if not passes:
        raise SystemExit("no passes discovered")

    workers = args.workers or min(8, max(1, (os.cpu_count() or 2) - 2))
    thr_desc = describe_thresholds(thresholds, band_edges)
    band_desc = " ".join(f"{lbl}\N{DEGREE SIGN}:{t:g}"
                         for lbl, t in zip(band_labels(band_edges), thresholds))
    print(f"results_root={args.results_root.resolve()}")
    print(f"cycles={len(cycle_dirs)} passes={len(passes)} files={n_files} "
          f"block={args.block}px workers={workers}")
    print(f"conf_var={args.conf_var} thresholds[{band_desc}] "
          f"min_pixels={args.min_pixels} min_count={args.min_count}")

    lons, lats, cnts, nobs = [], [], [], []
    common = (args.conf_var, thresholds, band_edges, args.block, args.min_pixels,
              args.min_count)
    done = 0
    if workers == 1:
        for p in passes:
            lo, la, c, no = process_pass(by_pass[p], *common)
            lons.append(lo); lats.append(la); cnts.append(c); nobs.append(no)
            done += 1
            if done % 50 == 0 or done == len(passes):
                print(f"  {done}/{len(passes)} passes", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(process_pass, by_pass[p], *common): p for p in passes}
            for f in as_completed(futs):
                lo, la, c, no = f.result()
                lons.append(lo); lats.append(la); cnts.append(c); nobs.append(no)
                done += 1
                if done % 50 == 0 or done == len(passes):
                    print(f"  {done}/{len(passes)} passes", flush=True)

    lon = np.concatenate(lons) if lons else np.empty(0, np.float32)
    lat = np.concatenate(lats)
    cnt = np.concatenate(cnts)
    nob = np.concatenate(nobs)
    if cnt.size == 0:
        raise SystemExit("no blocks cleared the cutoff in >= --min-count cycles")

    order = np.argsort(cnt, kind="stable")  # highest count drawn last / on top
    lon, lat, cnt, nob = lon[order], lat[order], cnt[order], nob[order]

    vmax = int(args.vmax) if args.vmax is not None else int(cnt.max())
    print(f"kept blocks={cnt.size:,} | count: median={int(np.median(cnt))} "
          f"max={int(cnt.max())} | observed cycles/block: median={int(np.median(nob))} "
          f"max={int(nob.max())} | vmax={vmax}")

    total = int(cnt.astype(np.int64).sum())
    print(f"region share of total detection counts ({total:,}):")
    region_rows = []
    for name, (lo, hi, la0, la1) in REGIONS.items():
        m = (lon >= lo) & (lon < hi) & (lat >= la0) & (lat < la1)
        rc = int(cnt[m].astype(np.int64).sum())
        pct = 100.0 * rc / total if total else 0.0
        print(f"  {name:4s} lon[{lo:g},{hi:g}] lat[{la0:g},{la1:g}]  "
              f"blocks={int(m.sum()):>6,}  counts={rc:>10,}  {pct:6.2f}%")
        region_rows.append((name, lo, hi, la0, la1, pct))

    print(f"scanning {len(passes)} passes' ground tracks for region overlap ...")
    membership = build_pass_region_membership(
        by_pass, passes, {n: REGIONS[n] for n in region_names}, margin=args.margin)

    print(f"selecting each region's top pass (conf > {args.section_threshold:g}) ...")
    panel_data = {}
    for name in region_names:
        box = REGIONS[name]
        cand_passes = membership.get(name, ())
        cand_files = [path for p in cand_passes for (_cyc, path) in by_pass[p]]
        if not cand_files:
            print(f"  {name}: no pass ground track enters the box, panel skipped")
            continue
        chosen = select_passes(cand_files, box, args.conf_var, args.section_threshold, n=1)
        if not chosen:
            print(f"  {name}: {len(cand_passes)} candidate pass(es), "
                  f"none clears conf > {args.section_threshold:g}, panel skipped")
            continue
        cyc, p, path, date, nd = chosen[0]
        print(f"  {name}: cycle {cyc} pass {p} {date}  "
              f"in-box conf>{args.section_threshold:g} px = {nd:,}  "
              f"(from {len(cand_passes)} candidate pass(es), {len(cand_files)} files)")
        panel_data[name] = (cyc, p, path, date)

    # ---- figure ----
    # main map on top (small); every example-pass panel in a single row below it,
    # side by side, left to right in PO/AR/IO/BoB/IDN/SCS order.
    left_order = ("PO", "AR", "IO")
    ordered_names = [n for n in left_order if n in region_names]
    ordered_names += [n for n in region_names if n not in ordered_names]
    NR = len(ordered_names)

    fig = plt.figure(figsize=(max(16.0, 4.5 * NR), 12.0))
    gs = GridSpec(nrows=2, ncols=NR, height_ratios=[0.8, 3.6],
                  hspace=0.15, wspace=0.15, figure=fig)

    ax_main = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
    ax_main.set_global()
    ax_main.add_feature(cfeature.LAND, facecolor="0.85", zorder=0)
    ax_main.coastlines(linewidth=0.4)

    ax_main.scatter(lon, lat, c=cnt, s=args.scatter_size, marker="s", linewidths=0,
                    cmap=args.cmap, vmin=args.min_count, vmax=max(vmax, args.min_count),
                    transform=ccrs.PlateCarree(), rasterized=True)

    for name, lo, hi, la0, la1, pct in region_rows:
        ax_main.add_patch(mpatches.Rectangle(
            (lo, la0), hi - lo, la1 - la0, transform=ccrs.PlateCarree(),
            fill=False, edgecolor="#ed0909", linewidth=1.1, zorder=5))
        ax_main.text(lo, la1 + 1.0, f"{name} {pct:.1f}%", transform=ccrs.PlateCarree(),
                     fontsize=8, color="#ed0909", weight="bold", va="bottom", zorder=6,
                     clip_on=True)

    gl = ax_main.gridlines(draw_labels=True, linewidth=0.3, color="0.6", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    ax_main.set_title(
        f"SWOT internal-wave detection frequency per {args.block}\N{MULTIPLICATION SIGN}{args.block} "
        f"block  |  {len(cycle_dirs)} cycles  |  {thr_desc}"
    )

    box_by_name = {name: (lo, hi, la0, la1) for name, lo, hi, la0, la1, _pct in region_rows}

    for i, name in enumerate(ordered_names):
        ax_panel = fig.add_subplot(gs[1, i], projection=ccrs.PlateCarree())
        ax_panel.set_title(name, fontsize=10, weight="bold")
        if name not in panel_data:
            ax_panel.axis("off")
            ax_panel.text(0.5, 0.5, "no example pass", ha="center", va="center",
                          fontsize=8, transform=ax_panel.transAxes)
            continue

        cyc, p, path, date = panel_data[name]
        ext = render_pass_panel(
            ax_panel, cyc, p, path, date, REGIONS[name], args.margin, args.hp_var,
            args.conf_var, args.section_threshold, args.clim, args.panel_cmap,
            args.mask_color, args.mask_alpha,
        )
        if ext is None:
            ax_panel.axis("off")
            ax_panel.text(0.5, 0.5, "no in-box data", ha="center", va="center",
                          fontsize=8, transform=ax_panel.transAxes)
            continue

        lo, hi, la0, la1 = box_by_name[name]
        con = ConnectionPatch(
            xyA=((lo + hi) / 2.0, la0), coordsA="data", axesA=ax_main,
            xyB=(0.5, 1.0), coordsB="axes fraction", axesB=ax_panel,
            linestyle="--", linewidth=0.8, color="#555555", zorder=1,
        )
        fig.add_artist(con)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
