"""Quadmesh-regrid the CNN-Transformer ISW detections into a Zarr v3 multiscale pyramid.

Each SWOT half-orbit "slim" NetCDF (``cnn-transformer/results/cycle_*/*.nc``, one per pass) is
rasterised with ``datashader``'s curvilinear **quadmesh** onto a regular global lat/lon lattice
(default ~2 km / 0.02 deg) -- each source pixel is treated as a quad, so the swath comes out
gap-free and every output cell carries the value of the source quad covering it (no averaging at
level 0). One pass = one ``time`` slice.

Each pyramid level is ONE 4-D array ``iw[time, channel, y, x]`` -- the fields share a ``channel``
axis so the viewer's shader can read the quality-flag channel alongside whichever channel is
displayed (needed for a cross-field quality mask). Each channel key is the source NetCDF
variable it holds; ``--vars`` picks the channels, in order:

    internal_wave_bbox_confidence   (model probability 0..1;  agg = max)
    gaussian_filter_highpass        (SSHA-like high-pass, m;   agg = mean)
    quality_flag                    (uint8 flag -> f4;         agg = max)

Coarser levels are 2x2 block reductions of level 0. The store also carries a root ``slices.json``
+ identical ``attrs.slices`` (one ``{i, time, date, cycle, pass}`` record per pass, sorted by
(cycle, pass)), ``attrs.channels`` (index -> name/source/reducer) and ``attrs.multiscales``.

Passes are gridded in parallel across processes (each writes disjoint ``time`` slices).

Example
-------
    cd swot_isw_viewer
    python build_isw_store.py --limit 5 --resolution 0.08 --levels 2                 # smoke
    python build_isw_store.py --cycle-glob 'cycle_0[12]'                             # cycles 1-2
    python build_isw_store.py --cycle-glob 'cycle_*' \
        --vars internal_wave_bbox_confidence,quality_flag --workers 8               # all 50, no highpass
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# keep numba single-threaded: parallelism is across processes, one thread each
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import xarray as xr
import zarr
from zarr.codecs import BloscCodec, BloscShuffle

import datashader as dsh
from datashader import reductions as _rd

_HERE = Path(__file__).resolve().parent

# channel key  ==  the source NetCDF variable it holds (kept identical so the store's
# attrs.channels[].name is the real field name, not a viewer-legacy alias)
VAR_MAP = {
    "internal_wave_bbox_confidence": "internal_wave_bbox_confidence",
    "gaussian_filter_highpass": "gaussian_filter_highpass",
    "quality_flag": "quality_flag",
}
# how a cell is filled: "max" keeps the strongest, "mean" averages (signed field)
REDUCE = {
    "internal_wave_bbox_confidence": "max",
    "gaussian_filter_highpass": "mean",
    "quality_flag": "max",
}

_FNAME_RE = re.compile(r"SWOT_L3_LR_SSH_Expert_(\d{3})_(\d{3})_(\d{8})T(\d{6})_")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results-root", type=Path,
                   default=_HERE.parent / "cnn-transformer" / "results")
    p.add_argument("--cycle-glob", default="cycle_0[12]")
    p.add_argument("--out", type=Path, default=_HERE / "isw-store")
    p.add_argument("--vars", default=",".join(VAR_MAP),
                   help=f"Comma list of channel names, in order (subset of {list(VAR_MAP)}).")
    p.add_argument("--resolution", type=float, default=0.02,
                   help="Level-0 grid cell size in degrees.")
    p.add_argument("--levels", type=int, default=4,
                   help="Pyramid levels (level L cell size = resolution * 2**L). "
                        "Level-0 grid must divide by 2**(levels-1).")
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--limit", type=int, default=0, help="If >0, only the first N passes.")
    p.add_argument("--workers", type=int, default=0,
                   help="Parallel grid workers (0 = auto = min(8, cpu-2)).")
    return p.parse_args(argv)


def discover(results_root: Path, cycle_glob: str):
    cycle_dirs = sorted(d for d in results_root.glob(cycle_glob) if d.is_dir())
    if not cycle_dirs:
        raise FileNotFoundError(f"No '{cycle_glob}' folders under {results_root.resolve()}")
    rows = []
    for d in cycle_dirs:
        for nc in sorted(d.glob("*.nc")):
            m = _FNAME_RE.match(nc.name)
            if not m:
                print(f"  skip (unparsed name): {nc.name}")
                continue
            cyc, pas, ymd, hms = m.groups()
            start = datetime.strptime(ymd + hms, "%Y%m%d%H%M%S")
            rows.append({
                "path": str(nc), "cycle": int(cyc), "pass": int(pas),
                "time": start.strftime("%Y-%m-%dT%H:%M:%S"),
                "date": start.strftime("%Y-%m-%d"),
                "epoch": int(start.replace(tzinfo=timezone.utc).timestamp()),
            })
    rows.sort(key=lambda r: (r["cycle"], r["pass"]))
    for i, r in enumerate(rows):
        r["i"] = i
    return rows


def choose_lon(lon_raw: np.ndarray):
    """Pick the branch cut ([-180,180) or [0,360)) the pass does not straddle."""
    lon360 = lon_raw % 360.0
    lon180 = ((lon360 + 180.0) % 360.0) - 180.0
    def span(a):
        a = a[np.isfinite(a)]
        return float(a.max() - a.min()) if a.size else 0.0
    return lon180 if span(lon180) <= span(lon360) else lon360


def block_reduce(a: np.ndarray, how: str) -> np.ndarray:
    ny, nx = a.shape
    a = a[: ny - ny % 2, : nx - nx % 2]
    q = a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2)
    if how == "max":
        return np.fmax(np.fmax(q[:, 0, :, 0], q[:, 1, :, 0]),
                       np.fmax(q[:, 0, :, 1], q[:, 1, :, 1]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(q, axis=(1, 3)).astype("f4")


def lon_runs(x_centers: np.ndarray, res: float, nx_g: int):
    """Global longitude cell index per output column, split into contiguous runs."""
    k = np.floor(((x_centers + 180.0) % 360.0) / res).astype(np.int64) % nx_g
    breaks = np.nonzero((k[1:] - k[:-1]) != 1)[0] + 1
    bounds = [0, *breaks.tolist(), len(k)]
    return [(bounds[j], bounds[j + 1], int(k[bounds[j]])) for j in range(len(bounds) - 1)]


def grid_one_pass(arrs, path, i, res0, L, step, dims_ny, dims_nx, names):
    """Quadmesh one pass NetCDF and write it into time slice `i` of every level."""
    with xr.open_dataset(path) as ds:
        lat = np.asarray(ds["latitude"].values, dtype="f8")
        lon = choose_lon(np.asarray(ds["longitude"].values, dtype="f8"))
        fin = np.isfinite(lat) & np.isfinite(lon)
        if not fin.any():
            return False

        xmin, xmax = np.nanmin(lon[fin]), np.nanmax(lon[fin])
        ymin, ymax = np.nanmin(lat[fin]), np.nanmax(lat[fin])
        jx0 = int(np.floor(xmin / res0)) - 1
        jx1 = int(np.ceil(xmax / res0)) + 1
        jy0 = int(np.floor(ymin / res0)) - 1
        jy1 = int(np.ceil(ymax / res0)) + 1
        W0 = (jx1 - jx0)
        H0 = (jy1 - jy0)
        W0 += (-W0) % step
        H0 += (-H0) % step
        x_lo, y_lo = jx0 * res0, jy0 * res0
        xc0 = x_lo + (np.arange(W0) + 0.5) * res0
        yc0 = y_lo + (np.arange(H0) + 0.5) * res0
        y_runs_L = []
        for lev in range(L):
            resL = res0 * (2 ** lev)
            ycL = yc0[:: 2 ** lev][: H0 // (2 ** lev)] if lev else yc0
            kk = np.floor((ycL + 90.0) / resL).astype(np.int64)
            y_runs_L.append((0, len(kk), int(kk[0])))

        for ch, k in enumerate(names):
            src = np.asarray(ds[VAR_MAP[k]].values, dtype="f4")
            da = xr.DataArray(src, name="v", dims=("l", "p"),
                              coords={"latitude": (("l", "p"), lat),
                                      "longitude": (("l", "p"), lon)})
            agg = _rd.max("v") if REDUCE[k] == "max" else _rd.mean("v")
            out = dsh.Canvas(plot_width=W0, plot_height=H0,
                             x_range=(x_lo, x_lo + W0 * res0),
                             y_range=(y_lo, y_lo + H0 * res0)).quadmesh(
                da, x="longitude", y="latitude", agg=agg)
            grid = np.asarray(out.values, dtype="f4")
            for lev in range(L):
                if lev:
                    grid = block_reduce(grid, REDUCE[k])
                if not np.isfinite(grid).any():
                    continue
                resL = res0 * (2 ** lev)
                nx_gL = dims_nx[lev]
                r_lo, r_hi, gr_lo = y_runs_L[lev]
                xcL = xc0[:: 2 ** lev][: grid.shape[1]] if lev else xc0
                dst = arrs[lev]
                gr_hi = gr_lo + (r_hi - r_lo)
                for c_lo, c_hi, gc_lo in lon_runs(xcL, resL, nx_gL):
                    w = c_hi - c_lo
                    if gc_lo + w <= nx_gL:
                        dst[i, ch, gr_lo:gr_hi, gc_lo:gc_lo + w] = grid[r_lo:r_hi, c_lo:c_hi]
                    else:  # wrap across the +180 seam
                        cut = nx_gL - gc_lo
                        dst[i, ch, gr_lo:gr_hi, gc_lo:nx_gL] = grid[r_lo:r_hi, c_lo:c_lo + cut]
                        dst[i, ch, gr_lo:gr_hi, 0:w - cut] = grid[r_lo:r_hi, c_lo + cut:c_hi]
    return True


def _grid_shard(store_path, shard, res0, L, step, dims_ny, dims_nx, names, total):
    """Worker: grid a list of (i, path) into an already-created store."""
    g = zarr.open_group(store=zarr.storage.LocalStore(store_path), mode="r+")
    arrs = [g[str(lev)]["iw"] for lev in range(L)]
    done = ok = 0
    for i, path in shard:
        try:
            if grid_one_pass(arrs, path, i, res0, L, step, dims_ny, dims_nx, names):
                ok += 1
        except Exception as exc:  # noqa: BLE001 - report, keep going
            print(f"    pid {os.getpid()} pass i={i} {Path(path).name}: FAILED {exc}")
        done += 1
        if done % 25 == 0 or done == len(shard):
            print(f"    pid {os.getpid()}: {done}/{len(shard)} of its shard "
                  f"(store {ok}+ ok)", flush=True)
    return ok


def main(argv=None):
    args = parse_args(argv)
    names = [v.strip() for v in args.vars.split(",") if v.strip()]
    bad = [v for v in names if v not in VAR_MAP]
    if bad:
        raise SystemExit(f"unknown --vars entries: {bad}; choose from {list(VAR_MAP)}")

    rows = discover(args.results_root, args.cycle_glob)
    if args.limit:
        rows = rows[: args.limit]
        for i, r in enumerate(rows):
            r["i"] = i
    n = len(rows)
    if not n:
        raise SystemExit("no passes discovered")

    res0 = float(args.resolution)
    L = int(args.levels)
    ny0, nx0 = round(180.0 / res0), round(360.0 / res0)
    step = 2 ** (L - 1)
    if ny0 % step or nx0 % step:
        raise SystemExit(f"level-0 grid {ny0}x{nx0} not divisible by 2**(levels-1)={step}; "
                         f"reduce --levels or change --resolution")
    dims_ny = [ny0 // (2 ** k) for k in range(L)]
    dims_nx = [nx0 // (2 ** k) for k in range(L)]
    C = len(names)
    workers = args.workers or min(8, max(1, (os.cpu_count() or 2) - 2))

    cyc_counts = {}
    for r in rows:
        cyc_counts[r["cycle"]] = cyc_counts.get(r["cycle"], 0) + 1
    print(f"results_root = {args.results_root.resolve()}")
    print(f"passes = {n}  over {len(cyc_counts)} cycles  workers = {workers}")
    print(f"out = {args.out.resolve()}")
    print(f"resolution = {res0} deg  levels = {L}  L0 = {ny0}x{nx0}  Lmax = {dims_ny[-1]}x{dims_nx[-1]}")
    print(f"channels = {', '.join(f'{k} ({REDUCE[k]})' for k in names)}")

    # ---- create the store skeleton (metadata + coords), once ----
    store = zarr.storage.LocalStore(str(args.out))
    root = zarr.open_group(store=store, mode="w", zarr_format=3)
    comp = BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)
    dim_names = ["time", "channel", "latitude", "longitude"]
    for lev in range(L):
        nyL, nxL = dims_ny[lev], dims_nx[lev]
        resL = res0 * (2 ** lev)
        g = root.create_group(str(lev))
        a = g.create_array(name="latitude", shape=(nyL,), chunks=(nyL,), dtype="f4",
                           dimension_names=["latitude"], compressors=[comp])
        a[:] = (-90.0 + (np.arange(nyL) + 0.5) * resL).astype("f4")
        a = g.create_array(name="longitude", shape=(nxL,), chunks=(nxL,), dtype="f4",
                           dimension_names=["longitude"], compressors=[comp])
        a[:] = (-180.0 + (np.arange(nxL) + 0.5) * resL).astype("f4")
        a = g.create_array(name="channel", shape=(C,), chunks=(C,), dtype="i4",
                           dimension_names=["channel"], compressors=[comp])
        a[:] = np.arange(C, dtype="i4")
        if lev == 0:
            a = g.create_array(name="time", shape=(n,), chunks=(n,), dtype="i8",
                               dimension_names=["time"], compressors=[comp])
            a[:] = np.array([r["epoch"] for r in rows], dtype="i8")
        cs = (1, 1, min(args.chunk, nyL), min(args.chunk, nxL))
        g.create_array(name="iw", shape=(n, C, nyL, nxL), chunks=cs, dtype="f4",
                       fill_value=np.float32("nan"), dimension_names=dim_names,
                       compressors=[comp])

    with xr.open_dataset(rows[0]["path"]) as ds0:
        for attr in ("cnntrans_model_path", "cnntrans_variant", "cnntrans_confidence_threshold"):
            if attr in ds0.attrs:
                root["0"].attrs[attr] = _jsonable(ds0.attrs[attr])
        root["0"].attrs["source_example"] = Path(rows[0]["path"]).name

    # ---- grid all passes in parallel (disjoint time slices -> disjoint chunk files) ----
    shards = [[(r["i"], r["path"]) for r in rows[w::workers]] for w in range(workers)]
    shards = [s for s in shards if s]
    ok_total = 0
    if workers == 1 or len(shards) == 1:
        ok_total = _grid_shard(str(args.out), shards[0], res0, L, step, dims_ny, dims_nx, names, n)
    else:
        with ProcessPoolExecutor(max_workers=len(shards)) as ex:
            futs = [ex.submit(_grid_shard, str(args.out), s, res0, L, step,
                              dims_ny, dims_nx, names, n) for s in shards]
            for k, f in enumerate(as_completed(futs), 1):
                ok_total += f.result()
                print(f"  shard {k}/{len(futs)} finished  (store {ok_total}/{n} passes)", flush=True)

    # ---- finalise: slices.json + root attrs + consolidated metadata ----
    slices = [{"i": r["i"], "time": r["time"], "date": r["date"],
               "cycle": r["cycle"], "pass": r["pass"]} for r in rows]
    root.attrs["multiscales"] = [{
        "datasets": ([{"path": "0", "crs": "EPSG:4326"}]
                     + [{"path": str(lev)} for lev in range(1, L)]),
        "type": "reduce",
        "metadata": {
            "method": "datashader curvilinear quadmesh onto a regular grid; "
                      + ", ".join(f"{k} agg={REDUCE[k]}" for k in names)
                      + "; coarser levels 2x2 block-reduced",
            "version": "isw-0.4",
        },
    }]
    root.attrs["slices"] = slices
    root.attrs["channels"] = [
        {"index": ch, "name": k, "source": VAR_MAP[k], "reduce": REDUCE[k]}
        for ch, k in enumerate(names)
    ]
    root.attrs["variable"] = "iw"
    (args.out / "slices.json").write_text(json.dumps(slices, indent=0))
    zarr.consolidate_metadata(store)

    print(f"\nwrote {ok_total}/{n} slices to {args.out.resolve()}")
    print(f"  levels {list(range(L))}  0/iw shape = {(n, C, dims_ny[0], dims_nx[0])}  "
          f"channels = {names}")
    print(f"  slices.json = {len(slices)} records (cycles {sorted({s['cycle'] for s in slices})})")
    print("\nserve it, then open:")
    print("  python serve_isw.py 8000")
    print("  http://localhost:8000/swot-viewer-secure/index.html?store=http://localhost:8000/isw-store")
    return 0


def _jsonable(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


if __name__ == "__main__":
    sys.exit(main())
