"""Render one or more GeoTIFFs (e.g. the outputs of ``plot_highpass_variance_geotiff.py``
/ ``plot_detection_frequency_geotiff.py``) as a self-contained Leaflet map you can open
in a web browser.

Every requested band of every input file becomes a toggleable overlay layer (via
``folium.LayerControl``), colour-mapped with matplotlib and embedded directly in the
saved HTML as a PNG data URI -- no tile server, no extra files, works offline once
saved (only the basemap tiles are fetched over the network when the page is opened).

Requires ``rasterio``, ``folium`` and ``branca``. On this machine only the
``gis-stuff`` conda env has them: ``/Users/aguilarj/miniforge3/envs/gis-stuff/bin/python``.

Example
-------
    cd cnn-transformer
    /Users/aguilarj/miniforge3/envs/gis-stuff/bin/python \\
      src/inference/visualize_geotiff.py \\
      --geotiff results/all_cycles/highpass_variance_perblock_banded_0.95-0.98-0.998.tif \\
      --log

    # compare both maps in one page:
    /Users/aguilarj/miniforge3/envs/gis-stuff/bin/python \\
      src/inference/visualize_geotiff.py \\
      --geotiff results/all_cycles/highpass_variance_perblock_banded_0.95-0.98-0.998.tif \\
                results/all_cycles/detection_frequency_perblock_banded_0.90-0.95-1.0.tif
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

import numpy as np
import rasterio
import rasterio.warp
import matplotlib
import matplotlib.colors as mcolors
import branca.colormap as bcm
import folium
from folium.raster_layers import ImageOverlay

_INFERENCE_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _INFERENCE_DIR.parent.parent  # cnn-transformer/

_BASEMAPS = {
    "OpenStreetMap": "OpenStreetMap",
    "CartoDB positron": "CartoDB positron",
    "CartoDB dark_matter": "CartoDB dark_matter",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--geotiff", type=Path, nargs="+", required=True,
                   help="One or more GeoTIFFs to overlay together.")
    p.add_argument("--band", type=int, nargs="+", default=None,
                   help="Band index/indices to render (1-based). Default: every "
                        "band in each file.")
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--log", action="store_true",
                   help="Log-scale colour normalisation (non-positive values are "
                        "treated as NoData).")
    p.add_argument("--clim-quantile", type=float, nargs=2, default=(0.0, 0.98),
                   metavar=("LO", "HI"),
                   help="Colour-limit quantiles over the finite per-band values, "
                        "fractions in [0,1]. --vmin/--vmax override the matching end.")
    p.add_argument("--vmin", type=float, default=None,
                   help="Bottom of the colour scale, applied to every rendered band.")
    p.add_argument("--vmax", type=float, default=None,
                   help="Top of the colour scale, applied to every rendered band.")
    p.add_argument("--opacity", type=float, default=0.8)
    p.add_argument("--basemap", choices=sorted(_BASEMAPS), default="CartoDB positron")
    p.add_argument("--out", type=Path, default=None,
                   help="Output HTML path. Default: alongside the first --geotiff, "
                        "same stem + .html")
    p.add_argument("--open", action=argparse.BooleanOptionalAction, default=True,
                   help="Open the saved HTML in the default browser (--no-open to skip).")
    return p.parse_args(argv)


def load_band(path: Path, band: int):
    """Read one band as float64 with NoData -> NaN, plus its WGS84 bounds/description."""
    with rasterio.open(path) as src:
        data = src.read(band).astype(np.float64)
        nodata = src.nodata
        if nodata is not None and not np.isnan(nodata):
            data[data == nodata] = np.nan
        mask = src.read_masks(band)
        data[mask == 0] = np.nan

        west, south, east, north = rasterio.warp.transform_bounds(
            src.crs, "EPSG:4326", *src.bounds
        )
        desc = src.descriptions[band - 1] if src.descriptions else None
    return data, (south, west, north, east), desc


def colorize(data: np.ndarray, cmap_name: str, log: bool, vmin, vmax, q_lo, q_hi):
    d = data.copy()
    if log:
        d = np.where(d > 0, d, np.nan)

    finite = d[np.isfinite(d)]
    if finite.size == 0:
        raise SystemExit("band has no finite values to render")

    lo = vmin if vmin is not None else float(np.nanquantile(finite, q_lo))
    hi = vmax if vmax is not None else float(np.nanquantile(finite, q_hi))
    if hi <= lo:
        hi = lo + (abs(lo) or 1.0) * 1e-3

    norm = mcolors.LogNorm(vmin=lo, vmax=hi) if log else mcolors.Normalize(vmin=lo, vmax=hi)
    cmap = matplotlib.colormaps[cmap_name]
    rgba = cmap(norm(d), bytes=True)  # NaN -> (0, 0, 0, 0), fully transparent
    return rgba, lo, hi


def main(argv=None):
    args = parse_args(argv)
    q_lo, q_hi = args.clim_quantile
    if not (0.0 <= q_lo < q_hi <= 1.0):
        raise SystemExit("--clim-quantile LO HI must satisfy 0 <= LO < HI <= 1")

    out = args.out or args.geotiff[0].with_suffix(".html")

    layers = []  # (rgba, bounds, name, lo, hi)
    for path in args.geotiff:
        if not path.exists():
            raise SystemExit(f"not found: {path}")
        with rasterio.open(path) as src:
            n_bands = src.count
        bands = args.band if args.band else list(range(1, n_bands + 1))

        for b in bands:
            data, bounds, desc = load_band(path, b)
            rgba, lo, hi = colorize(
                data, args.cmap, args.log, args.vmin, args.vmax, q_lo, q_hi
            )
            label = f"{path.stem} / band {b}" + (f" ({desc})" if desc else "")
            n_finite = int(np.isfinite(data).sum())
            print(f"{label}: bounds={tuple(round(v, 3) for v in bounds)} "
                  f"finite_cells={n_finite:,} vmin={lo:.4g} vmax={hi:.4g} log={args.log}")
            layers.append((rgba, bounds, label, lo, hi))

    south = min(b[0] for _, b, *_ in layers)
    west = min(b[1] for _, b, *_ in layers)
    north = max(b[2] for _, b, *_ in layers)
    east = max(b[3] for _, b, *_ in layers)

    m = folium.Map(tiles=_BASEMAPS[args.basemap])
    m.fit_bounds([[south, west], [north, east]])

    cmap_obj = matplotlib.colormaps[args.cmap]
    for rgba, (s, w, n, e), label, lo, hi in layers:
        ImageOverlay(
            image=rgba, bounds=[[s, w], [n, e]], origin="upper",
            opacity=args.opacity, name=label,
        ).add_to(m)

        colors = [mcolors.to_hex(cmap_obj(t)) for t in np.linspace(0.0, 1.0, 9)]
        caption = label + ("  (log scale, gradient shown linear)" if args.log else "")
        legend = bcm.LinearColormap(colors=colors, vmin=lo, vmax=hi, caption=caption)
        legend.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    print(f"wrote {out.resolve()}")

    if args.open:
        webbrowser.open(out.resolve().as_uri())

    return 0


if __name__ == "__main__":
    sys.exit(main())
