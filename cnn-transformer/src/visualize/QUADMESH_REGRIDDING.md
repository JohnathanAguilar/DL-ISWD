# Quadmesh regridding: SWOT ISW swaths → regular 2 km grid

How `build_isw_store.py` turns the per-pass CNN-Transformer detection NetCDFs into the regular
lat/lon Zarr store the `swot-viewer-secure` viewer renders, and exactly what that transform does
(and does not do) to the data.

---

## 1. Why regrid at all

The detection files are on **SWOT swath geometry**:

- shape `num_lines (~9800) × num_pixels (60)` — along-track × cross-track
- `latitude` and `longitude` are **2-D arrays** of the same shape: a curvilinear grid that
  follows the orbit. Row/column indices have no fixed relationship to degrees; each of the
  ~588 000 pixels has its own `(lat, lon)`.

The viewer's renderer (`@carbonplan/zarr-layer`) can only draw a **rectilinear** raster: a 2-D
array indexed by a **1-D** ascending `latitude` axis and a **1-D** `longitude` axis, evenly
spaced. It has no code path for 2-D curvilinear coordinates. So each swath has to be resampled
onto a common regular grid before it can be shown.

The regular grid used here is **global, 0.02° (~2 km at the equator)**: 9000 rows × 18000 columns,
cell centres at `lat = -90 + (i + 0.5)·0.02`, `lon = -180 + (j + 0.5)·0.02`. This ~matches SWOT's
native 2 km pixel size, so the resample is close to 1:1.

---

## 2. The two grids

```
   SOURCE MESH (one SWOT pass)              TARGET GRID (fixed, global)
   curvilinear, follows the orbit           rectilinear, evenly spaced

     \   \   \   \                          +--+--+--+--+--+--+
      \   \   \   \       each pixel        |  |  |  |  |  |  |
       +---+---+---+      is a quad,        +--+--+--+--+--+--+
       |   |   |   |      cornered by       |  |  |  |  |  |  |
       +---+---+---+  →   its real lon/lat  +--+--+--+--+--+--+
        \   \   \   \                       |  |  |  |  |  |  |
         \   \   \   \                      +--+--+--+--+--+--+
```

- **Source mesh** — the swath itself. Every pixel `(l, p)` becomes a quadrilateral whose corners
  are derived from the original 2-D `latitude`/`longitude`. The mesh is warped and unevenly
  spaced (denser toward the poles, wider cross-track cells at high latitude).
- **Target grid** — the 0.02° global lattice. Always the same cells, independent of which pass.

`datashader.Canvas.quadmesh(da, x="longitude", y="latitude", agg=…)` does the rasterisation, with
`da` a `DataArray` carrying the 2-D coordinate arrays.

---

## 3. The fill algorithm (target-cell-driven)

The key point: it is **not** "each source pixel is dropped into a target cell." It is the
reverse — **each target cell asks which source quad it sits inside**:

```
for every target cell C (centre at some lon/lat):
    Q = the source quad(s) whose polygon contains C's centre
    if no Q            →  C stays NaN            (outside the swath → transparent)
    if one Q           →  C = value(Q)
    if several Q        →  C = agg(values)        (see §4)
```

Consequences:

- **Gap-free.** Where the target grid is finer than the local pixel spacing (or the swath runs
  diagonally), several target cells fall inside one source quad and all get that quad's value.
  Plain forward scatter ("nearest source pixel → one cell") leaves holes between pixel centres
  and produces a stippled swath; quadmesh fills the quad interiors, so the swath is solid.
- **Piecewise-constant, not smoothed.** A target cell takes a source value verbatim — no bilinear
  interpolation, no blending across the swath. A wide edge quad simply paints several target
  cells the same value.
- **No coverage outside the swath.** A target cell only gets a value if its centre is
  geographically inside a real pixel footprint.

Measured coverage for one pass onto the 0.02° grid: ~1.4 % of the pass bounding box filled by
quadmesh vs ~0.8 % by nearest-point scatter — quadmesh fills ~1.75× more cells, all of them real.

---

## 4. Aggregation per variable (`agg`)

When more than one source quad of the **same pass** covers a target cell (mostly at swath edges
where pixels widen), the values are combined by a datashader reduction chosen per variable:

| store array | source variable | `agg` | why |
|---|---|---|---|
| `ssha` | `internal_wave_bbox_confidence` (0..1) | `max` | keep the strongest detection; never dilute a thin wave signature |
| `adt`  | `quality_flag` (uint8 → f4) | `max` | keep the "worst" (highest) flag |
| `sig0` | `gaussian_filter_highpass` (m) | `mean` | a signed field; max would bias it, mean is the faithful combine |

This only ever applies **within a single pass**. It is not a cross-pass composite (see §5).

---

## 5. One pass at a time — `time` slices, no cross-pass merge

The regrid runs **per SWOT pass**. Passes are sorted by `(cycle, pass)` and each is assigned an
index `i`; `slices.json` (and the root group's `attrs.slices`) has one
`{i, time, date, cycle, pass}` record per pass.

Each pass is quadmesh-rasterised once per variable and written to its own time slice:

```
0/ssha  shape [N, 9000, 18000]      0/ssha[i] = the regridded field for pass i
0/sig0  shape [N, 9000, 18000]
0/adt   shape [N, 9000, 18000]
```

So if **two passes cross the same location**, their values land in **different slices**
(`ssha[i, y, x]` and `ssha[j, y, x]`) — nothing is overwritten or averaged between passes. The
viewer shows one slice at a time (single-pass mode); the cycle/pass selector picks `i`.

There is **no** "combine every pass over this cell" array in this store (no composites were
requested). If one is wanted later, it would be an extra `ssha_cyc[C, 9000, 18000]` array =
`max` over each cycle's slices, which the viewer's "full cycle" mode reads.

---

## 6. Antimeridian handling

A SWOT half-orbit spans ~162° of longitude and roughly half of the passes cross ±180°. In a
single fixed longitude frame the swath would appear to wrap the whole globe and the quads would
smear across the seam.

For each pass, `choose_lon()` computes the longitude two ways — `[-180, 180)` and `[0, 360)` —
and keeps whichever gives the **smaller span** (i.e. the branch cut the half-orbit does not
straddle). The quadmesh is then rasterised in that contiguous frame onto a sub-window aligned to
the global grid.

When the sub-window is written back into the `[-180, 180)`-indexed store array, `lon_runs()`
converts each output column to its global longitude-cell index and splits the block into (at
most) two contiguous runs where the indices cross the +180 seam. Each run is a plain slice
assignment; the wrap case does two.

`y` (latitude) never wraps → always a single run.

---

## 7. Pyramid levels

The viewer is a Zarr **multiscale pyramid**: `@carbonplan/zarr-layer` reads
`attrs.multiscales = [{ datasets: [{path:"0"},{path:"1"},…] }]` and picks a level by map zoom so
it never fetches the full-resolution array at world view.

Level 0 is the quadmesh output at 0.02°. Coarser levels are **2×2 block reductions** of the level
above (`block_reduce()`):

| level | cell size | grid | reducer |
|---|---|---|---|
| 0 | 0.02° | 9000 × 18000 | quadmesh (§3–4) |
| 1 | 0.04° | 4500 × 9000 | `ssha`/`adt` block-max, `sig0` block-mean |
| 2 | 0.08° | 2250 × 4500 | " |
| 3 | 0.16° | 1125 × 2250 | " |

(`--levels` is capped by the level-0 grid dividing evenly: `9000 = 2³·1125`, so 4 levels at
0.02°.) Every level group also carries its own 1-D `latitude`/`longitude`; level 0 additionally
carries a `time` array (epoch seconds).

---

## 8. What is preserved, what changes

**Preserved**

- **Data values.** A target cell takes a source value verbatim (or `max`/`mean` of overlapping
  same-pass quads). No interpolation, no smoothing, no unit change (beyond `quality_flag`
  uint8 → float32).
- **Geolocation, undistorted.** A value appears at a target `(lat, lon)` only because a real
  swath pixel physically covered that point. The swath is not stretched, rotated, reprojected,
  or slid to fit the axes.
- **Per-pass identity.** Each pass stays a separate `time` slice with its `cycle`/`pass`/`time`
  metadata.

**Changed (this is why it is still a *resample*)**

- **Sample locations.** Output lives on the regular 0.02° lattice, not the swath's own points. A
  pixel measured at `(12.3412, −45.6789)` contributes to the cell centred at `(12.34, −45.68)`.
- **The original 2-D coordinate arrays are not kept.** The store has only the regular 1-D
  `latitude`/`longitude` axes. (The first build produced a raw-swath store that keeps the exact
  2-D coords — but the viewer can't render it. A `native/` group could be added alongside the
  grid if the originals need to be retained.)
- **Overlapping same-pass pixels are merged** by `agg` (edge cells only).
- **Coarser pyramid levels are aggregated** (block max / mean).

**Accuracy bound**

Worst-case position error ≈ half a target cell (~1 km at the equator, less at higher latitude in
the N–S direction) plus the source pixel footprint (~2 km). No systematic shift or rotation.

---

## 9. Store layout produced

```
isw-store/
  zarr.json          root group; attrs.multiscales (OME-NGFF datasets form) + attrs.slices
  slices.json        [{i, time, date, cycle, pass}, ...]   one per pass, sorted by (cycle, pass)
  0/
    time       [N]              i8   epoch seconds
    latitude   [9000]           f4   ascending, single chunk
    longitude  [18000]          f4   single chunk
    ssha       [N, 9000, 18000] f4   fill NaN, chunks (1,1024,1024)  <- internal_wave_bbox_confidence
    sig0       [N, 9000, 18000] f4                                   <- gaussian_filter_highpass
    adt        [N, 9000, 18000] f4                                   <- quality_flag
  1/ 2/ 3/           coarser levels (0.04 / 0.08 / 0.16°), each with its own latitude/longitude
```

`write_empty_chunks=False` (Zarr v3 default): all-NaN chunks are never written, so on-disk size is
dominated by the thin diagonal each swath occupies — ~5 GB for cycles 1 & 2 (980 passes).

Array names `ssha` / `sig0` / `adt` are the keys the viewer's hardcoded `VARS` looks for; the
real mapping is in `0/zarr.json` → `attributes.variable_map`.

---

## 10. Running it

```
cd swot_isw_viewer
# smoke run (coarse, a few passes)
/Users/aguilarj/miniforge3/envs/gis-stuff/bin/python build_isw_store.py --limit 6 --resolution 0.08 --levels 2
# full cycles 1 & 2
/Users/aguilarj/miniforge3/envs/gis-stuff/bin/python build_isw_store.py --resolution 0.02 --levels 4
```

| flag | default | meaning |
|---|---|---|
| `--results-root` | `../cnn-transformer/results` | folder of `cycle_*` dirs |
| `--cycle-glob` | `cycle_0[12]` | which cycles |
| `--out` | `./isw-store` | Zarr store (overwritten) |
| `--vars` | `ssha,sig0,adt` | which arrays to build |
| `--resolution` | `0.02` | level-0 cell size, degrees |
| `--levels` | `4` | pyramid levels (grid must divide by `2**(levels-1)`) |
| `--chunk` | `1024` | spatial chunk edge |
| `--limit` | `0` | if >0, only the first N passes |

`datashader`, `xarray`, `zarr` 3.x and `netCDF4` are in the `gis-stuff` / `second` / `third`
conda envs. Cost ≈ 2 s/pass, peak RAM ~3–4 GB.

Serve with `serve_isw.py` (Range + CORS) and open
`http://localhost:8000/swot-viewer-secure/index.html?store=http://localhost:8000/isw-store`.

---

## 11. Quadmesh vs the alternatives

| method | swath continuity | values | geolocation | notes |
|---|---|---|---|---|
| **nearest-point scatter** (`floor((lat+90)/res)`, `np.fmax.at`) | stippled — gaps between pixel centres | exact | ±½ cell | fast, but visually broken at 2 km |
| **quadmesh (this)** | solid — quad interiors filled | exact / `agg` at edges | ±½ cell, no warp | `datashader`, ~2 s/pass |
| **conservative regrid** (`xesmf`) | solid | area-weighted average (values change) | exact | heavier; changes values by design |
| **keep curvilinear** (raw swath store) | n/a | exact | exact (2-D coords kept) | viewer cannot render it |
