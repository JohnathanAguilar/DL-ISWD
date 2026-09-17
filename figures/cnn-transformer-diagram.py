# %% [markdown]
# # Architecture Figure — CNN–Transformer Detector + Dense Refinement Head
# 
# Publication figure for the model in
# [`src/models/token_model.py`](../../src/models/token_model.py), drawn in the same
# visual language as the existing pipeline figure (numbered stage panels, per-stage
# glyphs, math tensor annotations, symbol table) and extended with the new
# [`DenseRefinementHead`](../../src/models/dense_head.py).
# 
# **Reading the figure:** stages run left → right, and each panel lists that stage's
# layers top → bottom, ending in the tensor it emits. Stages 1–5 are the existing
# detector (bracketed as frozen during dense-head Stage 1 training). Stage 6 is new:
# it consumes the transformer tokens $Z_L$ *and* the full-resolution CNN features $F$
# via the skip connection, turning one confidence per block into one per pixel.
# 
# Every layer, channel count, and hyperparameter is read from `ModelConfig`, so the
# figure tracks the code rather than drifting from it. Running the notebook writes
# `cnn_transformer_dense_head_architecture.png/.pdf/.svg` into this folder; the PDF
# and SVG are vector, so the figure scales to any column width without softening.

# %% [markdown]
# ## Configuration
# 
# The dense-head variant; every number in the figure comes from here.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon


def find_model_src(start: Path) -> Path:
    """Locate the directory that holds ``src/models/model_config.py`` (the
    cnn-transformer project root), searching this script's own location and the
    current working directory and their parents."""
    candidates = [start, *start.parents, Path.cwd(), *Path.cwd().parents]
    for parent in candidates:
        if (parent / "src" / "models" / "model_config.py").exists():
            return parent / "src"
        if (parent / "cnn-transformer" / "src" / "models" / "model_config.py").exists():
            return parent / "cnn-transformer" / "src"
    raise RuntimeError(
        "Could not locate src/models/model_config.py near "
        f"{start} or {Path.cwd()}"
    )


sys.path.insert(0, str(find_model_src(Path(__file__).resolve().parent)))

from models.model_config import ModelConfig  # noqa: E402

# ModelConfig's dataclass defaults are stale relative to the trained detector
# this figure documents (they predate the final sweep). The real values come
# from runs/cnntrans_gfh_123/args.json (single-field Gaussian-high-pass variant,
# the one the dense head in Stage 6 is trained on top of): block_size 20,
# depth 64, num_heads 8, mlp_ratio 3.0, drop 0.15, SRA on with ratio 2,
# 2-D sine-cosine positional encoding, 1 input channel. Override them all here
# so every number in the figure tracks the code rather than the stale defaults.
cfg = ModelConfig(
    dense_head=True,
    in_channels=1,
    block_size=20,
    depth=64,
    num_heads=8,
    mlp_ratio=3.0,
    drop=0.15,
    attn_drop=0.1,
    sra=True,
    sra_ratio=2,
    positional="sincos2d",
)
D = cfg.embed_dim
BS = cfg.block_size
MLP_HIDDEN = int(D * cfg.mlp_ratio)
DENSE_IN = D * 2 if cfg.dense_use_cnn_features else D

cfg

# %% [markdown]
# ## Palette and geometry
# 
# One hue per stage, each with a border line, a pale panel fill, and a `soft` step
# for the sub-boxes inside it. Working in units of **points** (`S = 1/72`) keeps the
# geometry and the type scale in the same currency: a 13-unit sub-box is 13 pt tall,
# so it comfortably holds 6.6 pt text.

# %%
# ------------------------------------------------------------------ palette
INK = "#111111"
MUTED = "#555555"
WHITE = "#ffffff"

STAGE = {
    "cnn":   dict(line="#1f5fbf", fill="#eef4fd", soft="#cfe0f8"),
    "tok":   dict(line="#1a7a41", fill="#eef9f1", soft="#c9e9d5"),
    "pos":   dict(line="#5b2d91", fill="#f4effb", soft="#dccdf0"),
    "trf":   dict(line="#c98a00", fill="#fff9ec", soft="#fbe3a2"),
    "head":  dict(line="#c62828", fill="#fdf0f0", soft="#f7cfcf"),
    "dense": dict(line="#c2185b", fill="#fdeef4", soft="#f7cadd"),
}
ACCENT_MLP = "#f5c396"

# ------------------------------------------------------------------ geometry
S = 1 / 72          # inches per unit -> one data unit is one point
# Panels size to their content rather than sharing one width (the reference
# figure does the same: its stage 6 is visibly wider than stage 3). Stage 4
# (Transformer Encoder) needs the extra room for its nested encoder-block
# diagram, whose residual-connector lines get cramped against a narrow border.
PANEL_W = [112.0, 112.0, 112.0, 150.0, 128.0, 128.0]
GAP, X0 = 34.0, 104.0
PH = 272.0
PY0, PY1 = 0.0, PH

FS_HEAD = 9.2       # numbered stage header
FS_TITLE = 8.0      # bold title inside a panel
FS_BODY = 7.0       # body text
FS_SUB = 6.6        # text inside sub-boxes
FS_MATH = 7.6
FS_NOTE = 6.4
FS_TABLE = 6.8


def px(i):
    """Left edge of panel i."""
    return X0 + sum(PANEL_W[:i]) + i * GAP


def pcx(i):
    return px(i) + PANEL_W[i] / 2

# %% [markdown]
# ## Drawing primitives
# 
# `panel()` draws a numbered stage panel, `subbox()` the small labelled boxes inside
# it, and `vflow()` stacks those into an arrow-joined column. They draw onto the
# module-level `ax` created in the drawing step, which Python resolves at call time.

# %%
# ------------------------------------------------------------------ helpers
# These draw onto the module-level `ax`, which is created in the drawing step
# below; Python resolves it at call time, so definition order is free.
def text(x, y, s, fs=FS_BODY, color=INK, weight="normal", ha="center", va="center",
         style="normal", family=None, z=6):
    return ax.text(x, y, s, fontsize=fs, color=color, fontweight=weight, ha=ha, va=va,
                    style=style, fontfamily=family, zorder=z, linespacing=1.45)


def rbox(x0, y0, w, h, ec, fc, lw=1.0, ls="-", r=4.0, z=2):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                                 boxstyle=f"round,pad=0,rounding_size={r}",
                                 linewidth=lw, edgecolor=ec, facecolor=fc,
                                 linestyle=ls, mutation_aspect=1, zorder=z))


def arrow(p0, p1, color=INK, lw=1.1, ls="-", ms=8, z=5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                                  linewidth=lw, color=color, linestyle=ls,
                                  shrinkA=0, shrinkB=0, zorder=z))


def route(pts, color=INK, lw=1.1, ls="-", ms=8):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.add_line(Line2D([a[0], b[0]], [a[1], b[1]], color=color, linewidth=lw,
                            linestyle=ls, zorder=5, solid_capstyle="round"))
    arrow(pts[-2], pts[-1], color=color, lw=lw, ls=ls, ms=ms)


def oplus(cx, cy, r=4.6, color=INK, z=6):
    ax.add_patch(Circle((cx, cy), r, edgecolor=color, facecolor=WHITE, linewidth=0.9, zorder=z))
    ax.add_line(Line2D([cx - r * 0.55, cx + r * 0.55], [cy, cy], color=color, linewidth=0.9, zorder=z + 1))
    ax.add_line(Line2D([cx, cx], [cy - r * 0.55, cy + r * 0.55], color=color, linewidth=0.9, zorder=z + 1))


def panel(i, header, key):
    """Outer stage panel + its numbered header; returns the stage colors."""
    c = STAGE[key]
    rbox(px(i), PY0, PANEL_W[i], PH, c["line"], c["fill"], lw=1.3, r=6.0, z=2)
    text(pcx(i), PY1 + 10, header, fs=FS_HEAD, color=c["line"], weight="bold", va="bottom")
    return c


def subbox(cx, cy, w, h, label, ec, fc, fs=FS_SUB, weight="bold"):
    rbox(cx - w / 2, cy - h / 2, w, h, ec, fc, lw=0.8, r=2.5, z=6)
    text(cx, cy, label, fs=fs, weight=weight, z=7)


def vflow(cx, y_top, items, w, h=13.0, gap=8.0, ec=None, fc=None, heights=None):
    """Vertical stack of sub-boxes joined by arrows. Returns the bottom y."""
    y = y_top
    for k, label in enumerate(items):
        bh = heights[k] if heights else h
        if k:
            arrow((cx, y), (cx, y - gap), lw=0.9, ms=6)
            y -= gap
        subbox(cx, y - bh / 2, w, bh, label, ec, fc)
        y -= bh
    return y

# %% [markdown]
# ## Stage glyphs
# 
# The small pictograms that carry each stage's intuition: stacked feature-map sheets
# for the CNN, a 3-D block cube for tokenization, a learned position heat grid, and
# the block-grid → pixel-grid pair that is the dense head's whole point.

# %%
# ------------------------------------------------------------------ glyphs
def feature_stack(cx, cy, n=4, w=32, h=32, dy=4.0, color="cnn"):
    """Layered feature-map channels: square cards stacked straight up, built
    from the same rounded-rect primitive as every other glyph in the figure
    (a diagonal fan read as a skewed flag/ramp; a plain vertical stack reads
    as channels layered on top of one another instead)."""
    c = STAGE[color]
    for k in range(n - 1, 0, -1):
        rbox(cx - w / 2, cy - h / 2 + k * dy, w, h, c["line"], c["soft"], lw=0.7, r=2.0, z=4 + (n - k))
    rbox(cx - w / 2, cy - h / 2, w, h, c["line"], WHITE, lw=1.0, r=2.0, z=4 + n)
    for gx in (cx - w * 0.17, cx + w * 0.17):
        ax.add_line(Line2D([gx, gx], [cy - h / 2 + 3, cy + h / 2 - 3], color=c["line"], linewidth=0.4, alpha=0.6, zorder=5 + n))
    for gy in (cy - h * 0.17, cy + h * 0.17):
        ax.add_line(Line2D([cx - w / 2 + 3, cx + w / 2 - 3], [gy, gy], color=c["line"], linewidth=0.4, alpha=0.6, zorder=5 + n))


def block_cube(cx, cy, w=52, h=34, dx=10, dy=8, nx=4, ny=3, color="tok"):
    """3-D cube of pooled blocks."""
    c = STAGE[color]
    x0, y0 = cx - w / 2, cy - h / 2
    ax.add_patch(Polygon([(x0, y0 + h), (x0 + w, y0 + h), (x0 + w + dx, y0 + h + dy), (x0 + dx, y0 + h + dy)],
                          closed=True, facecolor=c["soft"], edgecolor=c["line"], linewidth=0.7, zorder=5))
    ax.add_patch(Polygon([(x0 + w, y0), (x0 + w + dx, y0 + dy), (x0 + w + dx, y0 + h + dy), (x0 + w, y0 + h)],
                          closed=True, facecolor=c["soft"], edgecolor=c["line"], linewidth=0.7, zorder=5))
    rbox(x0, y0, w, h, c["line"], c["fill"], lw=0.8, r=0.5, z=6)
    for k in range(1, nx):
        ax.add_line(Line2D([x0 + k * w / nx] * 2, [y0, y0 + h], color=c["line"], linewidth=0.5, zorder=7))
    for k in range(1, ny):
        ax.add_line(Line2D([x0, x0 + w], [y0 + k * h / ny] * 2, color=c["line"], linewidth=0.5, zorder=7))


def pe_heatmap(cx, cy, w=44, h=32, n=6, m=6):
    """Small learned-positional-encoding heat grid."""
    c = STAGE["pos"]
    x0, y0 = cx - w / 2, cy - h / 2
    vals = np.linspace(1, 0, n)[:, None] * np.linspace(1, 0.15, m)[None, :]
    for r in range(n):
        for q in range(m):
            v = vals[r, q]
            shade = tuple(1 - v * (1 - ch) for ch in (0.36, 0.18, 0.57))
            ax.add_patch(FancyBboxPatch((x0 + q * w / m, y0 + (n - 1 - r) * h / n),
                                         w / m, h / n, boxstyle="round,pad=0,rounding_size=0.2",
                                         linewidth=0.3, edgecolor="#ffffff", facecolor=shade, zorder=5))
    rbox(x0, y0, w, h, c["line"], "none", lw=0.8, r=0.5, z=6)


def resolution_glyph(cx, cy):
    """Coarse block grid -> per-pixel grid, the dense head's job."""
    c = STAGE["dense"]
    w = 30
    x0 = cx - 40
    y0 = cy - w / 2
    rbox(x0, y0, w, w, c["line"], c["fill"], lw=0.8, r=0.5, z=6)
    for k in range(1, 3):
        ax.add_line(Line2D([x0 + k * w / 3] * 2, [y0, y0 + w], color=c["line"], linewidth=0.5, zorder=7))
        ax.add_line(Line2D([x0, x0 + w], [y0 + k * w / 3] * 2, color=c["line"], linewidth=0.5, zorder=7))
    arrow((x0 + w + 3, cy), (x0 + w + 16, cy), lw=0.9, ms=6)
    x1 = x0 + w + 19
    rbox(x1, y0, w, w, c["line"], c["fill"], lw=0.8, r=0.5, z=6)
    for k in range(1, 10):
        ax.add_line(Line2D([x1 + k * w / 10] * 2, [y0, y0 + w], color=c["line"], linewidth=0.25, alpha=0.8, zorder=7))
        ax.add_line(Line2D([x1, x1 + w], [y0 + k * w / 10] * 2, color=c["line"], linewidth=0.25, alpha=0.8, zorder=7))
    text(x0 + w / 2, y0 - 6, "block", fs=FS_NOTE, color=MUTED, va="top")
    text(x1 + w / 2, y0 - 6, "pixel", fs=FS_NOTE, color=MUTED, va="top")

# %% [markdown]
# ## Draw the figure
# 
# Panels 1–6 left to right, then the wiring: the straight trunk arrows, the $Z_L$
# junction that feeds both heads, the dashed $F$ skip lane along the bottom, the
# frozen-detector bracket, and the symbol table.

# %%
# ------------------------------------------------------------------ draw
fig, ax = plt.subplots()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)  # axes must fill the figure
ax.axis("off")
ax.set_facecolor(WHITE)
fig.patch.set_facecolor(WHITE)

# ---- input
rng = np.random.default_rng(7)
yy, xx = np.mgrid[0:64, 0:64]
field = (np.sin((xx * 0.55 + 6 * np.sin(yy * 0.06)) * 0.9) * np.exp(-((yy - 26) ** 2) / 420)
         + 0.45 * np.sin((xx * 0.3 - yy * 0.22)) + 0.25 * rng.standard_normal((64, 64)))
for k, (ox, oy) in enumerate([(-9, -8), (-4.5, -4), (0, 0)]):
    ax.imshow(field, cmap="Greys_r", extent=(20 + ox, 76 + ox, PH / 2 - 28 + oy, PH / 2 + 28 + oy),
               aspect="auto", zorder=3 + k, interpolation="bilinear")
    ax.add_patch(FancyBboxPatch((20 + ox, PH / 2 - 28 + oy), 56, 56,
                                 boxstyle="round,pad=0,rounding_size=0.5", linewidth=0.7,
                                 edgecolor="#333333", facecolor="none", zorder=3 + k))
text(48, PH / 2 + 50, "Input Image", fs=FS_TITLE, weight="bold")
text(48, PH / 2 + 40, r"$X \in \mathbb{R}^{B \times C_{in} \times H \times W}$", fs=FS_MATH)
text(48, PH / 2 - 46, rf"$(C_{{in}} = {cfg.in_channels})$", fs=FS_MATH)
text(48, PH / 2 - 58, "SWOT SSHA + high-pass", fs=FS_NOTE, color=MUTED)
arrow((80, PH / 2), (px(0) - 4, PH / 2), lw=1.2, ms=9)

# Panels 1, 2, 3, 5, 6 have far less content than panel 4 (Transformer), whose
# nested encoder-block diagram sets PH for the whole row. Anchoring their
# output block to a fixed distance from the panel bottom (the original
# approach) left a large dead gap between the spec text and the output text.
# Instead each panel's output block now sits GAP below wherever that panel's
# own content ends, and the whole (now-compact) content is recentered in PH by
# picking a top offset that balances the top and bottom margins.
GAP_MID = 26

# ------------------------------------------------------------------ 1. CNN stem
c = panel(0, "1. Feature Extraction\n(CNN Stem)", "cnn")
y = PY1 - 41
text(pcx(0), y, "CNN Stem", fs=FS_TITLE, weight="bold"); y -= 16
for s in [rf"$3 \times$ [Conv2D", "(3×3, pad 1, no bias)", "+ BatchNorm2D + GELU]",
          f"width = {D} channels"]:
    text(pcx(0), y, s, fs=FS_BODY); y -= 11
feature_stack(pcx(0), y - 30, n=4, color="cnn")
note_y = y - 60
text(pcx(0), note_y, "stride 1 — full H×W kept", fs=FS_NOTE, color=MUTED)
bot = note_y - GAP_MID
text(pcx(0), bot, "Output Features", fs=FS_TITLE, weight="bold")
text(pcx(0), bot - 14, r"$F \in \mathbb{R}^{B \times D \times H \times W}$", fs=FS_MATH)
text(pcx(0), bot - 28, r"$(D = \mathrm{embed\_dim})$", fs=FS_MATH)

# ------------------------------------------------------------------ 2. tokenizer
c = panel(1, "2. Block\nTokenization", "tok")
y = PY1 - 48
text(pcx(1), y, "Block Tokenizer", fs=FS_TITLE, weight="bold"); y -= 16
for s in ["AvgPool2D", "(kernel = block_size,", "stride = block_size)",
          f"block_size = {BS}"]:
    text(pcx(1), y, s, fs=FS_BODY); y -= 11
block_cube(pcx(1), y - 30)
note_y = y - 56
text(pcx(1), note_y, "one token per block", fs=FS_NOTE, color=MUTED)
bot = note_y - GAP_MID
text(pcx(1), bot, "Output Tokens", fs=FS_TITLE, weight="bold")
text(pcx(1), bot - 14, r"$T \in \mathbb{R}^{B \times N \times D}$", fs=FS_MATH)
text(pcx(1), bot - 28, r"$(N = B_h \times B_w)$", fs=FS_MATH)

# ------------------------------------------------------------------ 3. posenc
c = panel(2, "3. Positional\nEncoding", "pos")
y = PY1 - 31
text(pcx(2), y, "2D Positional\nEncoding", fs=FS_TITLE, weight="bold"); y -= 26
for s in ["fixed 2D", "sine–cosine (not learned)"]:
    text(pcx(2), y, s, fs=FS_BODY); y -= 11
pe_heatmap(pcx(2) + 4, y - 26)
ax.annotate("", xy=(pcx(2) - 24, y - 10), xytext=(pcx(2) - 24, y - 42),
             arrowprops=dict(arrowstyle="<->", lw=0.7, color=INK), zorder=6)
text(pcx(2) - 29, y - 26, r"$B_h$", fs=FS_NOTE, ha="right")
ax.annotate("", xy=(pcx(2) - 18, y - 48), xytext=(pcx(2) + 26, y - 48),
             arrowprops=dict(arrowstyle="<->", lw=0.7, color=INK), zorder=6)
text(pcx(2) + 4, y - 55, r"$B_w$", fs=FS_NOTE, va="top")
note_y = y - 76
text(pcx(2), note_y, "one encoding per\nblock position", fs=FS_NOTE, color=MUTED, va="top")
bot = note_y - GAP_MID - 24
arrow((pcx(2), note_y - 20), (pcx(2), bot + 17), lw=0.9, ms=6)        # P flows down toward the add
text(pcx(2), bot + 11, "added to tokens", fs=FS_NOTE, color=MUTED, ha="center")
oplus(pcx(2), bot, r=5.2)
arrow((pcx(2), bot - 5.2 - 1), (pcx(2), bot - 22 + 4), lw=0.9, ms=6)  # sum flows out as Z_0
text(pcx(2), bot - 22, r"$T + P$", fs=FS_MATH)
text(pcx(2), bot - 36, "Output Tokens", fs=FS_TITLE, weight="bold")
text(pcx(2), bot - 50, r"$Z_0 \in \mathbb{R}^{B \times N \times D}$", fs=FS_MATH)

# ------------------------------------------------------------------ 4. transformer
c = panel(3, "4. Transformer\nEncoder", "trf")
y = PY1 - 18
text(pcx(3), y, "Transformer Encoder", fs=FS_TITLE, weight="bold"); y -= 12
text(pcx(3), y, rf"$L$ layers   $(L = {cfg.depth})$", fs=FS_BODY); y -= 10
text(pcx(3), y, rf"{cfg.num_heads} heads · pre-LN", fs=FS_NOTE, color=MUTED); y -= 12
blk_y1, blk_y0 = y, PY0 + 56
rbox(px(3) + 8, blk_y0, PANEL_W[3] - 16, blk_y1 - blk_y0, c["line"], WHITE, lw=0.8, r=3.5, z=4)
text(pcx(3), blk_y1 - 9, "Encoder Block (× L)", fs=FS_NOTE, weight="bold", z=6)
yy_ = blk_y1 - 20
TRF_W = PANEL_W[3] - 50   # sub-box width, incl. the MLP box
TRF_CH = PANEL_W[3] / 2 - 17  # residual channel: clear of both the sub-box edge and the block border
subbox(pcx(3), yy_ - 6.5, TRF_W, 13, "LayerNorm", c["line"], STAGE["trf"]["soft"])
arrow((pcx(3), yy_ - 13), (pcx(3), yy_ - 20), lw=0.9, ms=6)
subbox(pcx(3), yy_ - 32, TRF_W, 20,
       f"Multi-Head Self-Attention\n(spatial reduction, ratio {cfg.sra_ratio})",
       c["line"], ACCENT_MLP, fs=FS_SUB)
arrow((pcx(3), yy_ - 42), (pcx(3), yy_ - 51), lw=0.9, ms=6)
oplus(pcx(3), yy_ - 56, r=4.4)
route([(pcx(3) + TRF_W / 2, yy_ - 6.5), (pcx(3) + TRF_CH, yy_ - 6.5),
       (pcx(3) + TRF_CH, yy_ - 56), (pcx(3) + 5, yy_ - 56)], lw=0.7, ms=5)
arrow((pcx(3), yy_ - 61), (pcx(3), yy_ - 68), lw=0.9, ms=6)
subbox(pcx(3), yy_ - 80, TRF_W, 13, "LayerNorm", c["line"], STAGE["trf"]["soft"])
arrow((pcx(3), yy_ - 87), (pcx(3), yy_ - 94), lw=0.9, ms=6)
subbox(pcx(3), yy_ - 105, TRF_W, 16, f"MLP  {D} → {MLP_HIDDEN} → {D}", c["line"], ACCENT_MLP)
arrow((pcx(3), yy_ - 113), (pcx(3), yy_ - 121), lw=0.9, ms=6)
oplus(pcx(3), yy_ - 126, r=4.4)
route([(pcx(3) + TRF_W / 2, yy_ - 80), (pcx(3) + TRF_CH, yy_ - 80),
       (pcx(3) + TRF_CH, yy_ - 126), (pcx(3) + 5, yy_ - 126)], lw=0.7, ms=5)
text(pcx(3), PY0 + 40, "Output Tokens", fs=FS_TITLE, weight="bold")
text(pcx(3), PY0 + 24, r"$Z_L \in \mathbb{R}^{B \times N \times D}$", fs=FS_MATH)

# ------------------------------------------------------------------ 5. block head
c = panel(4, "5. Block Prediction\nHead", "head")
y = PY1 - 35
text(pcx(4), y, "Block Probability Head", fs=FS_TITLE, weight="bold"); y -= 12
text(pcx(4), y, "(Pre-LayerNorm MLP)", fs=FS_BODY); y -= 16
y = vflow(pcx(4), y, ["LayerNorm", rf"Linear:  $D \rightarrow D/2$", "GELU",
                       rf"Linear:  $D/2 \rightarrow 1$"],
          w=PANEL_W[4] - 30, h=13, gap=8, ec=c["line"], fc=c["soft"])
logits_y = y - 14
text(pcx(4), logits_y, r"Logits  $\ell \in \mathbb{R}^{B \times N}$", fs=FS_BODY)
sig_y = logits_y - 13
text(pcx(4), sig_y, r"$\sigma(\cdot)$  (in model forward)", fs=FS_NOTE, color=MUTED)
bot = sig_y - GAP_MID + 4
text(pcx(4), bot, "Block Probabilities", fs=FS_TITLE, weight="bold")
text(pcx(4), bot - 16, r"$p \in [0,1]^{B \times N}$", fs=FS_MATH)
text(pcx(4), bot - 31, f"one value per {BS}×{BS} block", fs=FS_NOTE, color=MUTED)

# ------------------------------------------------------------------ 6. dense head
c = panel(5, "6. Dense Refinement\nHead", "dense")
y = PY1 - 18
text(pcx(5), y, "Dense Refinement Head", fs=FS_TITLE, weight="bold"); y -= 13
text(pcx(5), y, rf"broadcast $\times {BS}$ ({cfg.dense_token_upsample})", fs=FS_BODY); y -= 16
resolution_glyph(pcx(5) + 5, y - 14)
y -= 52
text(pcx(5), y, r"$\oplus$  concat  $F$  (skip)", fs=FS_BODY); y -= 12
y = vflow(pcx(5), y, [rf"Conv 3×3:  ${DENSE_IN} \rightarrow {cfg.dense_hidden}$" + "\n+ GELU",
                       rf"Conv 3×3:  ${cfg.dense_hidden} \rightarrow {cfg.dense_hidden}$" + "\n+ GELU",
                       rf"Conv 1×1:  ${cfg.dense_hidden} \rightarrow 1$",
                       r"Sigmoid  $\sigma(\cdot)$"],
          w=PANEL_W[5] - 30, h=13, gap=6, ec=c["line"], fc=c["soft"], heights=[19, 19, 13, 13])
bot = y - GAP_MID
text(pcx(5), bot, "Dense Confidence", fs=FS_TITLE, weight="bold")
text(pcx(5), bot - 16, r"$\hat{Y} \in [0,1]^{B \times 1 \times H_t \times W_t}$", fs=FS_MATH)
text(pcx(5), bot - 31, "one value per pixel", fs=FS_NOTE, color=MUTED)

# ------------------------------------------------------------------ wiring
for i in range(4):
    arrow((px(i) + PANEL_W[i] + 2, PH / 2), (px(i + 1) - 2, PH / 2), lw=1.2, ms=9)

# Z_L feeds both heads: junction between the transformer and the block head
jx = (px(3) + PANEL_W[3] + px(4)) / 2
ax.add_patch(Circle((jx, PH / 2), 2.0, facecolor=INK, edgecolor="none", zorder=7))
LANE_Z, LANE_F = -26.0, -50.0
route([(jx, PH / 2), (jx, LANE_Z), (pcx(5) - 26, LANE_Z), (pcx(5) - 26, PY0 - 2)],
      color=STAGE["trf"]["line"], lw=1.1, ms=8)
text((jx + pcx(5) - 26) / 2, LANE_Z + 4, r"$Z_L$  (contextual block tokens)",
     fs=FS_NOTE, color=STAGE["trf"]["line"], va="bottom", style="italic")

route([(pcx(0), PY0 - 2), (pcx(0), LANE_F), (pcx(5) + 26, LANE_F), (pcx(5) + 26, PY0 - 2)],
      color=STAGE["cnn"]["line"], lw=1.1, ls=(0, (4, 2.5)), ms=8)
text((pcx(0) + pcx(5) + 26) / 2, LANE_F + 4,
     r"skip: $F$ — full-resolution CNN features (within-block spatial detail the tokenizer averaged away)",
     fs=FS_NOTE, color=STAGE["cnn"]["line"], va="bottom", style="italic")

# frozen-backbone bracket over stages 1-5, clear of the stage headers
fr_y = PY1 + 52
ax.add_line(Line2D([px(0), px(4) + PANEL_W[4]], [fr_y] * 2, color=MUTED, linewidth=0.8,
                    linestyle=(0, (5, 3)), zorder=3))
for xx_ in (px(0), px(4) + PANEL_W[4]):
    ax.add_line(Line2D([xx_] * 2, [fr_y, fr_y - 5], color=MUTED, linewidth=0.8, zorder=3))
text((px(0) + px(4) + PANEL_W[4]) / 2, fr_y + 3,
     "existing detector — weights frozen while the dense head is trained (Stage 1)",
     fs=FS_NOTE, color=MUTED, va="bottom", style="italic")

# ------------------------------------------------------------------ symbol table
TB_Y1, TB_Y0 = LANE_F - 14, LANE_F - 76
rbox(px(0), TB_Y0, px(5) + PANEL_W[5] - px(0), TB_Y1 - TB_Y0, "#333333", WHITE,
     lw=0.9, ls=(0, (4, 3)), r=3.0, z=2)
rows = [
    [(r"$B$", "batch size"), (r"$D$", "embedding dimension (embed_dim)"), (r"$L$", "number of transformer layers")],
    [(r"$C_{in}$", "number of input channels"), (r"$B_h, B_w$", "blocks in height and width"), (r"$F$", "full-resolution CNN features")],
    [(r"$H, W$", "input height and width"), (r"$N$", r"number of tokens $(N = B_h \times B_w)$"), (r"$Z_L$", "transformer output tokens")],
    [(r"$H_t, W_t$", "tokenized region (multiple of block_size)"), (r"$\sigma(\cdot)$", "sigmoid function"), (r"$\oplus$", "channel concatenation / residual add")],
]
col_w = (px(5) + PANEL_W[5] - px(0)) / 3
SYM_PAD = 34  # clears the widest right-aligned symbol ("H_t, W_t") off the box edge
for r_i, row in enumerate(rows):
    ty = TB_Y1 - 13 - r_i * 13
    for c_i, (sym, desc) in enumerate(row):
        sx = px(0) + SYM_PAD + c_i * col_w
        text(sx, ty, sym, fs=FS_TABLE, ha="right")
        text(sx + 4, ty, ":", fs=FS_TABLE, ha="left", color=MUTED)
        text(sx + 11, ty, desc, fs=FS_TABLE, ha="left", color=INK)

# ------------------------------------------------------------------ finish
x0, x1 = -4, px(5) + PANEL_W[5] + 6
y0, y1 = TB_Y0 - 8, fr_y + 16
ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)
ax.set_aspect("equal")
fig.set_size_inches((x1 - x0) * S, (y1 - y0) * S)

plt.show()

# %% [markdown]
# ## Save
# 
# PNG for quick viewing, PDF and SVG as vector for the paper.

# %%
OUT_DIR = Path.cwd()
for ext in ("png", "pdf", "svg"):
    out_path = OUT_DIR / f"cnn_transformer_dense_head_architecture.{ext}"
    fig.savefig(out_path, dpi=400, bbox_inches="tight", facecolor=WHITE)
    print(f"saved {out_path}")

print(f"\nfigure {(x1 - x0) * S:.2f} x {(y1 - y0) * S:.2f} in  (aspect {(x1 - x0) / (y1 - y0):.2f}:1)")


