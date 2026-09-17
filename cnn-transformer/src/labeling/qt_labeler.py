"""
Interactive Qt labeler for processed SWOT NetCDF files.

Three tabs, all sharing the same variable/colormap display controls:

  - "Label processed NCs" / "Edit block labels": drag a 20x20 grid over a
    displayed variable, toggle blocks on/off, save as `block_mask`. Same
    workflow as new_labler.ipynb but as a standalone app, plus drag-select
    (click-drag toggles a whole rectangle of blocks at once) and live
    variable / colormap switching (rolling_mean_highpass, gaussian_filter_
    highpass, ssha_unfiltered, ...).

  - "Label with bboxes": click-drag to draw a freeform pixel-level bounding
    box anywhere on the image (not snapped to the block grid); click inside
    an existing box to delete it. Sources files from the block-labeling
    output folder, not the raw data folder — only files already labeled in
    the "Label processed NCs" tab are available here. Saves a rasterized
    `iw_bbox_mask` to a user-chosen output folder, under the same filename
    as the source file. Revisiting a file re-loads its boxes from that
    folder (via connected-component reconstruction) so editing sessions
    resume where they left off.

Run with:
    conda run -n gis-stuff python src/labeling/qt_labeler.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr
from scipy import ndimage
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


# ============================================================
# Paths
# ============================================================

DATA_DIR = Path(__file__).resolve().parents[2] / "data/OOD/SWOT_L3_Mascarene_Ridge_Dataset/data"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data/OOD/SWOT_L3_Mascarene_Ridge_Dataset/labeled_data"
DEFAULT_BBOX_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data/splits/train_bbox_labels"

# Existing pixel-level YOLO bbox mask from the processed NetCDFs. Also the
# variable name the bbox tab writes its own drawn masks under.
SOURCE_MASK_VAR_NAME = "iw_bbox_mask"

# New block-level mask that the block tabs save.
OUTPUT_MASK_VAR_NAME = "block_mask"

# Variables offered first in the variable dropdown, if present in a file.
# Anything else 2D found in the file is appended after these, alphabetically.
PREFERRED_VARS = [
    "rolling_mean_highpass",
    "gaussian_filter_highpass",
    "fft_highpass",
    "stepped_convolution_highpass",
    "gradient_magnitude",
    "ssha_unfiltered",
    "ssha_filtered",
    "ssha_quality_masked",
    "ssha_unedited",
]

COLORMAPS = [
    "magma",
    "viridis",
    "plasma",
    "inferno",
    "cividis",
    "coolwarm",
    "seismic",
    "RdBu_r",
    "twilight",
    "gray",
    "jet",
]

BLOCK_H = 20
BLOCK_W = 20

# A block is preselected if more than this fraction of its pixels
# have SOURCE_MASK_VALUE in SOURCE_MASK_VAR_NAME.
MAJORITY_THRESHOLD = 0.5
SOURCE_MASK_VALUE = 1

VMIN_PERCENTILE = 2.5
VMAX_PERCENTILE = 97.5

DEFAULT_ZOOM = 6.0
# Zoom slider is stored as integer steps of (real zoom * ZOOM_SLIDER_SCALE),
# so a scale of 10 gives 0.1x increments instead of a full 1.0x per step.
ZOOM_SLIDER_SCALE = 10
ZOOM_MIN = 1.0
ZOOM_MAX = 30.0
GRID_COLOR = Qt.white
SELECT_ALPHA = 165  # 0-255

# Screen-space drag distance (in widget pixels) below which a bbox press+
# release is treated as a click (delete) rather than a drag (draw new box).
BBOX_CLICK_THRESHOLD_PX = 4.0


# ============================================================
# Pure data helpers (no Qt) - ported from new_labler.ipynb
# ============================================================


def list_nc_files(data_dir: Path, output_dir: Path | None = None):
    """Processed .nc files that still need block labeling."""
    files = []
    for p in sorted(data_dir.glob("*.nc")):
        if not p.is_file():
            continue
        if output_dir is not None and (output_dir / f"{p.stem}_labeled.nc").exists():
            continue
        files.append(p)
    return files


def list_labeled_nc_files(output_dir: Path):
    """Block-labeled .nc files in output_dir. Source list for both the
    "Edit block labels" and "Label with bboxes" tabs — bboxes are only
    drawn on files that already have a block_mask."""
    return sorted(p for p in output_dir.glob("*_labeled.nc") if p.is_file())


def count_labeled_files(data_dir: Path, output_dir: Path) -> tuple[int, int]:
    """(labeled_count, total_count) among source .nc files in data_dir."""
    total = 0
    labeled = 0
    for p in sorted(data_dir.glob("*.nc")):
        if not p.is_file():
            continue
        total += 1
        if (output_dir / f"{p.stem}_labeled.nc").exists():
            labeled += 1
    return labeled, total


def select_2d_da(da: xr.DataArray):
    """Select/squeeze a possibly higher-dimensional DataArray down to 2D."""
    selectors = {}
    if "file" in da.dims:
        selectors["file"] = 0
    for dim in da.dims:
        if dim not in selectors and da.sizes[dim] == 1:
            selectors[dim] = 0
    da2d = da.isel(selectors).squeeze(drop=True)
    if da2d.ndim != 2:
        raise ValueError(f"Expected 2D variable, got dims={da2d.dims}, shape={da2d.shape}")
    return da2d, selectors


def find_2d_vars(ds: xr.Dataset, target_shape: tuple[int, int]):
    """All variable names in ds whose squeezed shape matches target_shape."""
    names = []
    for name, var in ds.data_vars.items():
        try:
            da2d, _ = select_2d_da(var)
        except (ValueError, KeyError):
            continue
        if da2d.shape == target_shape:
            names.append(name)

    def sort_key(name):
        if name in PREFERRED_VARS:
            return (0, PREFERRED_VARS.index(name))
        return (1, name)

    return sorted(names, key=sort_key)


def grid_shape(H: int, W: int, bh: int, bw: int):
    return int(np.ceil(H / bh)), int(np.ceil(W / bw))


def block_bounds(r: int, c: int, H: int, W: int, bh: int, bw: int):
    y0, y1 = r * bh, min((r + 1) * bh, H)
    x0, x1 = c * bw, min((c + 1) * bw, W)
    return y0, y1, x0, x1


def blocks_to_pixel_mask(selected_blocks: np.ndarray, H: int, W: int, bh: int, bw: int):
    rows, cols = selected_blocks.shape
    pixel_mask = np.zeros((H, W), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            if selected_blocks[r, c]:
                y0, y1, x0, x1 = block_bounds(r, c, H, W, bh, bw)
                pixel_mask[y0:y1, x0:x1] = 1
    return pixel_mask


def pixel_mask_to_blocks_majority(pixel_mask, bh, bw, mask_value=1, threshold=0.5):
    H, W = pixel_mask.shape
    rows, cols = grid_shape(H, W, bh, bw)
    selected = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            y0, y1, x0, x1 = block_bounds(r, c, H, W, bh, bw)
            patch = pixel_mask[y0:y1, x0:x1]
            if patch.size and np.mean(patch == mask_value) > threshold:
                selected[r, c] = True
    return selected


def mask_to_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Reconstruct axis-aligned bounding boxes from a binary pixel mask.

    Each connected component becomes one (y0, x0, y1, x1) box. This is a
    lossy reconstruction (overlapping/adjacent hand-drawn boxes that were
    rasterized together come back as one merged box), used only to resume
    editing a previously saved bbox mask.
    """
    labeled, _ = ndimage.label(mask > 0)
    boxes = []
    for sl in ndimage.find_objects(labeled):
        if sl is None:
            continue
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        boxes.append((y0, x0, y1, x1))
    return boxes


def boxes_to_pixel_mask(boxes, H: int, W: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    for y0, x0, y1, x1 in boxes:
        y0c, y1c = max(0, y0), min(H, y1)
        x0c, x1c = max(0, x0), min(W, x1)
        if y1c > y0c and x1c > x0c:
            mask[y0c:y1c, x0c:x1c] = 1
    return mask


def robust_limits(arr, lower=2.5, upper=97.5):
    arr = np.asarray(arr)
    if not np.isfinite(arr).any():
        return None, None
    vmin, vmax = np.nanpercentile(arr, [lower, upper])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return None, None
    return float(vmin), float(vmax)


def _embed_2d_mask(base_da, mask_2d, spatial_dims, selectors):
    """Place a 2D mask into a full-size array matching base_da's dims."""
    full_shape = tuple(base_da.sizes[dim] for dim in base_da.dims)
    full = np.zeros(full_shape, dtype=np.uint8)

    indexer = []
    for dim in base_da.dims:
        if dim in spatial_dims:
            indexer.append(slice(None))
        elif dim in selectors:
            indexer.append(selectors[dim])
        else:
            raise ValueError(f"Dimension {dim!r} is neither spatial nor selected")
    full[tuple(indexer)] = mask_2d.astype(np.uint8)
    return full


def _coords_for(base_da):
    return {dim: base_da.coords[dim] for dim in base_da.dims if dim in base_da.coords}


def make_full_sized_mask_da(base_da, mask_2d, spatial_dims, selectors, name,
                             block_h, block_w, source_mask_var_name, source_mask_value,
                             majority_threshold, display_variable):
    full = _embed_2d_mask(base_da, mask_2d, spatial_dims, selectors)
    return xr.DataArray(
        full,
        dims=base_da.dims,
        coords=_coords_for(base_da),
        name=name,
        attrs={
            "description": "Block-level mask. 1=selected block, 0=not selected.",
            "block_h": int(block_h),
            "block_w": int(block_w),
            "source": "qt_labeler.py",
            "display_variable": display_variable,
            "source_mask_variable": source_mask_var_name,
            "source_mask_value": int(source_mask_value),
            "majority_threshold": float(majority_threshold),
            "rule": (
                f"Initial block selected when more than {majority_threshold:.2f} "
                f"of pixels in the {block_h}x{block_w} patch equal {source_mask_value} "
                f"in {source_mask_var_name}."
            ),
        },
    )


def make_bbox_mask_da(base_da, mask_2d, spatial_dims, selectors, name, n_boxes, display_variable):
    full = _embed_2d_mask(base_da, mask_2d, spatial_dims, selectors)
    return xr.DataArray(
        full,
        dims=base_da.dims,
        coords=_coords_for(base_da),
        name=name,
        attrs={
            "description": "Pixel-level bounding-box mask. 1=inside a drawn bbox, 0=outside.",
            "source": "qt_labeler.py bbox tab",
            "display_variable": display_variable,
            "n_boxes": int(n_boxes),
        },
    )


def array_to_qimage(arr: np.ndarray, cmap_name: str, vmin, vmax) -> QImage:
    """Render a float array to an RGBA8888 QImage using a matplotlib colormap."""
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax) if vmin is not None else mcolors.Normalize()
    cmap = mpl.colormaps[cmap_name]
    cmap = cmap.with_extremes(bad=(0.3, 0.3, 0.3, 1.0))
    masked = np.ma.masked_invalid(arr)
    rgba = np.ascontiguousarray(cmap(norm(masked), bytes=True))
    h, w, _ = rgba.shape
    image = QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888)
    return image.copy()


# ============================================================
# Canvas widget: renders the image + either a drag-selectable block grid
# or freeform drag-drawn bounding boxes, depending on mode.
# ============================================================


class ImageCanvas(QWidget):
    selectionChanged = Signal()  # block mode: selection grid changed
    boxesChanged = Signal()  # bbox mode: box list changed

    def __init__(self, block_h: int, block_w: int, parent=None):
        super().__init__(parent)
        self.block_h = block_h
        self.block_w = block_w
        self.mode = "block"  # "block" or "bbox"

        self.base_image: QImage | None = None
        self.H = 0
        self.W = 0
        self.rows = 0
        self.cols = 0
        self.zoom = DEFAULT_ZOOM
        self.show_mask_highlight = True

        # Block-mode state.
        self.selection = np.zeros((0, 0), dtype=bool)
        self._selection_before_drag: np.ndarray | None = None
        self._drag_anchor: tuple[int, int] | None = None
        self._drag_current: tuple[int, int] | None = None
        self._drag_paint_value = True

        # Bbox-mode state.
        self.boxes: list[tuple[int, int, int, int]] = []
        self._bbox_anchor_data: tuple[float, float] | None = None
        self._bbox_current_data: tuple[float, float] | None = None
        self._bbox_press_screen: QPointF | None = None

        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    # -- data -------------------------------------------------
    def set_data(self, arr: np.ndarray, cmap_name: str, vmin, vmax):
        self.H, self.W = arr.shape
        self.rows, self.cols = grid_shape(self.H, self.W, self.block_h, self.block_w)
        self.base_image = array_to_qimage(arr, cmap_name, vmin, vmax)
        self._update_size()
        self.update()

    def set_mode(self, mode: str):
        self.mode = mode
        self._drag_anchor = None
        self._drag_current = None
        self._selection_before_drag = None
        self._bbox_anchor_data = None
        self._bbox_current_data = None
        self._bbox_press_screen = None
        self.update()

    def set_selection(self, selection: np.ndarray):
        self.selection = selection
        self.update()

    def set_boxes(self, boxes: list[tuple[int, int, int, int]]):
        self.boxes = list(boxes)
        self.update()

    def set_zoom(self, zoom: float):
        self.zoom = zoom
        self._update_size()
        self.update()

    def set_show_mask_highlight(self, show: bool):
        self.show_mask_highlight = show
        self.update()

    def _update_size(self):
        self.setFixedSize(max(1, round(self.W * self.zoom)), max(1, round(self.H * self.zoom)))

    # -- painting -----------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        if self.base_image is None:
            return

        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
        painter.drawImage(QRectF(0, 0, self.width(), self.height()), self.base_image)

        if self.mode == "block":
            if self.show_mask_highlight and self.selection.size:
                self._paint_selection(painter)
            self._paint_grid(painter)
        elif self.mode == "bbox" and self.show_mask_highlight:
            self._paint_boxes(painter)

    def _live_selection(self) -> np.ndarray:
        """Block selection including any in-progress drag preview."""
        if self._drag_anchor is None or self._selection_before_drag is None:
            return self.selection

        sel = self._selection_before_drag.copy()
        r0, c0 = self._drag_anchor
        r1, c1 = self._drag_current
        rlo, rhi = sorted((r0, r1))
        clo, chi = sorted((c0, c1))
        sel[rlo:rhi + 1, clo:chi + 1] = self._drag_paint_value
        return sel

    def _paint_selection(self, painter: QPainter):
        sel = self._live_selection()
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(255, 255, 0, SELECT_ALPHA)))

        rows, cols = sel.shape
        for r in range(rows):
            for c in range(cols):
                if not sel[r, c]:
                    continue
                y0, y1, x0, x1 = block_bounds(r, c, self.H, self.W, self.block_h, self.block_w)
                rect = QRectF(x0 * self.zoom, y0 * self.zoom, (x1 - x0) * self.zoom, (y1 - y0) * self.zoom)
                painter.drawRect(rect)

    def _paint_grid(self, painter: QPainter):
        pen = QPen(GRID_COLOR)
        pen.setWidthF(max(0.5, self.zoom * 0.05))
        painter.setPen(pen)

        for y in sorted(set(range(0, self.H, self.block_h)) | {self.H}):
            yy = y * self.zoom
            painter.drawLine(QPointF(0, yy), QPointF(self.W * self.zoom, yy))
        for x in sorted(set(range(0, self.W, self.block_w)) | {self.W}):
            xx = x * self.zoom
            painter.drawLine(QPointF(xx, 0), QPointF(xx, self.H * self.zoom))

    def _paint_boxes(self, painter: QPainter):
        pen = QPen(QColor(255, 90, 0))
        pen.setWidthF(max(1.0, self.zoom * 0.15))
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(255, 120, 0, 60)))

        for y0, x0, y1, x1 in self.boxes:
            rect = QRectF(x0 * self.zoom, y0 * self.zoom, (x1 - x0) * self.zoom, (y1 - y0) * self.zoom)
            painter.drawRect(rect)

        if self._bbox_anchor_data is not None and self._bbox_current_data is not None:
            x0f, y0f = self._bbox_anchor_data
            x1f, y1f = self._bbox_current_data
            x0, x1 = sorted((x0f, x1f))
            y0, y1 = sorted((y0f, y1f))

            preview_pen = QPen(QColor(255, 255, 255))
            preview_pen.setStyle(Qt.DashLine)
            preview_pen.setWidthF(max(1.0, self.zoom * 0.12))
            painter.setPen(preview_pen)
            painter.setBrush(Qt.NoBrush)
            rect = QRectF(x0 * self.zoom, y0 * self.zoom, (x1 - x0) * self.zoom, (y1 - y0) * self.zoom)
            painter.drawRect(rect)

    # -- mouse dispatch -------------------------------------------
    def mousePressEvent(self, event):
        if self.base_image is None:
            return
        if self.mode == "block":
            self._mouse_press_block(event)
        elif self.mode == "bbox":
            self._mouse_press_bbox(event)

    def mouseMoveEvent(self, event):
        if self.mode == "block":
            self._mouse_move_block(event)
        elif self.mode == "bbox":
            self._mouse_move_bbox(event)

    def mouseReleaseEvent(self, event):
        if self.mode == "block":
            self._mouse_release_block(event)
        elif self.mode == "bbox":
            self._mouse_release_bbox(event)

    # -- mouse: block drag-select ----------------------------------
    def _block_at(self, pos: QPointF, clamp: bool):
        x = pos.x() / self.zoom
        y = pos.y() / self.zoom
        c = int(x // self.block_w)
        r = int(y // self.block_h)
        if clamp:
            r = min(max(r, 0), self.rows - 1)
            c = min(max(c, 0), self.cols - 1)
            return r, c
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return r, c
        return None

    def _mouse_press_block(self, event):
        if event.button() != Qt.LeftButton or self.selection.size == 0:
            return
        rc = self._block_at(event.position(), clamp=False)
        if rc is None:
            return
        self._drag_anchor = rc
        self._drag_current = rc
        self._selection_before_drag = self.selection.copy()
        self._drag_paint_value = not self.selection[rc]
        self.update()

    def _mouse_move_block(self, event):
        if self._drag_anchor is None:
            return
        rc = self._block_at(event.position(), clamp=True)
        if rc is None:
            return
        self._drag_current = rc
        self.update()

    def _mouse_release_block(self, event):
        if event.button() != Qt.LeftButton or self._drag_anchor is None:
            return
        self.selection = self._live_selection()
        self._drag_anchor = None
        self._drag_current = None
        self._selection_before_drag = None
        self.update()
        self.selectionChanged.emit()

    # -- mouse: bbox drag-draw / click-delete -----------------------
    def _data_point(self, pos: QPointF, clamp: bool) -> tuple[float, float]:
        x = pos.x() / self.zoom
        y = pos.y() / self.zoom
        if clamp:
            x = min(max(x, 0.0), float(self.W))
            y = min(max(y, 0.0), float(self.H))
        return x, y

    def _mouse_press_bbox(self, event):
        if event.button() != Qt.LeftButton:
            return
        self._bbox_anchor_data = self._data_point(event.position(), clamp=True)
        self._bbox_current_data = self._bbox_anchor_data
        self._bbox_press_screen = event.position()
        self.update()

    def _mouse_move_bbox(self, event):
        if self._bbox_anchor_data is None:
            return
        self._bbox_current_data = self._data_point(event.position(), clamp=True)
        self.update()

    def _mouse_release_bbox(self, event):
        if event.button() != Qt.LeftButton or self._bbox_anchor_data is None:
            return

        release_pos = event.position()
        press_pos = self._bbox_press_screen
        dx = release_pos.x() - press_pos.x()
        dy = release_pos.y() - press_pos.y()
        distance = (dx * dx + dy * dy) ** 0.5

        if distance < BBOX_CLICK_THRESHOLD_PX:
            x, y = self._data_point(release_pos, clamp=True)
            for i in range(len(self.boxes) - 1, -1, -1):
                y0, x0, y1, x1 = self.boxes[i]
                if y0 <= y < y1 and x0 <= x < x1:
                    del self.boxes[i]
                    break
        else:
            x0f, y0f = self._bbox_anchor_data
            x1f, y1f = self._data_point(release_pos, clamp=True)
            x0, x1 = sorted((int(round(x0f)), int(round(x1f))))
            y0, y1 = sorted((int(round(y0f)), int(round(y1f))))
            x0, x1 = max(0, min(x0, self.W)), max(0, min(x1, self.W))
            y0, y1 = max(0, min(y0, self.H)), max(0, min(y1, self.H))
            if x1 > x0 and y1 > y0:
                self.boxes.append((y0, x0, y1, x1))

        self._bbox_anchor_data = None
        self._bbox_current_data = None
        self._bbox_press_screen = None
        self.update()
        self.boxesChanged.emit()


# ============================================================
# Main window
# ============================================================


class LabelerWindow(QMainWindow):
    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        bbox_output_dir: Path = DEFAULT_BBOX_OUTPUT_DIR,
        source_mask_var_name: str = SOURCE_MASK_VAR_NAME,
        output_mask_var_name: str = OUTPUT_MASK_VAR_NAME,
        block_h: int = BLOCK_H,
        block_w: int = BLOCK_W,
        majority_threshold: float = MAJORITY_THRESHOLD,
        source_mask_value: int = SOURCE_MASK_VALUE,
    ):
        super().__init__()
        self.setWindowTitle("SWOT Labeler")

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bbox_output_dir = bbox_output_dir
        self.source_mask_var_name = source_mask_var_name
        self.output_mask_var_name = output_mask_var_name
        self.block_h = block_h
        self.block_w = block_w
        self.majority_threshold = majority_threshold
        self.source_mask_value = source_mask_value

        self.mode = "raw"  # "raw", "labeled", or "bbox"
        self.files: list[Path] = []
        self.idx = 0
        self.desired_var: str | None = None  # user-picked variable, sticky across files
        self.selections: dict[str, np.ndarray] = {}
        self.bbox_selections: dict[str, list[tuple[int, int, int, int]]] = {}

        self.current_ds: xr.Dataset | None = None
        self.current_ref_var: str | None = None
        self.current_shape: tuple[int, int] | None = None
        self.current_arr_cache: dict[str, np.ndarray] = {}

        self._build_ui()
        self._reload_file_list()

    # -- UI construction -----------------------------------------
    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)

        self.tabs = QTabWidget()
        self.tabs.addTab(QWidget(), "Label processed NCs")
        self.tabs.addTab(QWidget(), "Edit block labels")
        self.tabs.addTab(QWidget(), "Label with bboxes")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        root.addWidget(self.tabs)

        controls = QHBoxLayout()

        self.file_combo = QComboBox()
        self.file_combo.setMinimumWidth(320)
        self.file_combo.currentIndexChanged.connect(self._on_file_combo_changed)
        controls.addWidget(QLabel("File:"))
        controls.addWidget(self.file_combo, stretch=2)

        self.var_combo = QComboBox()
        self.var_combo.currentTextChanged.connect(self._on_variable_changed)
        controls.addWidget(QLabel("Variable:"))
        controls.addWidget(self.var_combo)

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(COLORMAPS)
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        controls.addWidget(QLabel("Colormap:"))
        controls.addWidget(self.cmap_combo)

        controls.addStretch(1)
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("font-weight: bold;")
        controls.addWidget(self.progress_label)

        root.addLayout(controls)

        # Bbox-tab-only controls: output folder picker + undo/clear.
        self.bbox_controls = QWidget()
        bbox_row = QHBoxLayout(self.bbox_controls)
        bbox_row.setContentsMargins(0, 0, 0, 0)
        bbox_row.addWidget(QLabel("Bbox output folder:"))
        self.bbox_dir_edit = QLineEdit(str(self.bbox_output_dir))
        self.bbox_dir_edit.editingFinished.connect(self._on_bbox_dir_edited)
        bbox_row.addWidget(self.bbox_dir_edit, stretch=1)
        self.bbox_browse_btn = QPushButton("Browse…")
        self.bbox_browse_btn.clicked.connect(self._on_bbox_browse)
        bbox_row.addWidget(self.bbox_browse_btn)
        self.bbox_undo_btn = QPushButton("Undo last box")
        self.bbox_undo_btn.clicked.connect(self._on_bbox_undo)
        bbox_row.addWidget(self.bbox_undo_btn)
        self.bbox_clear_btn = QPushButton("Clear boxes")
        self.bbox_clear_btn.clicked.connect(self._on_bbox_clear)
        bbox_row.addWidget(self.bbox_clear_btn)
        self.bbox_controls.setVisible(False)
        root.addWidget(self.bbox_controls)

        nav = QHBoxLayout()
        self.prev_btn = QPushButton("← Previous")
        self.next_btn = QPushButton("Next →")
        self.save_btn = QPushButton("Save")
        self.save_btn.setStyleSheet("font-weight: bold;")
        self.mask_toggle_btn = QPushButton("Overlay: On")
        self.mask_toggle_btn.setCheckable(True)
        self.mask_toggle_btn.setChecked(True)

        self.prev_btn.clicked.connect(self._on_prev)
        self.next_btn.clicked.connect(self._on_next)
        self.save_btn.clicked.connect(self._on_save)
        self.mask_toggle_btn.toggled.connect(self._on_mask_toggle)

        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        nav.addWidget(self.save_btn)
        nav.addWidget(self.mask_toggle_btn)

        nav.addWidget(QLabel("Zoom:"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(round(ZOOM_MIN * ZOOM_SLIDER_SCALE))
        self.zoom_slider.setMaximum(round(ZOOM_MAX * ZOOM_SLIDER_SCALE))
        self.zoom_slider.setValue(round(DEFAULT_ZOOM * ZOOM_SLIDER_SCALE))
        self.zoom_slider.setSingleStep(1)
        self.zoom_slider.setPageStep(ZOOM_SLIDER_SCALE)
        self.zoom_slider.setFixedWidth(160)
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        nav.addWidget(self.zoom_slider)
        self.zoom_value_label = QLabel(f"{DEFAULT_ZOOM:.1f}x")
        self.zoom_value_label.setFixedWidth(40)
        nav.addWidget(self.zoom_value_label)

        nav.addWidget(QLabel("Percentile clim:"))
        self.pct_spin = QDoubleSpinBox()
        self.pct_spin.setRange(0.0, 20.0)
        self.pct_spin.setSingleStep(0.5)
        self.pct_spin.setValue(VMIN_PERCENTILE)
        self.pct_spin.valueChanged.connect(lambda _: self._render_current_variable())
        nav.addWidget(self.pct_spin)

        nav.addStretch(1)
        root.addLayout(nav)

        self.canvas = ImageCanvas(self.block_h, self.block_w)
        self.canvas.selectionChanged.connect(self._on_selection_changed)
        self.canvas.boxesChanged.connect(self._on_boxes_changed)

        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(False)
        scroll.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        root.addWidget(scroll, stretch=1)

        self.setCentralWidget(central)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        QShortcut(QKeySequence(Qt.Key_Left), self, activated=self._on_prev)
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=self._on_next)
        QShortcut(QKeySequence.Save, self, activated=self._on_save)

        self.resize(950, 950)

    # -- file list handling ----------------------------------------
    def _reload_file_list(self, preferred_name: str | None = None):
        """Rebuild self.files for the current mode.

        If `preferred_name` is given (the file just saved in raw mode, which
        drops out of the raw list once labeled), land on the next file after
        it alphabetically instead of jumping back to the first file.
        """
        if self.mode == "raw":
            self.files = list_nc_files(self.data_dir, self.output_dir)
        else:  # "labeled" or "bbox" both draw from the block-labeling output
            self.files = list_labeled_nc_files(self.output_dir)

        if preferred_name is not None:
            idx = next((i for i, p in enumerate(self.files) if p.name > preferred_name), None)
            self.idx = idx if idx is not None else max(0, len(self.files) - 1)
        else:
            self.idx = 0

        self.file_combo.blockSignals(True)
        self.file_combo.clear()
        self.file_combo.addItems([p.name for p in self.files])
        self.file_combo.blockSignals(False)

        if self.mode == "raw":
            labeled, total = count_labeled_files(self.data_dir, self.output_dir)
            self.progress_label.setText(f"Labeled: {labeled}/{total}")

        if self.files:
            self._load_and_draw(self.idx)
        else:
            self.canvas.base_image = None
            self.canvas.update()
            self.status_bar.showMessage(f"No files found for mode '{self.mode}'.")

    def _on_tab_changed(self, index: int):
        self.mode = {0: "raw", 1: "labeled", 2: "bbox"}[index]
        self.bbox_controls.setVisible(self.mode == "bbox")
        self.progress_label.setVisible(self.mode == "raw")
        self._reload_file_list()

    def _on_file_combo_changed(self, index: int):
        if index < 0 or index == self.idx or not self.files:
            return
        self.idx = index
        self._load_and_draw(self.idx)

    # -- loading a file --------------------------------------------
    def _load_and_draw(self, idx: int):
        path = self.files[idx]

        with xr.open_dataset(path) as ds:
            ds = ds.load()
        self.current_ds = ds
        self.current_arr_cache = {}

        ref_var, H, W = self._find_reference_var(ds)
        self.current_ref_var = ref_var
        self.current_shape = (H, W)

        available_vars = find_2d_vars(ds, (H, W))
        self.var_combo.blockSignals(True)
        self.var_combo.clear()
        self.var_combo.addItems(available_vars)
        preferred_present = [v for v in PREFERRED_VARS if v in available_vars]
        default_var = preferred_present[0] if preferred_present else (available_vars[0] if available_vars else "")
        if self.desired_var and self.desired_var in available_vars:
            chosen_var = self.desired_var
        else:
            chosen_var = default_var
        if chosen_var:
            self.var_combo.setCurrentText(chosen_var)
        self.var_combo.blockSignals(False)

        fname = path.name
        if self.mode == "bbox":
            if fname not in self.bbox_selections:
                self.bbox_selections[fname] = self._initial_boxes(ds, fname, H, W)
        else:
            if fname not in self.selections:
                self.selections[fname] = self._initial_selection(ds, H, W)

        self.canvas.rows, self.canvas.cols = grid_shape(H, W, self.block_h, self.block_w)
        self.file_combo.blockSignals(True)
        self.file_combo.setCurrentIndex(idx)
        self.file_combo.blockSignals(False)

        self._render_current_variable()

    def _find_reference_var(self, ds: xr.Dataset):
        for name in PREFERRED_VARS + [self.output_mask_var_name, self.source_mask_var_name]:
            if name in ds:
                da2d, _ = select_2d_da(ds[name])
                return name, da2d.shape[0], da2d.shape[1]
        for name, var in ds.data_vars.items():
            try:
                da2d, _ = select_2d_da(var)
            except (ValueError, KeyError):
                continue
            return name, da2d.shape[0], da2d.shape[1]
        raise ValueError("No 2D variable found to determine grid shape")

    def _initial_selection(self, ds: xr.Dataset, H: int, W: int) -> np.ndarray:
        rows, cols = grid_shape(H, W, self.block_h, self.block_w)

        if self.output_mask_var_name in ds:
            mask_da, _ = select_2d_da(ds[self.output_mask_var_name])
            return pixel_mask_to_blocks_majority(
                mask_da.values, self.block_h, self.block_w, mask_value=1, threshold=0.5
            )

        if self.source_mask_var_name in ds:
            mask_da, _ = select_2d_da(ds[self.source_mask_var_name])
            return pixel_mask_to_blocks_majority(
                mask_da.values,
                self.block_h,
                self.block_w,
                mask_value=self.source_mask_value,
                threshold=self.majority_threshold,
            )

        return np.zeros((rows, cols), dtype=bool)

    def _initial_boxes(self, ds: xr.Dataset, fname: str, H: int, W: int) -> list[tuple[int, int, int, int]]:
        """Priority: a previously saved bbox output file, else the source
        file's own iw_bbox_mask, else start blank."""
        out_path = self.bbox_output_dir / fname
        if out_path.exists():
            try:
                with xr.open_dataset(out_path) as ds_existing:
                    if self.source_mask_var_name in ds_existing:
                        mask_da, _ = select_2d_da(ds_existing[self.source_mask_var_name])
                        if mask_da.shape == (H, W):
                            return mask_to_boxes(mask_da.values)
            except (OSError, ValueError, KeyError):
                pass

        if self.source_mask_var_name in ds:
            mask_da, _ = select_2d_da(ds[self.source_mask_var_name])
            if mask_da.shape == (H, W):
                return mask_to_boxes(mask_da.values)

        return []

    # -- rendering ----------------------------------------------
    def _render_current_variable(self):
        if self.current_ds is None or not self.files:
            return
        var_name = self.var_combo.currentText()
        if not var_name or var_name not in self.current_ds:
            return

        if var_name in self.current_arr_cache:
            arr = self.current_arr_cache[var_name]
        else:
            da2d, _ = select_2d_da(self.current_ds[var_name])
            arr = da2d.values.astype(np.float32)
            self.current_arr_cache[var_name] = arr

        lower = self.pct_spin.value()
        upper = 100.0 - lower
        vmin, vmax = robust_limits(arr, lower=lower, upper=upper)

        self.canvas.set_data(arr, self.cmap_combo.currentText(), vmin, vmax)

        fname = self.files[self.idx].name

        if self.mode == "bbox":
            self.canvas.set_mode("bbox")
            boxes = self.bbox_selections[fname]
            self.canvas.set_boxes(boxes)
            count_msg = f"{len(boxes)} boxes"
        else:
            self.canvas.set_mode("block")
            self.canvas.set_selection(self.selections[fname])
            n_selected = int(self.selections[fname].sum())
            count_msg = f"{self.canvas.rows}x{self.canvas.cols} blocks | {n_selected} selected"

        mode_label = {
            "raw": "labeling processed NC",
            "labeled": "editing saved block labels",
            "bbox": "drawing bbox mask",
        }[self.mode]
        clim_msg = "bad color limits" if vmin is None else f"vmin={vmin:.4g}, vmax={vmax:.4g}"
        self.status_bar.showMessage(
            f"[{self.idx + 1}/{len(self.files)}] {fname} | {mode_label} | "
            f"{count_msg} | var={var_name} | {clim_msg}"
        )

    def _on_variable_changed(self, name: str):
        # Fires only on real user interaction (programmatic changes in
        # _load_and_draw are made with the combo's signals blocked), so this
        # is the user's explicit choice and should stick across files.
        if name:
            self.desired_var = name
        self._render_current_variable()

    def _on_cmap_changed(self, _name: str):
        self._render_current_variable()

    def _on_selection_changed(self):
        if not self.files or self.mode == "bbox":
            return
        fname = self.files[self.idx].name
        self.selections[fname] = self.canvas.selection
        self._render_current_variable()

    def _on_boxes_changed(self):
        if not self.files or self.mode != "bbox":
            return
        fname = self.files[self.idx].name
        self.bbox_selections[fname] = self.canvas.boxes
        self._render_current_variable()

    def _on_mask_toggle(self, checked: bool):
        self.mask_toggle_btn.setText("Overlay: On" if checked else "Overlay: Off")
        self.canvas.set_show_mask_highlight(checked)

    def _on_zoom_changed(self, value: int):
        zoom = value / ZOOM_SLIDER_SCALE
        self.canvas.set_zoom(zoom)
        self.zoom_value_label.setText(f"{zoom:.1f}x")

    # -- bbox output folder -----------------------------------------
    def _on_bbox_dir_edited(self):
        text = self.bbox_dir_edit.text().strip()
        if text:
            self.bbox_output_dir = Path(text)
            self.status_bar.showMessage(f"Bbox output folder set to {self.bbox_output_dir}")

    def _on_bbox_browse(self):
        chosen = QFileDialog.getExistingDirectory(self, "Select bbox output folder", str(self.bbox_output_dir))
        if chosen:
            self.bbox_output_dir = Path(chosen)
            self.bbox_dir_edit.setText(str(self.bbox_output_dir))
            self.status_bar.showMessage(f"Bbox output folder set to {self.bbox_output_dir}")

    def _on_bbox_undo(self):
        if not self.files or self.mode != "bbox":
            return
        fname = self.files[self.idx].name
        boxes = self.bbox_selections.get(fname, [])
        if boxes:
            boxes.pop()
            self.canvas.set_boxes(boxes)
            self._render_current_variable()

    def _on_bbox_clear(self):
        if not self.files or self.mode != "bbox":
            return
        fname = self.files[self.idx].name
        self.bbox_selections[fname] = []
        self.canvas.set_boxes([])
        self._render_current_variable()

    # -- navigation -----------------------------------------------
    def _on_prev(self):
        if self.files and self.idx > 0:
            self.idx -= 1
            self._load_and_draw(self.idx)

    def _on_next(self):
        if self.files and self.idx < len(self.files) - 1:
            self.idx += 1
            self._load_and_draw(self.idx)

    # -- save -------------------------------------------------------
    def _on_save(self):
        if not self.files or self.current_ds is None:
            return
        if self.mode == "bbox":
            self._save_bbox()
        else:
            self._save_block()

    def _save_block(self):
        in_path = self.files[self.idx]
        fname = in_path.name
        selected = self.selections[fname]
        H, W = self.current_shape

        pixel_mask = blocks_to_pixel_mask(selected, H, W, self.block_h, self.block_w)

        ds_out = self.current_ds
        base_da = ds_out[self.current_ref_var]
        da2d, selectors = select_2d_da(base_da)

        if pixel_mask.shape != da2d.shape:
            self.status_bar.showMessage(
                f"ERROR: mask shape {pixel_mask.shape} does not match data shape {da2d.shape}"
            )
            return

        mask_da = make_full_sized_mask_da(
            base_da=base_da,
            mask_2d=pixel_mask,
            spatial_dims=da2d.dims,
            selectors=selectors,
            name=self.output_mask_var_name,
            block_h=self.block_h,
            block_w=self.block_w,
            source_mask_var_name=self.source_mask_var_name,
            source_mask_value=self.source_mask_value,
            majority_threshold=self.majority_threshold,
            display_variable=self.var_combo.currentText(),
        )
        ds_out[self.output_mask_var_name] = mask_da
        ds_out.attrs["block_labeler_note"] = (
            f"Added or updated {self.output_mask_var_name}. "
            f"Initial labels came from {self.source_mask_var_name} using "
            f"a {self.block_h}x{self.block_w} majority rule."
        )

        out_path = in_path if self.mode == "labeled" else self.output_dir / f"{in_path.stem}_labeled.nc"
        self._atomic_to_netcdf(ds_out, out_path)

        self.status_bar.showMessage(f"Saved to {out_path}")

        if self.mode == "raw":
            self._reload_file_list(preferred_name=fname)

    def _save_bbox(self):
        in_path = self.files[self.idx]
        fname = in_path.name
        boxes = self.bbox_selections.get(fname, [])
        H, W = self.current_shape

        pixel_mask = boxes_to_pixel_mask(boxes, H, W)

        ds_out = self.current_ds
        base_da = ds_out[self.current_ref_var]
        da2d, selectors = select_2d_da(base_da)

        if pixel_mask.shape != da2d.shape:
            self.status_bar.showMessage(
                f"ERROR: mask shape {pixel_mask.shape} does not match data shape {da2d.shape}"
            )
            return

        mask_da = make_bbox_mask_da(
            base_da=base_da,
            mask_2d=pixel_mask,
            spatial_dims=da2d.dims,
            selectors=selectors,
            name=self.source_mask_var_name,
            n_boxes=len(boxes),
            display_variable=self.var_combo.currentText(),
        )
        ds_out[self.source_mask_var_name] = mask_da
        ds_out.attrs["bbox_labeler_note"] = (
            f"Added or updated {self.source_mask_var_name} via the bbox tab "
            f"({len(boxes)} boxes)."
        )

        self.bbox_output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.bbox_output_dir / fname
        self._atomic_to_netcdf(ds_out, out_path)

        self.status_bar.showMessage(f"Saved {len(boxes)} boxes to {out_path}")

    @staticmethod
    def _atomic_to_netcdf(ds_out: xr.Dataset, out_path: Path):
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        ds_out.to_netcdf(tmp_path)
        if out_path.exists():
            out_path.unlink()
        tmp_path.replace(out_path)


def main():
    app = QApplication(sys.argv)
    window = LabelerWindow(data_dir=DATA_DIR, output_dir=OUTPUT_DIR, bbox_output_dir=DEFAULT_BBOX_OUTPUT_DIR)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
