#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pixel_editor.py -- a compact desktop pixel-art editor.

Dependencies:  pip install pillow      (Tkinter ships with CPython)

Run the editor:     python pixel_editor.py
Run accuracy tests: python pixel_editor.py --selftest

Design notes
------------
* The artwork is a single Pillow RGBA image ("logical pixels").  Nothing about
  zooming, panning, the grid, previews or measurement overlays ever touches
  that image; they are display-only concepts.
* All screen<->image coordinate math lives in `Viewport` so the conversion is
  defined exactly once (see `screen_to_pixel` / `pixel_to_screen`).
* Rendering crops only the visible part of the artwork, composites it over a
  checkerboard, then scales it with NEAREST resampling, so a 512x512 canvas at
  3200% costs no more than a viewport-sized bitmap.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import shutil
import sys
import tempfile
import zlib

import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox

from PIL import Image, ImageDraw

try:  # Pillow >= 9.1
    NEAREST = Image.Resampling.NEAREST
except AttributeError:  # pragma: no cover - very old Pillow
    NEAREST = Image.NEAREST

try:
    from PIL import ImageTk
except ImportError:  # pragma: no cover
    ImageTk = None


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

APP_NAME = "Pixel Editor"
PROJECT_EXT = ".pxproj"

MIN_DIM = 1
MAX_DIM = 1024          # maximum width/height for a new/resized canvas
MAX_OPEN_DIM = 4096     # maximum width/height accepted when importing a PNG
MAX_TOTAL_PIXELS = 1024 * 1024

# Integer zoom factors only: one logical pixel is always an exact square of
# `zoom` x `zoom` screen pixels, which keeps the grid perfectly aligned.
ZOOM_LEVELS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
DEFAULT_ZOOM = 8
GRID_MIN_ZOOM = 4       # below this zoom the pixel grid is hidden
BRUSH_MIN, BRUSH_MAX = 1, 16
EXPORT_SCALES = [1, 2, 4, 8, 10, 16]
MAX_RECENT_COLORS = 14
PRESET_SIZES = [(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512)]

TRANSPARENT = (0, 0, 0, 0)

# Dark editor palette
C_BG = "#1e1f22"
C_PANEL = "#26282c"
C_PANEL_DARK = "#1a1b1e"
C_BORDER = "#3a3d42"
C_TEXT = "#d7dae0"
C_TEXT_DIM = "#9aa0a8"
C_ACCENT = "#4c8dff"
C_BTN = "#32353a"
C_BTN_ACTIVE = "#41454b"
C_CANVAS_BG = "#141517"
C_OUTLINE = "#5a5f66"
C_MEASURE = "#ffd24a"
CHECKER_A = (74, 74, 78, 255)
CHECKER_B = (94, 94, 99, 255)
GRID_MINOR = (128, 128, 132, 90)
GRID_MAJOR = (176, 176, 184, 150)

FONT_UI = ("Segoe UI", 9)
FONT_UI_BOLD = ("Segoe UI", 9, "bold")
FONT_MONO = ("Consolas", 9)
FONT_SECTION = ("Segoe UI", 8, "bold")

# (key, label, shortcut) for every tool
TOOL_SPECS = [
    ("pencil", "Pencil", "P"),
    ("eraser", "Eraser", "E"),
    ("line", "Line", "L"),
    ("rect", "Rect", "R"),
    ("ellipse", "Circle", "C"),
    ("fill", "Fill", "F"),
    ("picker", "Pick", "I"),
    ("measure", "Measure", "M"),
    ("select", "Select", "S"),
    ("pan", "Pan", "H"),
]


# --------------------------------------------------------------------------- #
# Small formatting / colour helpers
# --------------------------------------------------------------------------- #

def commas(n: int) -> str:
    """1234567 -> '1,234,567'."""
    return f"{int(n):,}"


def clamp(value, low, high):
    return low if value < low else (high if value > high else value)


def format_half(value: float) -> str:
    """15.0 -> '15', 15.5 -> '15.5' (used for ellipse radii)."""
    return str(int(round(value))) if abs(value - round(value)) < 1e-9 else f"{value:g}"


def parse_hex_color(text: str, default_alpha: int = 255):
    """Parse '#rgb', '#rrggbb' or '#rrggbbaa' (with or without '#').

    Raises ValueError on anything else.
    """
    s = text.strip().lstrip("#").strip()
    if not all(c in "0123456789abcdefABCDEF" for c in s):
        raise ValueError("hex colour may only contain 0-9 and a-f")
    if len(s) == 3:
        r, g, b = (int(c * 2, 16) for c in s)
        return (r, g, b, default_alpha)
    if len(s) == 6:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), default_alpha)
    if len(s) == 8:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), int(s[6:8], 16))
    raise ValueError("hex colour must have 3, 6 or 8 digits")


def rgb_hex(color) -> str:
    """RGBA tuple -> '#RRGGBB' (alpha is reported separately in the UI)."""
    return "#%02X%02X%02X" % (color[0], color[1], color[2])


def rgba_hex(color) -> str:
    return "#%02X%02X%02X%02X" % tuple(color[:4])


def parse_dimension(text: str, name: str, maximum: int = MAX_DIM) -> int:
    """Validate a user supplied pixel dimension, raising ValueError with a
    human readable message."""
    s = (text or "").strip()
    if not s:
        raise ValueError(f"{name} is required.")
    try:
        value = int(s, 10)
    except ValueError:
        raise ValueError(f"{name} must be a whole number of pixels.") from None
    if value < MIN_DIM:
        raise ValueError(f"{name} must be at least {MIN_DIM} pixel.")
    if value > maximum:
        raise ValueError(f"{name} must be {maximum} pixels or less.")
    return value


# --------------------------------------------------------------------------- #
# Rasterisation helpers -- pure integer maths, never anti-aliased.
# These functions are GUI-free so they can be unit tested directly.
# --------------------------------------------------------------------------- #

def bresenham_line(x0: int, y0: int, x1: int, y1: int):
    """Classic integer Bresenham line, inclusive of both end points."""
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x1 >= x0 else -1
    sy = 1 if y1 >= y0 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return points


def normalize_box(x0: int, y0: int, x1: int, y1: int):
    """Return (left, top, right, bottom) with inclusive integer bounds."""
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def rect_fill_pixels(x0, y0, x1, y1):
    a, b, c, d = normalize_box(x0, y0, x1, y1)
    return [(x, y) for y in range(b, d + 1) for x in range(a, c + 1)]


def rect_outline_pixels(x0, y0, x1, y1):
    a, b, c, d = normalize_box(x0, y0, x1, y1)
    if c - a < 2 or d - b < 2:
        return rect_fill_pixels(a, b, c, d)
    pts = []
    for x in range(a, c + 1):
        pts.append((x, b))
        pts.append((x, d))
    for y in range(b + 1, d):
        pts.append((a, y))
        pts.append((c, y))
    return pts


def ellipse_fill_pixels(x0, y0, x1, y1):
    """Filled ellipse inside the inclusive bounding box.

    A pixel belongs to the ellipse when its *centre* (x + 0.5, y + 0.5) lies
    inside the continuous ellipse, which keeps the result symmetric and free
    of anti-aliasing.
    """
    a, b, c, d = normalize_box(x0, y0, x1, y1)
    w = c - a + 1
    h = d - b + 1
    if w <= 2 or h <= 2:
        # A 1- or 2-pixel wide "ellipse" degenerates into its bounding box.
        return rect_fill_pixels(a, b, c, d)
    rx = w / 2.0
    ry = h / 2.0
    cx = a + rx
    cy = b + ry
    pts = []
    for y in range(b, d + 1):
        ny = (y + 0.5 - cy) / ry
        rem = 1.0 - ny * ny
        if rem <= 0.0:
            continue
        span = math.sqrt(rem) * rx
        left = int(math.ceil(cx - span - 0.5))
        right = int(math.floor(cx + span - 0.5))
        left = max(left, a)
        right = min(right, c)
        for x in range(left, right + 1):
            pts.append((x, y))
    return pts


def ellipse_outline_pixels(x0, y0, x1, y1):
    """One-pixel outline of `ellipse_fill_pixels`.

    The extreme pixels of every row *and* every column are kept, which yields
    a closed ring even where the ellipse is nearly flat.
    """
    filled = ellipse_fill_pixels(x0, y0, x1, y1)
    if not filled:
        return []
    row_min, row_max, col_min, col_max = {}, {}, {}, {}
    for (x, y) in filled:
        if y not in row_min or x < row_min[y]:
            row_min[y] = x
        if y not in row_max or x > row_max[y]:
            row_max[y] = x
        if x not in col_min or y < col_min[x]:
            col_min[x] = y
        if x not in col_max or y > col_max[x]:
            col_max[x] = y
    out = set()
    for y, x in row_min.items():
        out.add((x, y))
    for y, x in row_max.items():
        out.add((x, y))
    for x, y in col_min.items():
        out.add((x, y))
    for x, y in col_max.items():
        out.add((x, y))
    return sorted(out, key=lambda p: (p[1], p[0]))


def brush_offsets(size: int):
    """Offsets of a square brush; size 1 is exactly one logical pixel."""
    size = max(1, int(size))
    if size == 1:
        return [(0, 0)]
    start = -(size // 2)
    return [(dx, dy) for dy in range(start, start + size)
            for dx in range(start, start + size)]


def expand_brush(points, size: int):
    """Expand a list of path points into the pixels a square brush covers."""
    size = max(1, int(size))
    if size == 1:
        return list(points)
    offsets = brush_offsets(size)
    out = set()
    for (x, y) in points:
        for (dx, dy) in offsets:
            out.add((x + dx, y + dy))
    return list(out)


def snap_line_endpoint(x0, y0, x1, y1):
    """Snap a line to the nearest multiple of 45 degrees (Shift constraint)."""
    dx, dy = x1 - x0, y1 - y0
    if dx == 0 and dy == 0:
        return x1, y1
    directions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    index = int(round(math.atan2(dy, dx) / (math.pi / 4))) % 8
    ux, uy = directions[index]
    length = max(abs(dx), abs(dy))
    return x0 + ux * length, y0 + uy * length


def square_endpoint(x0, y0, x1, y1):
    """Shift constraint for rectangles/ellipses: equal side lengths."""
    dx, dy = x1 - x0, y1 - y0
    side = max(abs(dx), abs(dy))
    sx = 1 if dx >= 0 else -1
    sy = 1 if dy >= 0 else -1
    return x0 + sx * side, y0 + sy * side


def line_metrics(x0, y0, x1, y1):
    """Measurements for a line between two logical pixel centres.

    `angle` is measured in screen space (positive y points down), matching the
    numbers shown in the UI, and `pixels` is the number of pixels a Bresenham
    rasterisation of the line would touch.
    """
    dx = x1 - x0
    dy = y1 - y0
    return {
        "x0": x0, "y0": y0, "x1": x1, "y1": y1,
        "dx": dx, "dy": dy,
        "abs_dx": abs(dx), "abs_dy": abs(dy),
        "distance": math.hypot(dx, dy),
        "angle": math.degrees(math.atan2(dy, dx)),
        "pixels": max(abs(dx), abs(dy)) + 1,
    }


def flood_fill_image(image: Image.Image, x: int, y: int, new_color):
    """Iterative 4-connected flood fill (no recursion, so huge regions are safe).

    Returns the number of pixels changed.
    """
    w, h = image.size
    if not (0 <= x < w and 0 <= y < h):
        return 0
    px = image.load()
    target = px[x, y]
    if tuple(target) == tuple(new_color):
        return 0
    stack = [(x, y)]
    seen = {(x, y)}
    changed = 0
    while stack:
        cx, cy = stack.pop()
        px[cx, cy] = new_color
        changed += 1
        for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in seen:
                if px[nx, ny] == target:
                    seen.add((nx, ny))
                    stack.append((nx, ny))
    return changed


def is_pixel_perfect_corner(a, b, c) -> bool:
    """True when `b` is a redundant corner pixel between `a` and `c`.

    Used by the pixel-perfect pencil: with a diagonal step the middle pixel of
    an L-shaped triple is dropped so freehand diagonals stay one pixel thin.
    """
    if abs(a[0] - c[0]) != 1 or abs(a[1] - c[1]) != 1:
        return False
    adj_a = abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
    adj_c = abs(c[0] - b[0]) + abs(c[1] - b[1]) == 1
    return adj_a and adj_c


# --------------------------------------------------------------------------- #
# Document: the artwork plus its undo history
# --------------------------------------------------------------------------- #

class PixelDocument:
    """Holds the one true RGBA artwork image and the undo/redo history.

    Snapshots are stored zlib-compressed so a long history stays cheap even
    for a 1024x1024 canvas.
    """

    MAX_HISTORY = 60

    def __init__(self, width: int, height: int, image: Image.Image | None = None):
        if image is None:
            self.image = Image.new("RGBA", (int(width), int(height)), TRANSPARENT)
        elif image.mode == "RGBA":
            self.image = image.copy()
        else:
            self.image = image.convert("RGBA")
        self.px = self.image.load()
        self._undo = []
        self._redo = []
        self.project_path = None
        self.export_path = None
        self.modified = False

    # -- basic properties ------------------------------------------------- #

    @property
    def width(self) -> int:
        return self.image.size[0]

    @property
    def height(self) -> int:
        return self.image.size[1]

    @property
    def size(self):
        return self.image.size

    @property
    def total_pixels(self) -> int:
        return self.width * self.height

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    # -- pixel access ----------------------------------------------------- #

    def get_pixel(self, x: int, y: int):
        if not self.in_bounds(x, y):
            return None
        return tuple(self.px[x, y])

    def set_pixel(self, x: int, y: int, color) -> None:
        if self.in_bounds(x, y):
            self.px[x, y] = color

    def write_pixels(self, coords, color) -> int:
        """Write `color` into every in-bounds coordinate. Returns count."""
        w, h = self.image.size
        px = self.px
        count = 0
        for (x, y) in coords:
            if 0 <= x < w and 0 <= y < h:
                px[x, y] = color
                count += 1
        return count

    def replace_image(self, image: Image.Image) -> None:
        """Swap in a new artwork image (resize/scale/paste operations)."""
        self.image = image if image.mode == "RGBA" else image.convert("RGBA")
        self.px = self.image.load()
        self.modified = True

    # -- history ---------------------------------------------------------- #

    def _snapshot(self, label: str):
        raw = self.image.tobytes()
        return (label, self.width, self.height, zlib.compress(raw, 1))

    def _restore(self, snap) -> None:
        _label, w, h, blob = snap
        raw = zlib.decompress(blob)
        self.image = Image.frombytes("RGBA", (w, h), raw)
        self.px = self.image.load()

    def begin_operation(self, label: str) -> None:
        """Record the pre-edit state.  Call once per user operation (one whole
        pencil stroke, one shape, one fill), never per mouse motion."""
        self._undo.append(self._snapshot(label))
        if len(self._undo) > self.MAX_HISTORY:
            del self._undo[0:len(self._undo) - self.MAX_HISTORY]
        self._redo.clear()
        self.modified = True

    def abort_operation(self) -> bool:
        """Roll back (and forget) the operation started by `begin_operation`."""
        if not self._undo:
            return False
        self._restore(self._undo.pop())
        return True

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self):
        if not self._undo:
            return None
        snap = self._undo.pop()
        self._redo.append(self._snapshot(snap[0]))
        if len(self._redo) > self.MAX_HISTORY:
            del self._redo[0:len(self._redo) - self.MAX_HISTORY]
        self._restore(snap)
        self.modified = True
        return snap[0]

    def redo(self):
        if not self._redo:
            return None
        snap = self._redo.pop()
        self._undo.append(self._snapshot(snap[0]))
        self._restore(snap)
        self.modified = True
        return snap[0]

    def undo_label(self):
        return self._undo[-1][0] if self._undo else None

    def redo_label(self):
        return self._redo[-1][0] if self._redo else None

    # -- export ----------------------------------------------------------- #

    def build_export_image(self, scale: int = 1, background=None,
                           grid_color=None) -> Image.Image:
        """Produce the image that will be written to disk.

        `scale` 1 always returns exactly the logical canvas dimensions.
        Upscaling is nearest-neighbour so pixels stay perfectly sharp.
        """
        scale = max(1, int(scale))
        out = self.image.copy()
        if background is not None:
            bg = Image.new("RGBA", out.size, tuple(background))
            out = Image.alpha_composite(bg, out)
        if scale != 1:
            out = out.resize((out.size[0] * scale, out.size[1] * scale), NEAREST)
            if grid_color is not None and scale >= 2:
                # Optional, explicitly requested grid burned into the export.
                draw = ImageDraw.Draw(out)
                w, h = out.size
                for gx in range(0, w, scale):
                    draw.line([(gx, 0), (gx, h - 1)], fill=tuple(grid_color))
                for gy in range(0, h, scale):
                    draw.line([(0, gy), (w - 1, gy)], fill=tuple(grid_color))
        return out


# --------------------------------------------------------------------------- #
# Viewport: the single place where screen <-> pixel conversion happens
# --------------------------------------------------------------------------- #

class Viewport:
    """Maps logical image pixels to widget coordinates.

    `pan_x` / `pan_y` are the widget coordinates of the top-left corner of
    logical pixel (0, 0).  They are kept as floats for smooth zoom maths but
    always rounded to whole screen pixels before use so that the bitmap, the
    grid and the cursor read-out agree exactly.
    """

    def __init__(self, zoom: int = DEFAULT_ZOOM):
        self.zoom = int(zoom)
        self.pan_x = 0.0
        self.pan_y = 0.0

    # -- derived values --------------------------------------------------- #

    @property
    def origin(self):
        return int(round(self.pan_x)), int(round(self.pan_y))

    @property
    def percent(self) -> int:
        return int(round(self.zoom * 100))

    # -- conversions ------------------------------------------------------ #

    def screen_to_pixel(self, sx, sy):
        """Widget coordinate -> logical pixel coordinate (may be off-canvas)."""
        ox, oy = self.origin
        return (math.floor((sx - ox) / self.zoom),
                math.floor((sy - oy) / self.zoom))

    def screen_to_pixel_float(self, sx, sy):
        """Continuous image-space position (used when zooming to the cursor)."""
        ox, oy = self.origin
        return ((sx - ox) / self.zoom, (sy - oy) / self.zoom)

    def pixel_to_screen(self, px, py):
        """Logical pixel -> widget coordinate of that pixel's top-left corner."""
        ox, oy = self.origin
        return (ox + px * self.zoom, oy + py * self.zoom)

    def pixel_rect_on_screen(self, px, py):
        """Widget rectangle (x0, y0, x1, y1) covered by one logical pixel."""
        x0, y0 = self.pixel_to_screen(px, py)
        return (x0, y0, x0 + self.zoom, y0 + self.zoom)

    def visible_pixel_bounds(self, view_w, view_h, img_w, img_h):
        """Inclusive-exclusive pixel window (x0, y0, x1, y1) visible on screen."""
        ox, oy = self.origin
        x0 = clamp(math.floor((0 - ox) / self.zoom), 0, img_w)
        y0 = clamp(math.floor((0 - oy) / self.zoom), 0, img_h)
        x1 = clamp(math.ceil((view_w - ox) / self.zoom), 0, img_w)
        y1 = clamp(math.ceil((view_h - oy) / self.zoom), 0, img_h)
        return x0, y0, x1, y1

    # -- zoom / pan ------------------------------------------------------- #

    def set_zoom_at(self, new_zoom: int, anchor_sx, anchor_sy) -> None:
        """Change zoom while keeping the image point under (anchor) fixed."""
        new_zoom = int(new_zoom)
        if new_zoom == self.zoom:
            return
        ix, iy = self.screen_to_pixel_float(anchor_sx, anchor_sy)
        self.zoom = new_zoom
        self.pan_x = anchor_sx - ix * new_zoom
        self.pan_y = anchor_sy - iy * new_zoom

    def center_on_image(self, view_w, view_h, img_w, img_h) -> None:
        self.pan_x = (view_w - img_w * self.zoom) / 2.0
        self.pan_y = (view_h - img_h * self.zoom) / 2.0

    def fit_zoom(self, view_w, view_h, img_w, img_h) -> int:
        """Largest zoom level that shows the whole image with a small margin."""
        if img_w <= 0 or img_h <= 0:
            return 1
        avail_w = max(32, view_w - 24)
        avail_h = max(32, view_h - 24)
        best = ZOOM_LEVELS[0]
        for level in ZOOM_LEVELS:
            if img_w * level <= avail_w and img_h * level <= avail_h:
                best = level
        return best

    def clamp_pan(self, view_w, view_h, img_w, img_h, margin: int = 48) -> None:
        """Keep at least `margin` screen pixels of artwork inside the viewport."""
        span_x = img_w * self.zoom
        span_y = img_h * self.zoom
        min_x = -(span_x - min(margin, span_x))
        max_x = view_w - min(margin, span_x)
        min_y = -(span_y - min(margin, span_y))
        max_y = view_h - min(margin, span_y)
        self.pan_x = clamp(self.pan_x, min_x, max_x)
        self.pan_y = clamp(self.pan_y, min_y, max_y)


# --------------------------------------------------------------------------- #
# Checkerboard / grid helpers used by the renderer
# --------------------------------------------------------------------------- #

def make_checkerboard(x0: int, y0: int, w: int, h: int, cell: int) -> Image.Image:
    """Checkerboard (at *image* resolution) for the pixel window starting at
    (x0, y0).  The pattern is keyed to absolute image coordinates so it does
    not crawl while panning."""
    cell = max(1, int(cell))
    cx0 = x0 // cell
    cy0 = y0 // cell
    off_x = x0 - cx0 * cell
    off_y = y0 - cy0 * cell
    gw = (w + off_x + cell - 1) // cell
    gh = (h + off_y + cell - 1) // cell
    gw = max(gw, 1)
    gh = max(gh, 1)
    small = Image.new("RGBA", (gw, gh))
    data = []
    for j in range(gh):
        for i in range(gw):
            data.append(CHECKER_A if (cx0 + i + cy0 + j) % 2 == 0 else CHECKER_B)
    small.putdata(data)
    big = small.resize((gw * cell, gh * cell), NEAREST)
    return big.crop((off_x, off_y, off_x + w, off_y + h))


def build_grid_overlay(size, first_px: int, first_py: int, zoom: int,
                       major_every: int = 8) -> Image.Image:
    """Transparent overlay of pixel-grid lines for an already scaled bitmap.

    `first_px` / `first_py` are the image coordinates of the bitmap's top-left
    pixel; they decide where the stronger every-8-pixels lines fall.
    """
    w, h = size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cols = w // zoom + 1
    rows = h // zoom + 1
    for i in range(cols + 1):
        sx = i * zoom
        if sx > w - 1:
            sx = w - 1
        major = (first_px + i) % major_every == 0
        draw.line([(sx, 0), (sx, h - 1)], fill=GRID_MAJOR if major else GRID_MINOR)
    for j in range(rows + 1):
        sy = j * zoom
        if sy > h - 1:
            sy = h - 1
        major = (first_py + j) % major_every == 0
        draw.line([(0, sy), (w - 1, sy)], fill=GRID_MAJOR if major else GRID_MINOR)
    return overlay


# --------------------------------------------------------------------------- #
# Tools
#
# Every tool receives *logical pixel* coordinates that the application already
# converted from widget coordinates, so no tool contains zoom/pan maths.
# Tools that draw a preview render into `app.preview` (a throw-away RGBA layer
# the same size as the artwork) and only write to the document on release.
# --------------------------------------------------------------------------- #

class Tool:
    name = "tool"
    label = "Tool"
    cursor = "crosshair"

    def __init__(self, app):
        self.app = app
        self.active = False

    # Mouse handlers (logical pixel coordinates + the raw Tk event) --------- #
    def on_press(self, x, y, event):
        pass

    def on_drag(self, x, y, event):
        pass

    def on_release(self, x, y, event):
        pass

    def on_hover(self, x, y, event):
        pass

    def cancel(self):
        """Escape / tool switch: abandon anything in progress."""
        self.active = False
        self.app.clear_preview()

    # Helpers -------------------------------------------------------------- #
    def clamped(self, x, y):
        doc = self.app.doc
        return clamp(x, 0, doc.width - 1), clamp(y, 0, doc.height - 1)


class PencilTool(Tool):
    name = "pencil"
    label = "Pencil"
    erase = False

    def __init__(self, app):
        super().__init__(app)
        self.path = []
        self.painted = 0
        self.last = None

    def paint_color(self):
        return TRANSPARENT if self.erase else self.app.color

    def on_press(self, x, y, event):
        # One mouse-down..mouse-up stroke is a single undo step.
        self.app.doc.begin_operation("Erase" if self.erase else "Pencil")
        self.path = []
        self.painted = 0
        self.last = (x, y)
        self.active = True
        self._feed(x, y)
        self.app.request_render()

    def on_drag(self, x, y, event):
        if not self.active:
            return
        if self.last is not None and (x, y) != self.last:
            # Interpolate so fast drags never leave gaps between samples.
            for point in bresenham_line(self.last[0], self.last[1], x, y)[1:]:
                self._feed(point[0], point[1])
        self.last = (x, y)
        self.app.request_render()

    def on_release(self, x, y, event):
        if not self.active:
            return
        self._flush()
        self.active = False
        self.last = None
        self.app.after_edit()

    def cancel(self):
        if self.active:
            self._flush()
            self.active = False
            self.last = None
            self.app.after_edit()
        super().cancel()

    # Pixel-perfect handling: a point is only committed once the following
    # point is known, so an L-shaped corner can still be dropped.
    def _feed(self, x, y):
        app = self.app
        pixel_perfect = app.pixel_perfect.get() and app.brush_size == 1
        if not pixel_perfect:
            app.paint([(x, y)], self.paint_color())
            return
        if self.path and self.path[-1] == (x, y):
            return
        self.path.append((x, y))
        if len(self.path) >= 3 and is_pixel_perfect_corner(self.path[-3], self.path[-2],
                                                           self.path[-1]):
            del self.path[-2]
            self.painted = min(self.painted, len(self.path) - 1)
        while self.painted < len(self.path) - 1:
            app.paint([self.path[self.painted]], self.paint_color())
            self.painted += 1

    def _flush(self):
        while self.painted < len(self.path):
            self.app.paint([self.path[self.painted]], self.paint_color())
            self.painted += 1


class EraserTool(PencilTool):
    name = "eraser"
    label = "Eraser"
    erase = True


class ShapeTool(Tool):
    """Shared press/drag/release plumbing for line, rectangle and ellipse."""

    undo_label = "Shape"

    def __init__(self, app):
        super().__init__(app)
        self.start = None
        self.end = None

    def on_press(self, x, y, event):
        self.start = self.clamped(x, y)
        self.end = self.start
        self.active = True
        self._update(event)

    def on_drag(self, x, y, event):
        if not self.active:
            return
        self.end = self.clamped(x, y)
        self._update(event)

    def on_release(self, x, y, event):
        if not self.active:
            return
        self.end = self.clamped(x, y)
        self.active = False
        points = self.shape_points(event)
        self.app.clear_preview()
        if points:
            self.app.doc.begin_operation(self.undo_label)
            self.app.paint(points, self.app.color)
            self.app.after_edit()
        self.report(event, committed=True)

    def cancel(self):
        self.active = False
        self.start = self.end = None
        self.app.clear_preview()
        self.app.set_hint("")
        self.app.request_render()

    # Subclasses implement these ------------------------------------------ #
    def resolved_end(self, event):
        return self.end

    def shape_points(self, event):
        return []

    def report(self, event, committed=False):
        pass

    def _update(self, event):
        app = self.app
        preview = app.ensure_preview()
        points = self.shape_points(event)
        app.paint(points, app.color, target=preview)
        self.report(event)
        app.request_render()


class LineTool(ShapeTool):
    name = "line"
    label = "Line"
    undo_label = "Line"

    def resolved_end(self, event):
        if self.app.shift_held(event) and self.start:
            ex, ey = snap_line_endpoint(self.start[0], self.start[1],
                                       self.end[0], self.end[1])
            return self.clamped(ex, ey)
        return self.end

    def shape_points(self, event):
        if not self.start:
            return []
        ex, ey = self.resolved_end(event)
        return bresenham_line(self.start[0], self.start[1], ex, ey)

    def report(self, event, committed=False):
        if not self.start:
            return
        ex, ey = self.resolved_end(event)
        metrics = line_metrics(self.start[0], self.start[1], ex, ey)
        metrics["rasterized"] = len(bresenham_line(self.start[0], self.start[1], ex, ey))
        self.app.set_line_report(metrics, committed=committed)


class RectangleTool(ShapeTool):
    name = "rect"
    label = "Rectangle"
    undo_label = "Rectangle"

    def resolved_end(self, event):
        if self.app.shift_held(event) and self.start:
            ex, ey = square_endpoint(self.start[0], self.start[1],
                                     self.end[0], self.end[1])
            return self.clamped(ex, ey)
        return self.end

    def shape_points(self, event):
        if not self.start:
            return []
        ex, ey = self.resolved_end(event)
        box = (self.start[0], self.start[1], ex, ey)
        if self.app.shape_fill.get():
            return rect_fill_pixels(*box)
        return rect_outline_pixels(*box)

    def report(self, event, committed=False):
        if not self.start:
            return
        ex, ey = self.resolved_end(event)
        a, b, c, d = normalize_box(self.start[0], self.start[1], ex, ey)
        # Inclusive pixel counts: a rectangle from x=4 to x=28 is 25 px wide.
        width = c - a + 1
        height = d - b + 1
        self.app.set_box_report("Rectangle", a, b, c, d, width, height,
                                extra={"Area": f"{commas(width * height)} px\u00b2"},
                                committed=committed)


class EllipseTool(ShapeTool):
    name = "ellipse"
    label = "Circle / Ellipse"
    undo_label = "Ellipse"

    def resolved_end(self, event):
        if self.app.shift_held(event) and self.start:
            ex, ey = square_endpoint(self.start[0], self.start[1],
                                     self.end[0], self.end[1])
            return self.clamped(ex, ey)
        return self.end

    def shape_points(self, event):
        if not self.start:
            return []
        ex, ey = self.resolved_end(event)
        box = (self.start[0], self.start[1], ex, ey)
        if self.app.shape_fill.get():
            return ellipse_fill_pixels(*box)
        return ellipse_outline_pixels(*box)

    def report(self, event, committed=False):
        if not self.start:
            return
        ex, ey = self.resolved_end(event)
        a, b, c, d = normalize_box(self.start[0], self.start[1], ex, ey)
        width = c - a + 1
        height = d - b + 1
        # An even diameter has a half-pixel radius, e.g. 30 px -> 15, 31 px -> 15.5
        extra = {
            "Radius X": f"{format_half(width / 2.0)} px",
            "Radius Y": f"{format_half(height / 2.0)} px",
        }
        self.app.set_box_report("Ellipse", a, b, c, d, width, height,
                                extra=extra, committed=committed)


class FillTool(Tool):
    name = "fill"
    label = "Fill Bucket"

    def on_press(self, x, y, event):
        app = self.app
        doc = app.doc
        if not doc.in_bounds(x, y):
            return
        doc.begin_operation("Fill")
        changed = 0
        for (sx, sy) in app.mirror_targets(x, y):
            if doc.in_bounds(sx, sy):
                changed += flood_fill_image(doc.image, sx, sy, app.color)
        if changed == 0:
            doc.abort_operation()
            app.set_hint("Fill: region already uses this colour")
        else:
            app.set_hint(f"Fill: {commas(changed)} px filled")
            app.after_edit()
        app.request_render()


class EyedropperTool(Tool):
    name = "picker"
    label = "Eyedropper"
    cursor = "dotbox"

    def on_press(self, x, y, event):
        self.active = True
        self._pick(x, y)

    def on_drag(self, x, y, event):
        if self.active:
            self._pick(x, y)

    def on_release(self, x, y, event):
        self.active = False

    def _pick(self, x, y):
        color = self.app.doc.get_pixel(x, y)
        if color is None:
            return
        self.app.set_color(color, add_recent=True)
        self.app.set_pick_report(x, y, color)


class MeasureTool(Tool):
    """Read-only tool: it never writes a single pixel into the artwork."""

    name = "measure"
    label = "Measure"
    cursor = "tcross"

    def on_press(self, x, y, event):
        point = self.clamped(x, y)
        self.active = True
        self.app.set_measurement(point, point)

    def on_drag(self, x, y, event):
        if not self.active:
            return
        end = self.clamped(x, y)
        if self.app.shift_held(event) and self.app.measurement:
            sx, sy = self.app.measurement[0]
            end = self.clamped(*snap_line_endpoint(sx, sy, end[0], end[1]))
        self.app.set_measurement(self.app.measurement[0], end)

    def on_release(self, x, y, event):
        if not self.active:
            return
        self.on_drag(x, y, event)
        self.active = False

    def cancel(self):
        self.active = False
        self.app.clear_measurement()


class PanTool(Tool):
    name = "pan"
    label = "Pan"
    cursor = "hand2"

    def on_press(self, x, y, event):
        self.app.begin_pan(event)

    def on_drag(self, x, y, event):
        self.app.drag_pan(event)

    def on_release(self, x, y, event):
        self.app.end_pan()


class SelectTool(Tool):
    """Rectangular marquee: select, drag to move, Delete/Ctrl+C/Ctrl+V."""

    name = "select"
    label = "Select"
    cursor = "crosshair"

    def __init__(self, app):
        super().__init__(app)
        self.mode = None
        self.start = None
        self.grab = None
        self.block = None
        self.origin_rect = None
        self.offset = (0, 0)

    def on_press(self, x, y, event):
        app = self.app
        px, py = self.clamped(x, y)
        sel = app.selection
        if sel and sel[0] <= px <= sel[2] and sel[1] <= py <= sel[3]:
            # Lift the selected block out of the artwork and float it.
            app.doc.begin_operation("Move Selection")
            self.block = app.doc.image.crop((sel[0], sel[1], sel[2] + 1, sel[3] + 1))
            app.doc.write_pixels(rect_fill_pixels(*sel), TRANSPARENT)
            self.mode = "move"
            self.origin_rect = sel
            self.grab = (px, py)
            self.offset = (0, 0)
            self.active = True
            self._preview_move()
        else:
            self.mode = "marquee"
            self.start = (px, py)
            self.active = True
            app.selection = (px, py, px, py)
            app.report_selection()
        app.request_render()

    def on_drag(self, x, y, event):
        if not self.active:
            return
        app = self.app
        px, py = self.clamped(x, y)
        if self.mode == "marquee":
            ex, ey = px, py
            if app.shift_held(event):
                ex, ey = self.clamped(*square_endpoint(self.start[0], self.start[1],
                                                       px, py))
            app.selection = normalize_box(self.start[0], self.start[1], ex, ey)
            app.report_selection()
        else:
            self.offset = (px - self.grab[0], py - self.grab[1])
            self._preview_move()
        app.request_render()

    def on_release(self, x, y, event):
        if not self.active:
            return
        app = self.app
        self.active = False
        if self.mode == "move" and self.block is not None:
            dx, dy = self.offset
            nx = self.origin_rect[0] + dx
            ny = self.origin_rect[1] + dy
            app.clear_preview()
            app.composite_block(self.block, nx, ny)
            app.selection = (nx, ny,
                             nx + self.block.size[0] - 1,
                             ny + self.block.size[1] - 1)
            self.block = None
            app.after_edit()
        app.report_selection()
        self.mode = None
        app.request_render()

    def cancel(self):
        app = self.app
        if self.active and self.mode == "move":
            # Put the floating block back exactly where it came from.
            app.doc.abort_operation()
            app.selection = self.origin_rect
            self.block = None
        elif not self.active:
            app.selection = None
        self.active = False
        self.mode = None
        app.clear_preview()
        app.report_selection()
        app.request_render()

    def _preview_move(self):
        app = self.app
        preview = app.ensure_preview()
        dx, dy = self.offset
        app.composite_block(self.block, self.origin_rect[0] + dx,
                            self.origin_rect[1] + dy, target=preview)
        app.preview_selection = (self.origin_rect[0] + dx, self.origin_rect[1] + dy,
                                 self.origin_rect[2] + dx, self.origin_rect[3] + dy)
        app.report_selection(self.offset)


TOOL_CLASSES = {
    "pencil": PencilTool,
    "eraser": EraserTool,
    "line": LineTool,
    "rect": RectangleTool,
    "ellipse": EllipseTool,
    "fill": FillTool,
    "picker": EyedropperTool,
    "measure": MeasureTool,
    "select": SelectTool,
    "pan": PanTool,
}


# --------------------------------------------------------------------------- #
# Dialogs
# --------------------------------------------------------------------------- #

def styled_button(parent, text, command, width=None, font=FONT_UI, **kwargs):
    return tk.Button(parent, text=text, command=command, width=width, font=font,
                     bg=C_BTN, fg=C_TEXT, activebackground=C_BTN_ACTIVE,
                     activeforeground=C_TEXT, relief="flat", bd=0,
                     highlightthickness=0, padx=8, pady=3, cursor="hand2", **kwargs)


def styled_label(parent, text="", font=FONT_UI, fg=C_TEXT, bg=C_PANEL, **kwargs):
    return tk.Label(parent, text=text, font=font, fg=fg, bg=bg, **kwargs)


def styled_entry(parent, textvariable, width=8):
    return tk.Entry(parent, textvariable=textvariable, width=width, font=FONT_MONO,
                    bg=C_PANEL_DARK, fg=C_TEXT, insertbackground=C_TEXT,
                    relief="flat", highlightthickness=1,
                    highlightbackground=C_BORDER, highlightcolor=C_ACCENT)


def styled_check(parent, text, variable, command=None):
    return tk.Checkbutton(parent, text=text, variable=variable, command=command,
                          font=FONT_UI, bg=C_PANEL, fg=C_TEXT,
                          activebackground=C_PANEL, activeforeground=C_TEXT,
                          selectcolor=C_PANEL_DARK, relief="flat",
                          highlightthickness=0, anchor="w", cursor="hand2")


def styled_radio(parent, text, variable, value, command=None):
    return tk.Radiobutton(parent, text=text, variable=variable, value=value,
                          command=command, font=FONT_UI, bg=C_PANEL, fg=C_TEXT,
                          activebackground=C_PANEL, activeforeground=C_TEXT,
                          selectcolor=C_PANEL_DARK, relief="flat",
                          highlightthickness=0, anchor="w", cursor="hand2")


class ModalDialog(tk.Toplevel):
    """Small helper base class: build in `body()`, read the value from `show()`."""

    def __init__(self, parent, title):
        super().__init__(parent)
        self.parent = parent
        self.result = None
        self.title(title)
        self.configure(bg=C_PANEL, padx=14, pady=12)
        self.resizable(False, False)
        try:
            self.transient(parent)
        except tk.TclError:
            pass
        self.protocol("WM_DELETE_WINDOW", self.on_cancel)
        self.bind("<Escape>", lambda _e: self.on_cancel())
        self.bind("<Return>", lambda _e: self.on_ok())

    def show(self):
        self.update_idletasks()
        self._center()
        try:
            self.wait_visibility()
            self.grab_set()
        except tk.TclError:
            pass
        self.focus_force()
        self.wait_window(self)
        return self.result

    def _center(self):
        try:
            px = self.parent.winfo_rootx()
            py = self.parent.winfo_rooty()
            pw = self.parent.winfo_width()
            ph = self.parent.winfo_height()
            if pw < 50:
                raise tk.TclError
        except tk.TclError:
            px = py = 0
            pw = self.winfo_screenwidth()
            ph = self.winfo_screenheight()
        w = self.winfo_reqwidth()
        h = self.winfo_reqheight()
        self.geometry("+%d+%d" % (px + max(0, (pw - w) // 2),
                                  py + max(0, (ph - h) // 3)))

    def on_ok(self):
        self.destroy()

    def on_cancel(self):
        self.result = None
        self.destroy()

    def section(self, text):
        lbl = styled_label(self, text, font=FONT_SECTION, fg=C_TEXT_DIM)
        lbl.pack(anchor="w", pady=(8, 2))
        return lbl


class NewCanvasDialog(ModalDialog):
    """Ask for exact canvas dimensions in pixels and preview the pixel count."""

    def __init__(self, parent, width=64, height=64, startup=False):
        super().__init__(parent, "New Canvas")
        self.startup = startup
        self.width_var = tk.StringVar(value=str(width))
        self.height_var = tk.StringVar(value=str(height))
        self.total_var = tk.StringVar()
        self.error_var = tk.StringVar()

        head = styled_label(self, "Canvas size in logical pixels",
                            font=FONT_UI_BOLD)
        head.pack(anchor="w")
        styled_label(self, f"Allowed range: {MIN_DIM} - {MAX_DIM} px per side",
                     fg=C_TEXT_DIM).pack(anchor="w", pady=(0, 6))

        grid = tk.Frame(self, bg=C_PANEL)
        grid.pack(anchor="w")
        styled_label(grid, "Width:").grid(row=0, column=0, sticky="w", pady=2)
        w_entry = styled_entry(grid, self.width_var)
        w_entry.grid(row=0, column=1, padx=(8, 4), pady=2)
        styled_label(grid, "px", fg=C_TEXT_DIM).grid(row=0, column=2, sticky="w")
        styled_label(grid, "Height:").grid(row=1, column=0, sticky="w", pady=2)
        styled_entry(grid, self.height_var).grid(row=1, column=1, padx=(8, 4), pady=2)
        styled_label(grid, "px", fg=C_TEXT_DIM).grid(row=1, column=2, sticky="w")

        styled_label(self, textvariable=self.total_var, font=FONT_MONO,
                     fg=C_ACCENT).pack(anchor="w", pady=(8, 0))
        styled_label(self, textvariable=self.error_var, fg="#ff7b72",
                     wraplength=280, justify="left").pack(anchor="w")

        self.section("PRESETS")
        presets = tk.Frame(self, bg=C_PANEL)
        presets.pack(anchor="w")
        for index, (pw, ph) in enumerate(PRESET_SIZES):
            styled_button(presets, f"{pw}x{ph}",
                          lambda a=pw, b=ph: self._use_preset(a, b),
                          width=8).grid(row=index // 3, column=index % 3,
                                        padx=2, pady=2)

        if startup:
            styled_label(self, "Cancel keeps the default 32 \u00d7 32 canvas; "
                               "you can start over with Ctrl+N.",
                         fg=C_TEXT_DIM, wraplength=280,
                         justify="left").pack(anchor="w", pady=(10, 0))

        buttons = tk.Frame(self, bg=C_PANEL)
        buttons.pack(anchor="e", pady=(14, 0))
        styled_button(buttons, "Cancel", self.on_cancel, width=8).pack(side="right",
                                                                      padx=(6, 0))
        create = styled_button(buttons, "Create Canvas", self.on_ok)
        create.pack(side="right")
        create.configure(bg=C_ACCENT, activebackground="#6aa0ff", fg="#ffffff")

        self.width_var.trace_add("write", lambda *_: self._refresh())
        self.height_var.trace_add("write", lambda *_: self._refresh())
        self._refresh()
        w_entry.focus_set()
        w_entry.selection_range(0, "end")

    def _use_preset(self, w, h):
        self.width_var.set(str(w))
        self.height_var.set(str(h))

    def _parse(self):
        w = parse_dimension(self.width_var.get(), "Width")
        h = parse_dimension(self.height_var.get(), "Height")
        if w * h > MAX_TOTAL_PIXELS:
            raise ValueError(f"Total pixels must not exceed {commas(MAX_TOTAL_PIXELS)}.")
        return w, h

    def _refresh(self):
        try:
            w, h = self._parse()
        except ValueError as exc:
            self.total_var.set("Total Pixels: --")
            self.error_var.set(str(exc))
            return
        self.total_var.set(f"Total Pixels: {commas(w * h)}")
        self.error_var.set("")

    def on_ok(self):
        try:
            w, h = self._parse()
        except ValueError as exc:
            self.error_var.set(str(exc))
            return
        self.result = (w, h)
        self.destroy()


class ExportDialog(ModalDialog):
    """Choose native or scaled (nearest-neighbour) PNG output."""

    def __init__(self, parent, doc_width, doc_height, last_color="#000000"):
        super().__init__(parent, "Export PNG")
        self.doc_width = doc_width
        self.doc_height = doc_height
        self.mode_var = tk.StringVar(value="native")
        self.scale_var = tk.IntVar(value=8)
        self.bg_var = tk.StringVar(value="transparent")
        self.grid_var = tk.BooleanVar(value=False)
        self.solid_color = last_color
        self.result_var = tk.StringVar()

        styled_label(self, "Native Size", font=FONT_SECTION,
                     fg=C_TEXT_DIM).pack(anchor="w")
        styled_label(self, f"{doc_width} \u00d7 {doc_height} px "
                           f"({commas(doc_width * doc_height)} pixels)",
                     font=FONT_MONO).pack(anchor="w")

        self.section("RESOLUTION")
        styled_radio(self, "Native resolution (1x, recommended)", self.mode_var,
                     "native", self._refresh).pack(anchor="w")
        styled_radio(self, "Scaled pixel-art resolution (nearest neighbour)",
                     self.mode_var, "scaled", self._refresh).pack(anchor="w")

        scales = tk.Frame(self, bg=C_PANEL)
        scales.pack(anchor="w", padx=(18, 0), pady=(2, 0))
        self.scale_buttons = []
        for index, factor in enumerate(EXPORT_SCALES):
            rb = styled_radio(scales, f"{factor}x", self.scale_var, factor,
                              self._refresh)
            rb.grid(row=0, column=index, sticky="w", padx=(0, 6))
            self.scale_buttons.append(rb)

        self.section("BACKGROUND")
        styled_radio(self, "Transparent (keep alpha)", self.bg_var, "transparent",
                     self._refresh).pack(anchor="w")
        solid_row = tk.Frame(self, bg=C_PANEL)
        solid_row.pack(anchor="w", fill="x")
        styled_radio(solid_row, "Solid colour", self.bg_var, "solid",
                     self._refresh).pack(side="left")
        self.swatch = tk.Frame(solid_row, bg=self.solid_color, width=26, height=16,
                               highlightthickness=1, highlightbackground=C_BORDER)
        self.swatch.pack(side="left", padx=6)
        styled_button(solid_row, "Pick...", self._pick_color).pack(side="left")

        self.grid_check = styled_check(self, "Burn pixel grid into the exported file",
                                      self.grid_var, self._refresh)
        self.grid_check.pack(anchor="w", pady=(8, 0))

        styled_label(self, textvariable=self.result_var, font=FONT_MONO,
                     fg=C_ACCENT).pack(anchor="w", pady=(10, 0))

        buttons = tk.Frame(self, bg=C_PANEL)
        buttons.pack(anchor="e", pady=(12, 0))
        styled_button(buttons, "Cancel", self.on_cancel, width=8).pack(side="right",
                                                                      padx=(6, 0))
        export = styled_button(buttons, "Choose File & Export", self.on_ok)
        export.pack(side="right")
        export.configure(bg=C_ACCENT, activebackground="#6aa0ff", fg="#ffffff")
        self._refresh()

    def _pick_color(self):
        chosen = colorchooser.askcolor(color=self.solid_color, parent=self,
                                       title="Export background colour")
        if chosen and chosen[1]:
            self.solid_color = chosen[1]
            self.swatch.configure(bg=self.solid_color)
            self.bg_var.set("solid")
            self._refresh()

    def effective_scale(self):
        return self.scale_var.get() if self.mode_var.get() == "scaled" else 1

    def _refresh(self):
        scale = self.effective_scale()
        state = "normal" if self.mode_var.get() == "scaled" else "disabled"
        for rb in self.scale_buttons:
            rb.configure(state=state)
        self.grid_check.configure(state="normal" if scale >= 2 else "disabled")
        out_w = self.doc_width * scale
        out_h = self.doc_height * scale
        self.result_var.set(
            f"Exported Image: {out_w} \u00d7 {out_h} px  "
            f"({commas(out_w * out_h)} pixels)")

    def on_ok(self):
        scale = self.effective_scale()
        background = None
        if self.bg_var.get() == "solid":
            try:
                background = parse_hex_color(self.solid_color)
            except ValueError:
                background = (0, 0, 0, 255)
        self.result = {
            "scale": scale,
            "background": background,
            "grid": bool(self.grid_var.get()) and scale >= 2,
        }
        self.destroy()


class ResizeDialog(ModalDialog):
    """Resize the canvas (crop/pad) or rescale the artwork - never zoom."""

    def __init__(self, parent, width, height, mode="canvas"):
        super().__init__(parent, "Resize Canvas / Image")
        self.cur_w = width
        self.cur_h = height
        self.width_var = tk.StringVar(value=str(width))
        self.height_var = tk.StringVar(value=str(height))
        self.mode_var = tk.StringVar(value=mode)
        self.anchor_var = tk.StringVar(value="nw")
        self.info_var = tk.StringVar()
        self.error_var = tk.StringVar()

        styled_label(self, f"Current canvas: {width} \u00d7 {height} px "
                           f"({commas(width * height)} pixels)",
                     font=FONT_MONO).pack(anchor="w")
        styled_label(self, "Changing these numbers changes the real image "
                           "resolution.\nZoom never does.",
                     fg=C_TEXT_DIM, justify="left").pack(anchor="w", pady=(2, 6))

        grid = tk.Frame(self, bg=C_PANEL)
        grid.pack(anchor="w")
        styled_label(grid, "Width:").grid(row=0, column=0, sticky="w", pady=2)
        entry = styled_entry(grid, self.width_var)
        entry.grid(row=0, column=1, padx=(8, 4))
        styled_label(grid, "px", fg=C_TEXT_DIM).grid(row=0, column=2)
        styled_label(grid, "Height:").grid(row=1, column=0, sticky="w", pady=2)
        styled_entry(grid, self.height_var).grid(row=1, column=1, padx=(8, 4))
        styled_label(grid, "px", fg=C_TEXT_DIM).grid(row=1, column=2)

        self.section("MODE")
        styled_radio(self, "Resize canvas - keep pixels, crop or pad",
                     self.mode_var, "canvas", self._refresh).pack(anchor="w")
        styled_radio(self, "Resize image - rescale artwork (nearest neighbour)",
                     self.mode_var, "image", self._refresh).pack(anchor="w")

        self.anchor_frame = tk.Frame(self, bg=C_PANEL)
        self.anchor_frame.pack(anchor="w", padx=(18, 0))
        styled_label(self.anchor_frame, "Anchor:", fg=C_TEXT_DIM).pack(side="left")
        for value, text in (("nw", "Top-Left"), ("center", "Centre")):
            styled_radio(self.anchor_frame, text, self.anchor_var, value).pack(side="left",
                                                                              padx=4)

        styled_label(self, textvariable=self.info_var, font=FONT_MONO,
                     fg=C_ACCENT).pack(anchor="w", pady=(10, 0))
        styled_label(self, textvariable=self.error_var, fg="#ff7b72",
                     wraplength=300, justify="left").pack(anchor="w")

        buttons = tk.Frame(self, bg=C_PANEL)
        buttons.pack(anchor="e", pady=(12, 0))
        styled_button(buttons, "Cancel", self.on_cancel, width=8).pack(side="right",
                                                                      padx=(6, 0))
        apply_btn = styled_button(buttons, "Apply", self.on_ok, width=8)
        apply_btn.pack(side="right")
        apply_btn.configure(bg=C_ACCENT, activebackground="#6aa0ff", fg="#ffffff")

        self.width_var.trace_add("write", lambda *_: self._refresh())
        self.height_var.trace_add("write", lambda *_: self._refresh())
        self._refresh()
        entry.focus_set()

    def _parse(self):
        w = parse_dimension(self.width_var.get(), "Width")
        h = parse_dimension(self.height_var.get(), "Height")
        if w * h > MAX_TOTAL_PIXELS:
            raise ValueError(f"Total pixels must not exceed {commas(MAX_TOTAL_PIXELS)}.")
        return w, h

    def _refresh(self):
        state = "normal" if self.mode_var.get() == "canvas" else "disabled"
        for child in self.anchor_frame.winfo_children():
            if isinstance(child, tk.Radiobutton):
                child.configure(state=state)
        try:
            w, h = self._parse()
        except ValueError as exc:
            self.info_var.set("New Total Pixels: --")
            self.error_var.set(str(exc))
            return
        self.info_var.set(f"New Total Pixels: {commas(w * h)}")
        self.error_var.set("")

    def on_ok(self):
        try:
            w, h = self._parse()
        except ValueError as exc:
            self.error_var.set(str(exc))
            return
        self.result = {"width": w, "height": h, "mode": self.mode_var.get(),
                       "anchor": self.anchor_var.get()}
        self.destroy()


# --------------------------------------------------------------------------- #
# Tooltips
# --------------------------------------------------------------------------- #

class Tooltip:
    """Minimal hover tooltip (also used to surface keyboard shortcuts)."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.window = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _show(self, _event=None):
        if self.window or not self.text:
            return
        x = self.widget.winfo_rootx() + 14
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        self.window.wm_geometry(f"+{x}+{y}")
        tk.Label(self.window, text=self.text, font=FONT_UI, bg="#0f1012",
                 fg=C_TEXT, bd=1, relief="solid", padx=6, pady=2,
                 justify="left").pack()

    def _hide(self, _event=None):
        if self.window is not None:
            self.window.destroy()
            self.window = None


# --------------------------------------------------------------------------- #
# The application
# --------------------------------------------------------------------------- #

TOOL_KEYS = {
    "p": "pencil", "e": "eraser", "l": "line", "r": "rect", "c": "ellipse",
    "f": "fill", "i": "picker", "m": "measure", "s": "select", "h": "pan",
}


class PixelEditorApp:
    """Owns the document, the viewport, the tools and the whole Tk interface."""

    def __init__(self, root: tk.Tk, width: int = 32, height: int = 32):
        self.root = root
        self.doc = PixelDocument(width, height)
        self.viewport = Viewport(DEFAULT_ZOOM)

        # Drawing state -------------------------------------------------- #
        self.color = (255, 255, 255, 255)
        self.recent_colors = [(255, 255, 255, 255), (0, 0, 0, 255),
                              (255, 64, 64, 255), (64, 200, 96, 255),
                              (72, 140, 255, 255), (255, 208, 64, 255)]
        self.preview = None            # transient RGBA layer, never exported
        self.preview_selection = None
        self.selection = None          # (x0, y0, x1, y1) inclusive
        self.clipboard = None
        self.measurement = None        # ((x0, y0), (x1, y1)) overlay only
        self.hover_pixel = None
        self.hint = ""
        self.report_title = "Info"
        self.report_lines = ["Draw a line, shape or measurement", "to see numbers here."]
        self.export_bg_hex = "#000000"

        # Interaction state ---------------------------------------------- #
        self._render_pending = False
        self._photo = None
        self._checker_cache = None
        self._grid_cache = None
        self._pending_fit = None
        self._pan_state = None
        self._alt_pick = False
        self._dragging = False
        self.space_held = False
        self._did_initial_fit = False

        # Tk variables ---------------------------------------------------- #
        self.brush_var = tk.IntVar(value=1)
        self.alpha_var = tk.IntVar(value=255)
        self.hex_var = tk.StringVar(value=rgb_hex(self.color))
        self.shape_fill = tk.BooleanVar(value=False)
        self.mirror_x = tk.BooleanVar(value=False)
        self.mirror_y = tk.BooleanVar(value=False)
        self.pixel_perfect = tk.BooleanVar(value=True)
        self.show_grid = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar()
        self.header_var = tk.StringVar()
        self.canvas_info_var = tk.StringVar()
        self.report_var = tk.StringVar()
        self.report_title_var = tk.StringVar(value="INFO")
        self.tool_var = tk.StringVar(value="Pencil")
        self.zoom_var = tk.StringVar(value="800%")
        self.brush_label_var = tk.StringVar(value="Brush: 1 px")
        self.color_text_var = tk.StringVar()

        # Tools ----------------------------------------------------------- #
        self.tools = {key: cls(self) for key, cls in TOOL_CLASSES.items()}
        self.tool = self.tools["pencil"]

        self._build_ui()
        self._bind_events()
        self.set_tool("pencil")
        self.refresh_color_widgets()
        self.update_title()
        self.update_readouts()
        self.update_report()

    # ------------------------------------------------------------------ #
    # Convenience accessors
    # ------------------------------------------------------------------ #

    @property
    def brush_size(self) -> int:
        try:
            return clamp(int(self.brush_var.get()), BRUSH_MIN, BRUSH_MAX)
        except (tk.TclError, ValueError):
            return 1

    @staticmethod
    def shift_held(event) -> bool:
        return bool(getattr(event, "state", 0) & 0x0001)

    @staticmethod
    def ctrl_held(event) -> bool:
        return bool(getattr(event, "state", 0) & 0x0004)

    @staticmethod
    def alt_held(event) -> bool:
        state = getattr(event, "state", 0)
        # Tk reports Alt as 0x20000 on Windows/macOS and as Mod1 (0x8) on X11.
        if sys.platform.startswith("win") or sys.platform == "darwin":
            return bool(state & 0x20000)
        return bool(state & 0x0008)

    # ------------------------------------------------------------------ #
    # Interface construction
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        root = self.root
        root.title(APP_NAME)
        root.configure(bg=C_BG)
        root.geometry("1300x840")
        root.minsize(940, 620)

        self._build_menu()

        # Top toolbar ----------------------------------------------------- #
        toolbar = tk.Frame(root, bg=C_PANEL, padx=6, pady=5)
        toolbar.pack(side="top", fill="x")
        actions = [
            ("New", self.new_canvas, "New canvas (Ctrl+N)"),
            ("Open PNG", self.open_png, "Import a PNG at its native size (Ctrl+O)"),
            ("Save Project", self.save_project, "Save editable project (Ctrl+Shift+S)"),
            ("Export PNG", self.export_png, "Export PNG, native or scaled (Ctrl+S)"),
            None,
            ("Undo", self.undo, "Undo (Ctrl+Z)"),
            ("Redo", self.redo, "Redo (Ctrl+Y)"),
            None,
            ("Zoom -", lambda: self.zoom_step(-1), "Zoom out (-)"),
            ("Zoom +", lambda: self.zoom_step(1), "Zoom in (+)"),
            ("Reset Zoom", self.reset_zoom, "Back to 100% (0)"),
            ("Fit", self.fit_to_window, "Fit canvas in window (Ctrl+0)"),
            ("Grid", self.toggle_grid, "Toggle pixel grid (G)"),
        ]
        for item in actions:
            if item is None:
                tk.Frame(toolbar, bg=C_BORDER, width=1).pack(side="left", fill="y",
                                                             padx=5, pady=2)
                continue
            text, command, tip = item
            btn = styled_button(toolbar, text, command)
            btn.pack(side="left", padx=2)
            Tooltip(btn, tip)

        styled_label(toolbar, textvariable=self.header_var, font=FONT_MONO,
                     fg=C_TEXT_DIM).pack(side="right", padx=8)

        # Middle area ----------------------------------------------------- #
        middle = tk.Frame(root, bg=C_BG)
        middle.pack(side="top", fill="both", expand=True)

        # Left tool column
        left = tk.Frame(middle, bg=C_PANEL, padx=5, pady=6)
        left.pack(side="left", fill="y")
        styled_label(left, "TOOLS", font=FONT_SECTION, fg=C_TEXT_DIM).pack(anchor="w",
                                                                          pady=(0, 4))
        self.tool_buttons = {}
        tool_tips = {
            "pencil": "Pencil - paint pixels (P)",
            "eraser": "Eraser - erase to transparent (E)",
            "line": "Line - Bresenham line with live measurements (L)",
            "rect": "Rectangle - outline or filled (R)",
            "ellipse": "Circle / Ellipse - hold Shift for a circle (C)",
            "fill": "Fill bucket - 4-connected flood fill (F)",
            "picker": "Eyedropper - pick a pixel colour (I)",
            "measure": "Measure - distances without touching the artwork (M)",
            "select": "Select - marquee, drag to move (S)",
            "pan": "Pan - drag the view (H, or hold Space / middle mouse)",
        }
        for key, label, shortcut in TOOL_SPECS:
            btn = styled_button(left, f"{label} ({shortcut})",
                                lambda k=key: self.set_tool(k), width=12)
            btn.pack(fill="x", pady=1)
            Tooltip(btn, tool_tips.get(key, label))
            self.tool_buttons[key] = btn

        tk.Frame(left, bg=C_BORDER, height=1).pack(fill="x", pady=8)
        styled_label(left, "OPTIONS", font=FONT_SECTION, fg=C_TEXT_DIM).pack(anchor="w")
        styled_check(left, "Filled shapes", self.shape_fill,
                     self.request_render).pack(fill="x")
        styled_check(left, "Mirror X", self.mirror_x).pack(fill="x")
        styled_check(left, "Mirror Y", self.mirror_y).pack(fill="x")
        styled_check(left, "Pixel perfect", self.pixel_perfect).pack(fill="x")
        styled_check(left, "Show grid", self.show_grid,
                     self.request_render).pack(fill="x")

        # Centre canvas
        center = tk.Frame(middle, bg=C_BG, bd=0)
        center.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(center, bg=C_CANVAS_BG, highlightthickness=0,
                                bd=0, cursor="crosshair", takefocus=True)
        self.canvas.pack(fill="both", expand=True)

        # Right panel
        self._build_right_panel(middle)

        # Status bar ------------------------------------------------------ #
        status = tk.Frame(root, bg=C_PANEL_DARK)
        status.pack(side="bottom", fill="x")
        styled_label(status, textvariable=self.status_var, font=FONT_MONO,
                     bg=C_PANEL_DARK, fg=C_TEXT_DIM, anchor="w").pack(side="left",
                                                                      padx=8, pady=3)

    def _build_right_panel(self, parent):
        right = tk.Frame(parent, bg=C_PANEL, padx=8, pady=6, width=232)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        def section(text):
            styled_label(right, text, font=FONT_SECTION,
                         fg=C_TEXT_DIM).pack(anchor="w", pady=(10, 3))

        section("COLOR")
        self.color_swatch = tk.Frame(right, bg=rgb_hex(self.color), height=30,
                                     highlightthickness=1,
                                     highlightbackground=C_BORDER)
        self.color_swatch.pack(fill="x")
        styled_label(right, textvariable=self.color_text_var, font=FONT_MONO,
                     fg=C_TEXT_DIM, justify="left", anchor="w").pack(fill="x",
                                                                    pady=(3, 0))
        hex_row = tk.Frame(right, bg=C_PANEL)
        hex_row.pack(fill="x", pady=(4, 0))
        styled_label(hex_row, "Hex").pack(side="left")
        self.hex_entry = styled_entry(hex_row, self.hex_var, width=10)
        self.hex_entry.pack(side="left", padx=4)
        self.hex_entry.bind("<Return>", lambda _e: self.apply_hex())
        self.hex_entry.bind("<FocusOut>", lambda _e: self.apply_hex(silent=True))
        styled_button(hex_row, "Set", self.apply_hex).pack(side="left")
        styled_button(right, "Colour Picker...", self.choose_color).pack(fill="x",
                                                                        pady=(4, 0))

        alpha_row = tk.Frame(right, bg=C_PANEL)
        alpha_row.pack(fill="x")
        styled_label(alpha_row, "Alpha").pack(side="left")
        scale = tk.Scale(alpha_row, from_=0, to=255, orient="horizontal",
                         variable=self.alpha_var, bg=C_PANEL, fg=C_TEXT,
                         troughcolor=C_PANEL_DARK, highlightthickness=0,
                         sliderrelief="flat", bd=0, font=FONT_UI, length=140,
                         command=lambda _v: self.apply_alpha())
        scale.pack(side="left", fill="x", expand=True)

        section("RECENT COLORS")
        self.recent_frame = tk.Frame(right, bg=C_PANEL)
        self.recent_frame.pack(fill="x")

        section("BRUSH")
        styled_label(right, textvariable=self.brush_label_var,
                     font=FONT_MONO).pack(anchor="w")
        tk.Scale(right, from_=BRUSH_MIN, to=BRUSH_MAX, orient="horizontal",
                 variable=self.brush_var, bg=C_PANEL, fg=C_TEXT,
                 troughcolor=C_PANEL_DARK, highlightthickness=0,
                 sliderrelief="flat", bd=0, font=FONT_UI, showvalue=False,
                 command=lambda _v: self.on_brush_change()).pack(fill="x")

        section("CANVAS")
        styled_label(right, textvariable=self.canvas_info_var, font=FONT_MONO,
                     justify="left", anchor="w").pack(fill="x")

        section("VIEW")
        view_row = tk.Frame(right, bg=C_PANEL)
        view_row.pack(fill="x")
        styled_label(view_row, "Zoom").pack(side="left")
        styled_label(view_row, textvariable=self.zoom_var, font=FONT_MONO,
                     fg=C_ACCENT).pack(side="left", padx=4)
        tool_row = tk.Frame(right, bg=C_PANEL)
        tool_row.pack(fill="x")
        styled_label(tool_row, "Tool").pack(side="left")
        styled_label(tool_row, textvariable=self.tool_var, font=FONT_MONO,
                     fg=C_ACCENT).pack(side="left", padx=4)

        tk.Frame(right, bg=C_BORDER, height=1).pack(fill="x", pady=(10, 0))
        styled_label(right, textvariable=self.report_title_var, font=FONT_SECTION,
                     fg=C_TEXT_DIM).pack(anchor="w", pady=(8, 3))
        styled_label(right, textvariable=self.report_var, font=FONT_MONO,
                     justify="left", anchor="nw", wraplength=210).pack(fill="x")
        styled_button(right, "Clear Measurement",
                      self.clear_measurement).pack(fill="x", pady=(8, 0))
        self.rebuild_recent_swatches()

    def _build_menu(self):
        root = self.root
        menubar = tk.Menu(root, bg=C_PANEL, fg=C_TEXT, activebackground=C_ACCENT,
                          activeforeground="#ffffff", bd=0)

        def make_menu():
            return tk.Menu(menubar, tearoff=0, bg=C_PANEL, fg=C_TEXT,
                           activebackground=C_ACCENT, activeforeground="#ffffff",
                           selectcolor=C_ACCENT, bd=0)

        file_menu = make_menu()
        file_menu.add_command(label="New Canvas...", accelerator="Ctrl+N",
                              command=self.new_canvas)
        file_menu.add_command(label="Open PNG...", accelerator="Ctrl+O",
                              command=self.open_png)
        file_menu.add_command(label="Open Project...", command=self.open_project)
        file_menu.add_separator()
        file_menu.add_command(label="Save Project", accelerator="Ctrl+Shift+S",
                              command=self.save_project)
        file_menu.add_command(label="Export PNG...", accelerator="Ctrl+S",
                              command=self.export_png)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        edit_menu = make_menu()
        edit_menu.add_command(label="Undo", accelerator="Ctrl+Z", command=self.undo)
        edit_menu.add_command(label="Redo", accelerator="Ctrl+Y", command=self.redo)
        edit_menu.add_separator()
        edit_menu.add_command(label="Cut", accelerator="Ctrl+X", command=self.cut)
        edit_menu.add_command(label="Copy", accelerator="Ctrl+C", command=self.copy)
        edit_menu.add_command(label="Paste", accelerator="Ctrl+V", command=self.paste)
        edit_menu.add_command(label="Delete Selection", accelerator="Del",
                              command=self.delete_selection)
        edit_menu.add_separator()
        edit_menu.add_command(label="Select All", accelerator="Ctrl+A",
                              command=self.select_all)
        edit_menu.add_command(label="Deselect", command=self.deselect)
        edit_menu.add_separator()
        edit_menu.add_command(label="Clear Canvas", command=self.clear_canvas)
        menubar.add_cascade(label="Edit", menu=edit_menu)

        view_menu = make_menu()
        view_menu.add_command(label="Zoom In", accelerator="+",
                              command=lambda: self.zoom_step(1))
        view_menu.add_command(label="Zoom Out", accelerator="-",
                              command=lambda: self.zoom_step(-1))
        view_menu.add_command(label="Reset Zoom (100%)", accelerator="0",
                              command=self.reset_zoom)
        view_menu.add_command(label="Fit In Window", accelerator="Ctrl+0",
                              command=self.fit_to_window)
        zoom_menu = make_menu()
        for level in ZOOM_LEVELS:
            zoom_menu.add_command(label=f"{level * 100}%",
                                  command=lambda z=level: self.set_zoom(z))
        view_menu.add_cascade(label="Zoom Level", menu=zoom_menu)
        view_menu.add_separator()
        view_menu.add_checkbutton(label="Pixel Grid", accelerator="G",
                                  variable=self.show_grid,
                                  command=self.request_render)
        view_menu.add_command(label="Centre Canvas", command=self.center_view)
        menubar.add_cascade(label="View", menu=view_menu)

        image_menu = make_menu()
        image_menu.add_command(label="Resize Canvas...",
                               command=lambda: self.resize_dialog("canvas"))
        image_menu.add_command(label="Resize Image (rescale artwork)...",
                               command=lambda: self.resize_dialog("image"))
        menubar.add_cascade(label="Image", menu=image_menu)

        tools_menu = make_menu()
        for key, label, shortcut in TOOL_SPECS:
            tools_menu.add_command(label=label, accelerator=shortcut,
                                   command=lambda k=key: self.set_tool(k))
        tools_menu.add_separator()
        tools_menu.add_checkbutton(label="Filled Shapes", variable=self.shape_fill)
        tools_menu.add_checkbutton(label="Mirror X", variable=self.mirror_x)
        tools_menu.add_checkbutton(label="Mirror Y", variable=self.mirror_y)
        tools_menu.add_checkbutton(label="Pixel-Perfect Pencil",
                                   variable=self.pixel_perfect)
        tools_menu.add_separator()
        tools_menu.add_command(label="Clear Measurement", accelerator="Esc",
                               command=self.clear_measurement)
        menubar.add_cascade(label="Tools", menu=tools_menu)

        help_menu = make_menu()
        help_menu.add_command(label="Keyboard Shortcuts", command=self.show_shortcuts)
        help_menu.add_command(label="Run Accuracy Self-Tests",
                              command=self.show_self_tests)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        root.config(menu=menubar)

    def _bind_events(self):
        canvas = self.canvas
        canvas.bind("<Configure>", self.on_configure)
        canvas.bind("<Button-1>", self.on_press)
        canvas.bind("<B1-Motion>", self.on_drag)
        canvas.bind("<ButtonRelease-1>", self.on_release)
        canvas.bind("<Motion>", self.on_hover)
        canvas.bind("<Leave>", self.on_leave)
        canvas.bind("<Button-2>", lambda e: self.begin_pan(e))
        canvas.bind("<B2-Motion>", lambda e: self.drag_pan(e))
        canvas.bind("<ButtonRelease-2>", lambda e: self.end_pan())
        canvas.bind("<MouseWheel>", self.on_wheel)              # Windows / macOS
        canvas.bind("<Button-4>", lambda e: self.on_wheel(e, 120))   # X11 up
        canvas.bind("<Button-5>", lambda e: self.on_wheel(e, -120))  # X11 down

        root = self.root
        root.bind("<KeyPress>", self.on_keypress)
        root.bind("<KeyPress-space>", self.on_space_press)
        root.bind("<KeyRelease-space>", self.on_space_release)
        root.bind("<Escape>", lambda _e: self.cancel_current())
        root.bind("<Control-n>", lambda _e: self.new_canvas())
        root.bind("<Control-o>", lambda _e: self.open_png())
        root.bind("<Control-s>", lambda _e: self.export_png())
        root.bind("<Control-S>", lambda _e: self.save_project())
        root.bind("<Control-z>", lambda _e: self.undo())
        root.bind("<Control-y>", lambda _e: self.redo())
        root.bind("<Control-Z>", lambda _e: self.redo())  # Ctrl+Shift+Z
        root.bind("<Control-a>", lambda _e: self.select_all())
        root.bind("<Control-c>", lambda _e: self.copy())
        root.bind("<Control-x>", lambda _e: self.cut())
        root.bind("<Control-v>", lambda _e: self.paste())
        root.bind("<Control-Key-0>", lambda _e: self.fit_to_window())
        root.protocol("WM_DELETE_WINDOW", self.on_close)


    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def request_render(self):
        """Coalesce repaint requests so one mouse event never renders twice."""
        if self._render_pending:
            return
        self._render_pending = True
        self.root.after_idle(self.render)

    def render(self):
        self._render_pending = False
        if ImageTk is None:
            return
        canvas = self.canvas
        view_w = canvas.winfo_width()
        view_h = canvas.winfo_height()
        if view_w < 10 or view_h < 10:
            return

        vp = self.viewport
        doc = self.doc
        img_w, img_h = doc.size
        vp.clamp_pan(view_w, view_h, img_w, img_h)
        canvas.delete("all")

        # Only the visible pixel window is ever scaled, so extreme zoom on a
        # large canvas still costs a viewport-sized bitmap.
        x0, y0, x1, y1 = vp.visible_pixel_bounds(view_w, view_h, img_w, img_h)
        if x1 > x0 and y1 > y0:
            box = (x0, y0, x1, y1)
            artwork = doc.image.crop(box)
            if self.preview is not None:
                # Previews live in their own layer: the artwork is untouched.
                artwork = Image.alpha_composite(artwork, self.preview.crop(box))
            # Checkerboard cells stay roughly 8 screen pixels wide.
            cell = max(1, 8 // vp.zoom)
            checker = self._checkerboard_for(x0, y0, x1 - x0, y1 - y0, cell)
            composite = Image.alpha_composite(checker, artwork)
            scaled = composite.resize(((x1 - x0) * vp.zoom, (y1 - y0) * vp.zoom),
                                      NEAREST)  # never blur pixel art
            if self.show_grid.get() and vp.zoom >= GRID_MIN_ZOOM:
                overlay = self._grid_overlay_for(scaled.size, x0, y0, vp.zoom)
                scaled = Image.alpha_composite(scaled, overlay)
            self._photo = ImageTk.PhotoImage(scaled)
            sx, sy = vp.pixel_to_screen(x0, y0)
            canvas.create_image(sx, sy, anchor="nw", image=self._photo)

        # Canvas border
        bx0, by0 = vp.pixel_to_screen(0, 0)
        bx1, by1 = vp.pixel_to_screen(img_w, img_h)
        canvas.create_rectangle(bx0 - 1, by0 - 1, bx1, by1, outline=C_OUTLINE)

        self._draw_selection()
        self._draw_measurement()
        self._draw_hover()

    # The checkerboard and the grid only change when the view changes, so they
    # are cached across the many repaints of a single drawing stroke.
    def _checkerboard_for(self, x0, y0, w, h, cell):
        key = (x0, y0, w, h, cell)
        if self._checker_cache is not None and self._checker_cache[0] == key:
            return self._checker_cache[1]
        image = make_checkerboard(x0, y0, w, h, cell)
        self._checker_cache = (key, image)
        return image

    def _grid_overlay_for(self, size, first_px, first_py, zoom):
        key = (size, first_px % 8, first_py % 8, zoom)
        if self._grid_cache is not None and self._grid_cache[0] == key:
            return self._grid_cache[1]
        overlay = build_grid_overlay(size, first_px, first_py, zoom)
        self._grid_cache = (key, overlay)
        return overlay

    def _draw_selection(self):
        rect = self.preview_selection or self.selection
        if not rect:
            return
        vp = self.viewport
        x0, y0 = vp.pixel_to_screen(rect[0], rect[1])
        x1, y1 = vp.pixel_to_screen(rect[2] + 1, rect[3] + 1)
        self.canvas.create_rectangle(x0, y0, x1, y1, outline="#000000",
                                     width=1, tags="selection")
        self.canvas.create_rectangle(x0, y0, x1, y1, outline="#ffffff",
                                     width=1, dash=(4, 4), tags="selection")

    def _draw_measurement(self):
        if not self.measurement:
            return
        (mx0, my0), (mx1, my1) = self.measurement
        vp = self.viewport
        half = vp.zoom / 2.0
        ax, ay = vp.pixel_to_screen(mx0, my0)
        bx, by = vp.pixel_to_screen(mx1, my1)
        ax, ay = ax + half, ay + half            # centre of the start pixel
        bx, by = bx + half, by + half            # centre of the end pixel
        canvas = self.canvas
        canvas.create_line(ax, ay, bx, by, fill="#000000", width=3,
                           tags="measure")
        canvas.create_line(ax, ay, bx, by, fill=C_MEASURE, width=1,
                           tags="measure")
        for (px, py) in ((mx0, my0), (mx1, my1)):
            rx0, ry0, rx1, ry1 = vp.pixel_rect_on_screen(px, py)
            canvas.create_rectangle(rx0, ry0, rx1, ry1, outline=C_MEASURE,
                                    tags="measure")
        metrics = line_metrics(mx0, my0, mx1, my1)
        label = (f"{metrics['distance']:.2f} px  "
                 f"{metrics['angle']:.2f}\u00b0  "
                 f"(\u0394{metrics['abs_dx']}, \u0394{metrics['abs_dy']})")
        # Offset the label perpendicular to the line so it never covers it.
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length, dx / length
        if ny > 0:
            nx, ny = -nx, -ny
        tx = (ax + bx) / 2.0 + nx * 20
        ty = (ay + by) / 2.0 + ny * 20
        canvas.create_text(tx + 1, ty + 1, text=label, fill="#000000",
                           font=FONT_MONO, tags="measure")
        canvas.create_text(tx, ty, text=label, fill=C_MEASURE, font=FONT_MONO,
                           tags="measure")

    def _draw_hover(self):
        """Cursor cell outline; redrawn on its own so plain mouse moves are cheap."""
        canvas = self.canvas
        canvas.delete("hover")
        if self.hover_pixel is None or self.viewport.zoom < 3:
            return
        x, y = self.hover_pixel
        if not self.doc.in_bounds(x, y):
            return
        size = self.brush_size
        start = -(size // 2) if size > 1 else 0
        vp = self.viewport
        x0, y0 = vp.pixel_to_screen(x + start, y + start)
        x1, y1 = vp.pixel_to_screen(x + start + size, y + start + size)
        canvas.create_rectangle(x0, y0, x1, y1, outline="#000000", tags="hover")
        canvas.create_rectangle(x0 - 1, y0 - 1, x1 + 1, y1 + 1,
                                outline="#ffffff", tags="hover")

    # ------------------------------------------------------------------ #
    # Painting helpers shared by all tools
    # ------------------------------------------------------------------ #

    def ensure_preview(self) -> Image.Image:
        """Return a fresh, empty preview layer (display only, never exported)."""
        self.preview = Image.new("RGBA", self.doc.size, TRANSPARENT)
        return self.preview

    def clear_preview(self):
        self.preview = None
        self.preview_selection = None

    def mirror_targets(self, x, y):
        """A point plus its mirrored partners for the active mirror modes."""
        w, h = self.doc.size
        points = [(x, y)]
        mx, my = self.mirror_x.get(), self.mirror_y.get()
        if mx:
            points.append((w - 1 - x, y))
        if my:
            points.append((x, h - 1 - y))
        if mx and my:
            points.append((w - 1 - x, h - 1 - y))
        return points

    def paint(self, points, color, target=None, size=None):
        """Write `color` at `points`, expanded by the brush and mirror settings.

        `target` selects the preview layer instead of the artwork.
        """
        if not points:
            return 0
        size = self.brush_size if size is None else size
        coords = expand_brush(points, size)
        if self.mirror_x.get() or self.mirror_y.get():
            mirrored = []
            for (x, y) in coords:
                mirrored.extend(self.mirror_targets(x, y))
            coords = mirrored
        if target is None:
            return self.doc.write_pixels(coords, color)
        tw, th = target.size
        px = target.load()
        count = 0
        for (x, y) in coords:
            if 0 <= x < tw and 0 <= y < th:
                px[x, y] = color
                count += 1
        return count

    def composite_block(self, block: Image.Image, x: int, y: int, target=None):
        """Alpha-composite an RGBA block at (x, y), clipped to the canvas."""
        image = self.doc.image if target is None else target
        iw, ih = image.size
        bw, bh = block.size
        sx0 = max(0, -x)
        sy0 = max(0, -y)
        sx1 = min(bw, iw - x)
        sy1 = min(bh, ih - y)
        if sx1 <= sx0 or sy1 <= sy0:
            return
        piece = block.crop((sx0, sy0, sx1, sy1))
        image.alpha_composite(piece, (x + sx0, y + sy0))

    def after_edit(self):
        """Called once a document-modifying operation is complete."""
        self.doc.modified = True
        self.update_title()
        self.update_readouts()
        self.request_render()

    # ------------------------------------------------------------------ #
    # Read-outs / reports
    # ------------------------------------------------------------------ #

    def set_hint(self, text):
        self.hint = text or ""
        self.update_readouts()

    def update_title(self):
        doc = self.doc
        name = os.path.basename(doc.project_path or doc.export_path or "Untitled")
        mark = "*" if doc.modified else ""
        self.root.title(f"{APP_NAME} - {name}{mark} - {doc.width} \u00d7 "
                        f"{doc.height} px")

    def update_readouts(self):
        doc = self.doc
        vp = self.viewport
        if self.hover_pixel is not None and doc.in_bounds(*self.hover_pixel):
            cx, cy = self.hover_pixel
            cur_x, cur_y = str(cx), str(cy)
            cursor_line = f"Cursor: X {cx}, Y {cy}"
        else:
            cur_x = cur_y = "-"
            cursor_line = "Cursor: outside canvas"
        self.canvas_info_var.set("\n".join([
            f"Canvas: {doc.width} \u00d7 {doc.height} px",
            f"Width:  {doc.width} px",
            f"Height: {doc.height} px",
            f"Total Pixels: {commas(doc.total_pixels)}",
            f"Zoom: {vp.percent}%",
            cursor_line,
        ]))
        self.header_var.set(f"{doc.width} \u00d7 {doc.height} px   |   "
                            f"Zoom {vp.percent}%")
        self.zoom_var.set(f"{vp.percent}%")
        parts = [self.tool.label, f"X: {cur_x}  Y: {cur_y}",
                 f"{doc.width} \u00d7 {doc.height} px", f"Zoom {vp.percent}%",
                 f"Brush {self.brush_size}"]
        if self.hint:
            parts.append(self.hint)
        self.status_var.set("   |   ".join(parts))

    def update_report(self):
        self.report_title_var.set(self.report_title)
        self.report_var.set("\n".join(self.report_lines))

    def set_line_report(self, m, committed=False):
        self.report_title = "LINE (LAST)" if committed else "LINE"
        self.report_lines = [
            f"Start: ({m['x0']}, {m['y0']})",
            f"End:   ({m['x1']}, {m['y1']})",
            f"\u0394X: {m['abs_dx']} px",
            f"\u0394Y: {m['abs_dy']} px",
            f"Straight Distance: {m['distance']:.2f} px",
            f"Rasterized Pixels: {commas(m.get('rasterized', m['pixels']))}",
            f"Angle: {m['angle']:.2f}\u00b0",
        ]
        self.set_hint(f"Line | Start ({m['x0']},{m['y0']}) \u2192 End "
                      f"({m['x1']},{m['y1']}) | Length {m['distance']:.2f} px | "
                      f"Angle {m['angle']:.2f}\u00b0")
        self.update_report()

    def set_box_report(self, kind, a, b, c, d, width, height, extra=None,
                       committed=False):
        extra = extra or {}
        self.report_title = f"{kind.upper()}{' (LAST)' if committed else ''}"
        self.report_lines = [
            f"Top-Left:     ({a}, {b})",
            f"Bottom-Right: ({c}, {d})",
            f"Width:  {width} px",
            f"Height: {height} px",
        ] + [f"{k}: {v}" for k, v in extra.items()]
        tail = "".join(f" | {k} {v}" for k, v in extra.items())
        self.set_hint(f"{kind} | {width} \u00d7 {height} px{tail}")
        self.update_report()

    def set_pick_report(self, x, y, color):
        self.report_title = "PICKED COLOR"
        self.report_lines = [
            f"Pixel: ({x}, {y})",
            f"RGB:  {color[0]}, {color[1]}, {color[2]}",
            f"RGBA: {color[0]}, {color[1]}, {color[2]}, {color[3]}",
            f"HEX:  {rgb_hex(color)}",
            f"HEX8: {rgba_hex(color)}",
        ]
        self.set_hint(f"Picked {rgb_hex(color)} (alpha {color[3]}) at ({x}, {y})")
        self.update_report()

    def set_measurement(self, start, end):
        self.measurement = (start, end)
        m = line_metrics(start[0], start[1], end[0], end[1])
        self.report_title = "MEASUREMENT"
        self.report_lines = [
            f"Start: X {m['x0']}, Y {m['y0']}",
            f"End:   X {m['x1']}, Y {m['y1']}",
            f"Horizontal: {m['abs_dx']} px",
            f"Vertical:   {m['abs_dy']} px",
            f"Euclidean Distance: {m['distance']:.2f} px",
            f"Angle: {m['angle']:.2f}\u00b0",
            f"Pixels Crossed: {commas(m['pixels'])}",
        ]
        self.set_hint(f"Measure | ({m['x0']},{m['y0']}) \u2192 ({m['x1']},{m['y1']})"
                      f" | Length {m['distance']:.2f} px | "
                      f"Angle {m['angle']:.2f}\u00b0")
        self.update_report()
        self.request_render()

    def clear_measurement(self):
        if self.measurement is None:
            return
        self.measurement = None
        self.report_title = "INFO"
        self.report_lines = ["Measurement cleared."]
        self.update_report()
        self.set_hint("Measurement cleared")
        self.request_render()

    def report_selection(self, offset=None):
        rect = self.preview_selection or self.selection
        if not rect:
            self.report_title = "SELECTION"
            self.report_lines = ["No selection."]
            self.update_report()
            return
        x0, y0, x1, y1 = rect
        width = x1 - x0 + 1
        height = y1 - y0 + 1
        lines = [
            f"Top-Left:     ({x0}, {y0})",
            f"Bottom-Right: ({x1}, {y1})",
            f"Width:  {width} px",
            f"Height: {height} px",
            f"Area: {commas(width * height)} px\u00b2",
        ]
        if offset:
            lines.append(f"Moved: \u0394X {offset[0]}, \u0394Y {offset[1]}")
        self.report_title = "SELECTION"
        self.report_lines = lines
        self.update_report()
        self.set_hint(f"Selection | {width} \u00d7 {height} px | "
                      f"Area {commas(width * height)} px\u00b2")

    # ------------------------------------------------------------------ #
    # Tools / colours
    # ------------------------------------------------------------------ #

    def set_tool(self, key):
        if key not in self.tools:
            return
        if self.tool is not None and self.tool.name != key:
            self.tool.cancel()
        self.tool = self.tools[key]
        for name, button in self.tool_buttons.items():
            active = name == key
            button.configure(bg=C_ACCENT if active else C_BTN,
                             fg="#ffffff" if active else C_TEXT)
        self.canvas.configure(cursor=self.tool.cursor)
        self.tool_var.set(self.tool.label)
        self.set_hint(f"{self.tool.label}")
        self.request_render()

    def cancel_current(self):
        self.tool.cancel()
        if self.measurement is not None and self.tool.name != "measure":
            self.clear_measurement()
        self.clear_preview()
        self.set_hint("Cancelled")
        self.request_render()

    def set_color(self, color, add_recent=False):
        color = (int(color[0]), int(color[1]), int(color[2]),
                 int(color[3]) if len(color) > 3 else 255)
        self.color = color
        if add_recent:
            self.add_recent_color(color)
        self.refresh_color_widgets()

    def add_recent_color(self, color):
        if color in self.recent_colors:
            self.recent_colors.remove(color)
        self.recent_colors.insert(0, color)
        del self.recent_colors[MAX_RECENT_COLORS:]
        self.rebuild_recent_swatches()

    def rebuild_recent_swatches(self):
        for child in self.recent_frame.winfo_children():
            child.destroy()
        for index, color in enumerate(self.recent_colors):
            swatch = tk.Frame(self.recent_frame, bg=rgb_hex(color), width=24,
                              height=18, highlightthickness=1,
                              highlightbackground=C_BORDER, cursor="hand2")
            swatch.grid(row=index // 7, column=index % 7, padx=1, pady=1)
            swatch.bind("<Button-1>", lambda _e, c=color: self.set_color(c))
            Tooltip(swatch, f"{rgb_hex(color)} (alpha {color[3]})")

    def refresh_color_widgets(self):
        self._updating_color = True
        try:
            self.color_swatch.configure(bg=rgb_hex(self.color))
            self.hex_var.set(rgb_hex(self.color))
            self.alpha_var.set(self.color[3])
            self.color_text_var.set(
                f"RGB  {self.color[0]}, {self.color[1]}, {self.color[2]}\n"
                f"RGBA {self.color[0]}, {self.color[1]}, {self.color[2]}, "
                f"{self.color[3]}\nHEX  {rgb_hex(self.color)}")
        finally:
            self._updating_color = False

    def apply_hex(self, silent=False):
        if getattr(self, "_updating_color", False):
            return
        try:
            color = parse_hex_color(self.hex_var.get(), self.alpha_var.get())
        except ValueError as exc:
            if not silent:
                messagebox.showerror("Invalid colour", str(exc), parent=self.root)
            self.hex_var.set(rgb_hex(self.color))
            return
        self.set_color(color, add_recent=True)

    def apply_alpha(self):
        if getattr(self, "_updating_color", False):
            return
        try:
            alpha = clamp(int(self.alpha_var.get()), 0, 255)
        except (tk.TclError, ValueError):
            return
        self.set_color((self.color[0], self.color[1], self.color[2], alpha))

    def choose_color(self):
        chosen = colorchooser.askcolor(color=rgb_hex(self.color), parent=self.root,
                                       title="Choose drawing colour")
        if chosen and chosen[1]:
            try:
                color = parse_hex_color(chosen[1], self.color[3])
            except ValueError:
                return
            self.set_color(color, add_recent=True)

    def on_brush_change(self):
        size = self.brush_size
        self.brush_label_var.set(f"Brush: {size} px ({size} \u00d7 {size})")
        self.update_readouts()
        self._draw_hover()

    # ------------------------------------------------------------------ #
    # Zoom / pan
    # ------------------------------------------------------------------ #

    def after_view_change(self):
        vp = self.viewport
        vp.clamp_pan(self.canvas.winfo_width(), self.canvas.winfo_height(),
                     self.doc.width, self.doc.height)
        self.update_readouts()
        self.request_render()

    def zoom_step(self, direction, anchor=None):
        vp = self.viewport
        if vp.zoom in ZOOM_LEVELS:
            index = ZOOM_LEVELS.index(vp.zoom)
        else:
            index = min(range(len(ZOOM_LEVELS)),
                        key=lambda i: abs(ZOOM_LEVELS[i] - vp.zoom))
        index = clamp(index + direction, 0, len(ZOOM_LEVELS) - 1)
        self.set_zoom(ZOOM_LEVELS[index], anchor)

    def set_zoom(self, level, anchor=None):
        vp = self.viewport
        if anchor is None:
            anchor = (self.canvas.winfo_width() / 2.0,
                      self.canvas.winfo_height() / 2.0)
        # Zooming only changes the view: the image keeps its exact resolution.
        vp.set_zoom_at(int(level), anchor[0], anchor[1])
        self.set_hint(f"Zoom {vp.percent}%")
        self.after_view_change()

    def reset_zoom(self):
        self.viewport.zoom = 1
        self.center_view()
        self.set_hint("Zoom reset to 100%")

    def fit_to_window(self, cap=None):
        view_w = self.canvas.winfo_width()
        view_h = self.canvas.winfo_height()
        if view_w < 10 or view_h < 10:
            self._pending_fit = cap if cap is not None else 0
            return
        vp = self.viewport
        zoom = vp.fit_zoom(view_w, view_h, self.doc.width, self.doc.height)
        if cap:
            zoom = min(zoom, cap)
        vp.zoom = zoom
        vp.center_on_image(view_w, view_h, self.doc.width, self.doc.height)
        self.after_view_change()

    def center_view(self):
        self.viewport.center_on_image(self.canvas.winfo_width(),
                                      self.canvas.winfo_height(),
                                      self.doc.width, self.doc.height)
        self.after_view_change()

    def toggle_grid(self):
        self.show_grid.set(not self.show_grid.get())
        self.set_hint("Grid on" if self.show_grid.get() else "Grid off")
        self.request_render()

    def begin_pan(self, event):
        self._pan_state = (event.x, event.y, self.viewport.pan_x,
                           self.viewport.pan_y)
        self.canvas.configure(cursor="hand2")

    def drag_pan(self, event):
        if self._pan_state is None:
            return
        sx, sy, px, py = self._pan_state
        self.viewport.pan_x = px + (event.x - sx)
        self.viewport.pan_y = py + (event.y - sy)
        self.after_view_change()

    def end_pan(self):
        self._pan_state = None
        self.canvas.configure(cursor="hand2" if self.space_held
                              else self.tool.cursor)

    # ------------------------------------------------------------------ #
    # Mouse / keyboard events
    # ------------------------------------------------------------------ #

    def on_configure(self, _event=None):
        pending = self._pending_fit
        if pending is not None:
            self._pending_fit = None
            self.fit_to_window(pending or None)
            return
        if not self._did_initial_fit:
            self._did_initial_fit = True
            self.fit_to_window(cap=16)
            return
        self.request_render()

    def on_press(self, event):
        self.canvas.focus_set()
        if self.space_held or self.tool.name == "pan":
            self.begin_pan(event)
            return
        x, y = self.viewport.screen_to_pixel(event.x, event.y)
        self.hover_pixel = (x, y)
        if self.alt_held(event):
            self._alt_pick = True
            self._quick_pick(x, y)
        else:
            self._dragging = True
            self.tool.on_press(x, y, event)
        self.update_readouts()

    def on_drag(self, event):
        if self._pan_state is not None:
            self.drag_pan(event)
            return
        x, y = self.viewport.screen_to_pixel(event.x, event.y)
        self.hover_pixel = (x, y)
        if self._alt_pick:
            self._quick_pick(x, y)
        elif self._dragging:
            self.tool.on_drag(x, y, event)
        self.update_readouts()
        self._draw_hover()

    def on_release(self, event):
        if self._pan_state is not None:
            self.end_pan()
            return
        x, y = self.viewport.screen_to_pixel(event.x, event.y)
        if self._alt_pick:
            self._alt_pick = False
        elif self._dragging:
            self._dragging = False
            self.tool.on_release(x, y, event)
        self.update_readouts()

    def on_hover(self, event):
        x, y = self.viewport.screen_to_pixel(event.x, event.y)
        self.hover_pixel = (x, y)
        self.tool.on_hover(x, y, event)
        self.update_readouts()
        self._draw_hover()

    def on_leave(self, _event=None):
        self.hover_pixel = None
        self.canvas.delete("hover")
        self.update_readouts()

    def _quick_pick(self, x, y):
        color = self.doc.get_pixel(x, y)
        if color is not None:
            self.set_color(color, add_recent=True)
            self.set_pick_report(x, y, color)

    def on_wheel(self, event, delta=None):
        delta = event.delta if delta is None else delta
        if not delta:
            return
        if self.shift_held(event):
            self.viewport.pan_x += 60 if delta > 0 else -60
            self.after_view_change()
            return
        # Zoom towards the pointer so the pixel under the cursor stays put.
        self.zoom_step(1 if delta > 0 else -1, anchor=(event.x, event.y))

    def _keys_enabled(self):
        widget = self.root.focus_get()
        return not isinstance(widget, (tk.Entry, tk.Spinbox, tk.Text))

    def on_space_press(self, _event=None):
        if not self._keys_enabled():
            return
        if not self.space_held:
            self.space_held = True
            self.canvas.configure(cursor="hand2")
        return "break"

    def on_space_release(self, _event=None):
        self.space_held = False
        if self._pan_state is None:
            self.canvas.configure(cursor=self.tool.cursor)

    def on_keypress(self, event):
        if not self._keys_enabled():
            return
        if self.ctrl_held(event):
            return  # handled by the explicit Control-* bindings
        key = event.keysym.lower()
        if key in TOOL_KEYS:
            self.set_tool(TOOL_KEYS[key])
        elif key == "g":
            self.toggle_grid()
        elif key in ("plus", "equal", "kp_add"):
            self.zoom_step(1, anchor=self._pointer_anchor())
        elif key in ("minus", "kp_subtract"):
            self.zoom_step(-1, anchor=self._pointer_anchor())
        elif key == "0":
            self.reset_zoom()
        elif key in ("delete", "backspace"):
            self.delete_selection()
        elif key in ("bracketleft", "bracketright"):
            step = -1 if key == "bracketleft" else 1
            self.brush_var.set(clamp(self.brush_size + step, BRUSH_MIN, BRUSH_MAX))
            self.on_brush_change()
        elif key in "123456789" and len(key) == 1:
            self.brush_var.set(clamp(int(key), BRUSH_MIN, BRUSH_MAX))
            self.on_brush_change()
        elif key in ("left", "right", "up", "down"):
            dx = (-40 if key == "left" else 40) if key in ("left", "right") else 0
            dy = (-40 if key == "up" else 40) if key in ("up", "down") else 0
            self.viewport.pan_x -= dx
            self.viewport.pan_y -= dy
            self.after_view_change()

    def _pointer_anchor(self):
        """Anchor keyboard zoom on the pointer when it is over the canvas."""
        if self.hover_pixel is None:
            return None
        sx, sy = self.viewport.pixel_to_screen(*self.hover_pixel)
        return (sx + self.viewport.zoom / 2.0, sy + self.viewport.zoom / 2.0)

    # ------------------------------------------------------------------ #
    # File operations
    # ------------------------------------------------------------------ #

    def confirm_discard(self, action: str) -> bool:
        """Ask before throwing away unsaved pixels."""
        if not self.doc.modified:
            return True
        return messagebox.askyesno(
            "Unsaved changes",
            f"This drawing has unsaved changes.\n\nDiscard them and {action}?",
            parent=self.root)

    def load_document(self, document: PixelDocument, fit=True, cap=16):
        self.tool.cancel()
        self.doc = document
        self.selection = None
        self.clear_preview()
        self.measurement = None
        if fit:
            self.fit_to_window(cap=cap)
        else:
            self.center_view()
        self.update_title()
        self.update_readouts()
        self.update_report()
        self.request_render()

    def new_canvas(self, startup=False):
        if not startup and not self.confirm_discard("start a new canvas"):
            return False
        result = NewCanvasDialog(self.root, self.doc.width, self.doc.height,
                                 startup=startup).show()
        if result is None:
            return False
        width, height = result
        self.load_document(PixelDocument(width, height))
        self.report_title = "CANVAS CREATED"
        self.report_lines = [f"Width: {width} px", f"Height: {height} px",
                             f"Total Pixels: {commas(width * height)}"]
        self.update_report()
        self.set_hint(f"New canvas {width} \u00d7 {height} px "
                      f"({commas(width * height)} pixels)")
        return True

    def open_png(self):
        if not self.confirm_discard("open another image"):
            return
        path = filedialog.askopenfilename(
            parent=self.root, title="Open PNG",
            filetypes=[("PNG images", "*.png"),
                       ("Image files", "*.png *.gif *.bmp *.jpg *.jpeg *.webp"),
                       ("All files", "*.*")])
        if not path:
            self.set_hint("Open cancelled")
            return
        try:
            with Image.open(path) as handle:
                handle.load()
                image = handle.convert("RGBA")   # keeps existing transparency
        except (OSError, ValueError) as exc:
            messagebox.showerror("Could not open image",
                                 f"{os.path.basename(path)} could not be read.\n\n{exc}",
                                 parent=self.root)
            return
        width, height = image.size
        if width > MAX_OPEN_DIM or height > MAX_OPEN_DIM:
            messagebox.showerror(
                "Image too large",
                f"This image is {width} \u00d7 {height} px. The editor accepts "
                f"up to {MAX_OPEN_DIM} px per side.", parent=self.root)
            return
        if width * height > MAX_TOTAL_PIXELS:
            proceed = messagebox.askyesno(
                "Large image",
                f"{width} \u00d7 {height} px is {commas(width * height)} pixels, "
                "which may make editing slow.\n\nOpen it anyway?",
                parent=self.root)
            if not proceed:
                return
        document = PixelDocument(width, height, image)
        document.export_path = path
        self.load_document(document)
        self.doc.modified = False
        self.report_title = "LOADED IMAGE"
        self.report_lines = [os.path.basename(path), f"Width: {width} px",
                             f"Height: {height} px",
                             f"Total Pixels: {commas(width * height)}"]
        self.update_report()
        self.update_title()
        self.set_hint(f"Loaded {os.path.basename(path)} at native "
                      f"{width} \u00d7 {height} px")

    def open_project(self):
        if not self.confirm_discard("open a project"):
            return
        path = filedialog.askopenfilename(
            parent=self.root, title="Open Project",
            filetypes=[("Pixel Editor project", "*" + PROJECT_EXT),
                       ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            blob = base64.b64decode(data["image_png_base64"])
            with Image.open(io.BytesIO(blob)) as handle:
                handle.load()
                image = handle.convert("RGBA")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            messagebox.showerror("Could not open project",
                                 f"{os.path.basename(path)} is not a valid "
                                 f"project file.\n\n{exc}", parent=self.root)
            return
        document = PixelDocument(image.size[0], image.size[1], image)
        document.project_path = path
        self.load_document(document)
        self.doc.modified = False
        # Restore palette and zoom when present, ignoring anything malformed.
        try:
            self.set_color(parse_hex_color(data.get("color", "#FFFFFFFF")))
        except ValueError:
            pass
        recent = []
        for item in data.get("recent_colors", []):
            try:
                recent.append(parse_hex_color(item))
            except (ValueError, TypeError):
                continue
        if recent:
            self.recent_colors = recent[:MAX_RECENT_COLORS]
            self.rebuild_recent_swatches()
        zoom = data.get("zoom")
        if isinstance(zoom, int) and zoom in ZOOM_LEVELS:
            self.set_zoom(zoom)
        self.update_title()
        self.set_hint(f"Opened project {os.path.basename(path)}")

    def save_project(self):
        path = self.doc.project_path
        if not path:
            path = filedialog.asksaveasfilename(
                parent=self.root, title="Save Project",
                defaultextension=PROJECT_EXT,
                initialfile="untitled" + PROJECT_EXT,
                filetypes=[("Pixel Editor project", "*" + PROJECT_EXT)])
            if not path:
                self.set_hint("Save cancelled")
                return
        buffer = io.BytesIO()
        self.doc.image.save(buffer, "PNG")
        data = {
            "format": "pixel_editor_project",
            "version": 1,
            "width": self.doc.width,
            "height": self.doc.height,
            "zoom": self.viewport.zoom,
            "color": rgba_hex(self.color),
            "recent_colors": [rgba_hex(c) for c in self.recent_colors],
            "image_png_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(data, handle)
        except OSError as exc:
            messagebox.showerror("Could not save project", str(exc),
                                 parent=self.root)
            return
        self.doc.project_path = path
        self.doc.modified = False
        self.update_title()
        self.set_hint(f"Project saved to {os.path.basename(path)}")

    def export_png(self):
        self.tool.cancel()
        options = ExportDialog(self.root, self.doc.width, self.doc.height,
                               self.export_bg_hex).show()
        if not options:
            self.set_hint("Export cancelled")
            return
        if options["background"]:
            self.export_bg_hex = rgb_hex(options["background"])
        suggested = "pixel-art.png"
        if self.doc.export_path:
            suggested = os.path.basename(self.doc.export_path)
        elif self.doc.project_path:
            suggested = os.path.splitext(os.path.basename(
                self.doc.project_path))[0] + ".png"
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Export PNG", defaultextension=".png",
            initialfile=suggested, filetypes=[("PNG image", "*.png")])
        if not path:
            self.set_hint("Export cancelled")
            return
        grid_color = (110, 110, 116, 255) if options["grid"] else None
        try:
            # Built from the artwork alone: no checkerboard, no preview,
            # no measurement overlay, and no grid unless explicitly asked for.
            image = self.doc.build_export_image(options["scale"],
                                                options["background"], grid_color)
            image.save(path, "PNG")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Export failed", str(exc), parent=self.root)
            return
        self.doc.export_path = path
        self.update_title()
        self.report_title = "EXPORTED PNG"
        self.report_lines = [
            os.path.basename(path),
            f"Canvas: {self.doc.width} \u00d7 {self.doc.height} px",
            f"Scale: {options['scale']}x",
            f"File: {image.size[0]} \u00d7 {image.size[1]} px",
            "Background: " + ("solid" if options["background"] else "transparent"),
        ]
        self.update_report()
        self.set_hint(f"Exported {image.size[0]} \u00d7 {image.size[1]} px to "
                      f"{os.path.basename(path)}")

    # ------------------------------------------------------------------ #
    # Edit operations
    # ------------------------------------------------------------------ #

    def undo(self):
        self.tool.cancel()
        label = self.doc.undo()
        if label is None:
            self.set_hint("Nothing to undo")
            return
        self.selection = None
        self.clear_preview()
        self.update_title()
        self.set_hint(f"Undo: {label}")
        self.request_render()

    def redo(self):
        self.tool.cancel()
        label = self.doc.redo()
        if label is None:
            self.set_hint("Nothing to redo")
            return
        self.selection = None
        self.clear_preview()
        self.update_title()
        self.set_hint(f"Redo: {label}")
        self.request_render()

    def clear_canvas(self):
        self.doc.begin_operation("Clear Canvas")
        self.doc.image.paste(TRANSPARENT, (0, 0, self.doc.width, self.doc.height))
        self.set_hint("Canvas cleared")
        self.after_edit()

    def delete_selection(self):
        if not self.selection:
            self.set_hint("Nothing selected")
            return
        self.doc.begin_operation("Delete Selection")
        self.doc.write_pixels(rect_fill_pixels(*self.selection), TRANSPARENT)
        self.set_hint("Selection cleared")
        self.after_edit()

    def copy(self):
        if not self.selection:
            self.set_hint("Nothing selected to copy")
            return
        x0, y0, x1, y1 = self.selection
        self.clipboard = self.doc.image.crop((x0, y0, x1 + 1, y1 + 1)).copy()
        self.set_hint(f"Copied {x1 - x0 + 1} \u00d7 {y1 - y0 + 1} px")

    def cut(self):
        if not self.selection:
            self.set_hint("Nothing selected to cut")
            return
        self.copy()
        self.delete_selection()
        self.set_hint("Cut selection")

    def paste(self):
        if self.clipboard is None:
            self.set_hint("Clipboard is empty")
            return
        if self.selection:
            x, y = self.selection[0], self.selection[1]
        elif self.hover_pixel and self.doc.in_bounds(*self.hover_pixel):
            x, y = self.hover_pixel
        else:
            x, y = 0, 0
        self.doc.begin_operation("Paste")
        self.composite_block(self.clipboard, x, y)
        self.selection = (x, y, x + self.clipboard.size[0] - 1,
                          y + self.clipboard.size[1] - 1)
        self.report_selection()
        self.set_hint(f"Pasted {self.clipboard.size[0]} \u00d7 "
                      f"{self.clipboard.size[1]} px at ({x}, {y})")
        self.after_edit()

    def select_all(self):
        self.selection = (0, 0, self.doc.width - 1, self.doc.height - 1)
        self.set_tool("select")
        self.report_selection()
        self.request_render()

    def deselect(self):
        self.selection = None
        self.preview_selection = None
        self.report_selection()
        self.request_render()

    def resize_dialog(self, mode="canvas"):
        result = ResizeDialog(self.root, self.doc.width, self.doc.height,
                              mode).show()
        if not result:
            return
        new_w, new_h = result["width"], result["height"]
        if (new_w, new_h) == self.doc.size and result["mode"] == "canvas":
            self.set_hint("Canvas size unchanged")
            return
        self.doc.begin_operation("Resize Canvas" if result["mode"] == "canvas"
                                 else "Resize Image")
        if result["mode"] == "image":
            # Deliberate artwork rescale - nearest neighbour keeps hard pixels.
            new_image = self.doc.image.resize((new_w, new_h), NEAREST)
        else:
            new_image = Image.new("RGBA", (new_w, new_h), TRANSPARENT)
            if result["anchor"] == "center":
                offset = ((new_w - self.doc.width) // 2,
                          (new_h - self.doc.height) // 2)
            else:
                offset = (0, 0)
            new_image.paste(self.doc.image, offset)   # paste clips for us
        self.doc.replace_image(new_image)
        self.selection = None
        self.clear_preview()
        self.fit_to_window(cap=16)
        self.set_hint(f"{'Image' if result['mode'] == 'image' else 'Canvas'} "
                      f"resized to {new_w} \u00d7 {new_h} px "
                      f"({commas(new_w * new_h)} pixels)")
        self.after_edit()

    # ------------------------------------------------------------------ #
    # Help / shutdown
    # ------------------------------------------------------------------ #

    def show_shortcuts(self):
        messagebox.showinfo("Keyboard Shortcuts", SHORTCUT_TEXT, parent=self.root)

    def show_self_tests(self):
        results = run_self_tests()
        failures = [line for ok, line in results if not ok]
        body = "\n".join(line for _ok, line in results)
        if failures:
            messagebox.showerror("Self-tests", body, parent=self.root)
        else:
            messagebox.showinfo("Self-tests", body, parent=self.root)

    def show_about(self):
        messagebox.showinfo(
            "About " + APP_NAME,
            f"{APP_NAME}\n\nA small pixel-art editor built with Tkinter and "
            "Pillow.\nThe artwork is always a true pixel grid: zoom changes "
            "only the view,\nnever the image resolution.",
            parent=self.root)

    def on_close(self):
        if self.doc.modified:
            keep = messagebox.askyesno(
                "Unsaved changes",
                "This drawing has unsaved changes.\n\nQuit anyway?",
                parent=self.root)
            if not keep:
                return
        self.root.destroy()


SHORTCUT_TEXT = """Tools
  P Pencil        E Eraser       L Line
  R Rectangle     C Circle       F Fill
  I Eyedropper    M Measure      S Select
  H Pan           1-9 Brush size  [ ] Brush -/+

View
  +/-  Zoom in / out (towards the cursor)
  0    Reset zoom to 100%
  Ctrl+0 Fit canvas in window
  G    Toggle pixel grid
  Mouse wheel  Zoom at pointer
  Middle drag / Space+drag  Pan
  Arrow keys   Pan the view

Edit
  Ctrl+Z Undo      Ctrl+Y Redo
  Ctrl+A Select all  Ctrl+C/X/V Copy/Cut/Paste
  Delete Clear selection
  Alt+click  Pick colour with any tool
  Shift  Snap line to 45 deg, square / circle
  Escape Cancel shape or clear measurement

File
  Ctrl+N New canvas    Ctrl+O Open PNG
  Ctrl+S Export PNG    Ctrl+Shift+S Save project"""


# --------------------------------------------------------------------------- #
# Accuracy self-tests (no GUI required: `python pixel_editor.py --selftest`)
# --------------------------------------------------------------------------- #

def run_self_tests():
    """Verify the pixel-accuracy guarantees. Returns [(ok, message), ...]."""
    results = []

    def check(name, condition, detail=""):
        ok = bool(condition)
        text = f"[{'PASS' if ok else 'FAIL'}] {name}"
        if detail:
            text += f" - {detail}"
        results.append((ok, text))
        return ok

    # Test 1: exact resolution and a native (1x) export that matches it. ---- #
    doc = PixelDocument(32, 32)
    doc.write_pixels([(0, 0), (31, 31), (5, 9)], (10, 20, 30, 255))
    native = doc.build_export_image(1)
    check("Test 1: 32 x 32 canvas has 1,024 pixels",
          doc.total_pixels == 1024, f"total={commas(doc.total_pixels)}")
    check("Test 1: native export is exactly 32 x 32",
          native.size == (32, 32), f"export={native.size}")
    check("Test 1: native export is byte-identical to the artwork "
          "(no grid / checkerboard / overlay)",
          native.tobytes() == doc.image.tobytes())

    # Test 2: one pixel at 1600% zoom changes exactly one logical pixel. ---- #
    doc2 = PixelDocument(32, 32)
    vp = Viewport(16)
    vp.pan_x, vp.pan_y = -37.0, 11.5     # awkward pan on purpose
    target = (23, 7)
    tx, ty = vp.pixel_to_screen(*target)
    px, py = vp.screen_to_pixel(tx + 9, ty + 3)    # click inside that cell
    doc2.write_pixels(expand_brush([(px, py)], 1), (255, 0, 0, 255))
    opaque = sum(1 for a in doc2.image.getchannel("A").tobytes() if a > 0)
    check("Test 2: click at 1600% maps back to the same pixel",
          (px, py) == target, f"got=({px}, {py}) expected={target}")
    check("Test 2: brush size 1 changes exactly one logical pixel",
          opaque == 1 and doc2.get_pixel(*target) == (255, 0, 0, 255),
          f"changed={opaque}")

    # Test 3: line (0,0) -> (10,0). ----------------------------------------- #
    m3 = line_metrics(0, 0, 10, 0)
    raster3 = bresenham_line(0, 0, 10, 0)
    check("Test 3: line (0,0)->(10,0) dX=10, dY=0",
          m3["dx"] == 10 and m3["dy"] == 0)
    check("Test 3: Euclidean distance = 10",
          abs(m3["distance"] - 10.0) < 1e-9, f"{m3['distance']:.4f}")
    check("Test 3: angle = 0 deg", abs(m3["angle"]) < 1e-9, f"{m3['angle']:.4f}")
    check("Test 3: rasterized pixels = 11",
          len(raster3) == 11 == m3["pixels"], f"{len(raster3)}")

    # Test 4: measurement (0,0) -> (3,4) is 5 px. --------------------------- #
    m4 = line_metrics(0, 0, 3, 4)
    check("Test 4: measure (0,0)->(3,4) distance = 5.00 px",
          abs(m4["distance"] - 5.0) < 1e-9, f"{m4['distance']:.4f}")
    check("Test 4: pixels crossed = 5", m4["pixels"] == 5, f"{m4['pixels']}")

    # Test 5: 64 x 64 exported at 8x is 512 x 512 with sharp square pixels. - #
    doc5 = PixelDocument(64, 64)
    for i in range(64):
        doc5.set_pixel(i, i, (0, 128, 255, 255))
    doc5.set_pixel(0, 63, (255, 255, 0, 255))
    scaled = doc5.build_export_image(8)
    sharp = True
    sample = scaled.load()
    for (cx, cy) in ((0, 0), (5, 5), (0, 63), (63, 63), (12, 40)):
        expected = doc5.get_pixel(cx, cy)
        for ox in (0, 3, 7):
            for oy in (0, 3, 7):
                if sample[cx * 8 + ox, cy * 8 + oy] != expected:
                    sharp = False
    check("Test 5: 64 x 64 exported at 8x is 512 x 512",
          scaled.size == (512, 512), f"export={scaled.size}")
    check("Test 5: every exported pixel is a uniform 8 x 8 block "
          "(nearest neighbour, no blur)", sharp)

    # Coordinate conversion round trip at several zooms and pans. ----------- #
    round_trip = True
    for zoom in ZOOM_LEVELS:
        view = Viewport(zoom)
        for pan in (0.0, -123.4, 57.6, 999.0):
            view.pan_x = view.pan_y = pan
            for pixel in ((0, 0), (1, 3), (17, 42), (63, 63)):
                sx, sy = view.pixel_to_screen(*pixel)
                back = view.screen_to_pixel(sx + zoom / 2.0, sy + zoom / 2.0)
                if back != pixel:
                    round_trip = False
    check("Coordinates: pixel -> screen -> pixel is stable at every zoom/pan",
          round_trip)

    # Zooming towards the cursor keeps that image point under the cursor. --- #
    view = Viewport(4)
    view.pan_x, view.pan_y = 30.0, -12.0
    anchor = (211.0, 148.0)
    before = view.screen_to_pixel_float(*anchor)
    view.set_zoom_at(16, *anchor)
    after = view.screen_to_pixel_float(*anchor)
    check("Zoom: the image point under the cursor stays put",
          abs(before[0] - after[0]) < 0.05 and abs(before[1] - after[1]) < 0.05,
          f"{before} -> {after}")

    # Shape measurement conventions (inclusive pixel counts). -------------- #
    a, b, c, d = normalize_box(4, 10, 28, 21)
    width, height = c - a + 1, d - b + 1
    check("Rectangle: 25 x 12 px covers an area of 300 px^2",
          (width, height, width * height) == (25, 12, 300),
          f"{width}x{height}={width * height}")
    check("Rectangle: filled pixel count equals width * height",
          len(rect_fill_pixels(a, b, c, d)) == 300)
    check("Rectangle: outline pixel count = perimeter",
          len(rect_outline_pixels(a, b, c, d)) == 2 * width + 2 * height - 4,
          f"{len(rect_outline_pixels(a, b, c, d))}")

    # Ellipse: inside the bounding box, symmetric, outline is a subset. ----- #
    fill = ellipse_fill_pixels(0, 0, 29, 29)
    outline = set(ellipse_outline_pixels(0, 0, 29, 29))
    in_box = all(0 <= x <= 29 and 0 <= y <= 29 for (x, y) in fill)
    symmetric = all((29 - x, y) in set(fill) and (x, 29 - y) in set(fill)
                    for (x, y) in fill)
    check("Ellipse: 30 x 30 fill stays inside its bounding box", in_box)
    check("Ellipse: fill is symmetric on both axes", symmetric)
    check("Ellipse: outline is a subset of the fill and touches all 4 sides",
          outline.issubset(set(fill))
          and any(y == 0 for (_x, y) in outline)
          and any(y == 29 for (_x, y) in outline)
          and any(x == 0 for (x, _y) in outline)
          and any(x == 29 for (x, _y) in outline))

    # Flood fill: iterative, 4-connected. ---------------------------------- #
    doc6 = PixelDocument(10, 10)
    doc6.write_pixels([(5, y) for y in range(10)], (0, 0, 0, 255))  # divider
    filled = flood_fill_image(doc6.image, 0, 0, (255, 0, 0, 255))
    check("Fill: 4-connected fill stops at the divider",
          filled == 50, f"filled={filled}")
    check("Fill: the far side is untouched",
          doc6.get_pixel(9, 9) == TRANSPARENT)

    # Undo history: one stroke is one undo step. ---------------------------- #
    doc7 = PixelDocument(8, 8)
    doc7.begin_operation("Pencil")
    for point in bresenham_line(0, 0, 7, 7):
        doc7.set_pixel(point[0], point[1], (1, 2, 3, 255))
    painted = sum(1 for a in doc7.image.getchannel("A").tobytes() if a > 0)
    doc7.undo()
    after_undo = sum(1 for a in doc7.image.getchannel("A").tobytes() if a > 0)
    doc7.redo()
    after_redo = sum(1 for a in doc7.image.getchannel("A").tobytes() if a > 0)
    check("Undo: a whole stroke is a single undo step",
          painted == 8 and after_undo == 0 and after_redo == 8,
          f"painted={painted} undo={after_undo} redo={after_redo}")

    # Undo across a canvas resize restores the original resolution. -------- #
    doc8 = PixelDocument(16, 16)
    doc8.begin_operation("Resize Canvas")
    doc8.replace_image(doc8.image.resize((64, 48), NEAREST))
    resized = doc8.size
    doc8.undo()
    check("Undo: restores the previous canvas resolution",
          resized == (64, 48) and doc8.size == (16, 16),
          f"{resized} -> {doc8.size}")

    # Pixel-perfect pencil drops redundant corner pixels. ------------------ #
    check("Pixel-perfect: L-shaped corner is detected",
          is_pixel_perfect_corner((0, 0), (1, 0), (1, 1))
          and not is_pixel_perfect_corner((0, 0), (1, 0), (2, 0)))

    # Hex parsing and error handling. --------------------------------------- #
    bad_inputs = ["", "#12", "#GGGGGG", "12345", "hello"]
    rejected = 0
    for text in bad_inputs:
        try:
            parse_hex_color(text)
        except ValueError:
            rejected += 1
    check("Colours: valid hex parses, invalid hex is rejected",
          parse_hex_color("#FF8000") == (255, 128, 0, 255)
          and parse_hex_color("#0f0") == (0, 255, 0, 255)
          and parse_hex_color("#11223344") == (17, 34, 51, 68)
          and rejected == len(bad_inputs),
          f"rejected {rejected}/{len(bad_inputs)}")

    bad_dims = 0
    for text in ["0", "-8", str(MAX_DIM + 1), "abc", "", "3.5"]:
        try:
            parse_dimension(text, "Width")
        except ValueError:
            bad_dims += 1
    check("Dimensions: zero, negative, oversized and non-numeric are rejected",
          bad_dims == 6 and parse_dimension("128", "Width") == 128,
          f"rejected {bad_dims}/6")

    # Export options. ------------------------------------------------------- #
    doc9 = PixelDocument(10, 6)
    doc9.set_pixel(0, 0, (255, 255, 255, 255))
    solid = doc9.build_export_image(1, background=(0, 0, 0, 255))
    with_grid = doc9.build_export_image(4, background=None,
                                        grid_color=(110, 110, 116, 255))
    no_grid = doc9.build_export_image(4)
    check("Export: solid background makes every pixel opaque",
          solid.size == (10, 6)
          and all(a == 255 for a in solid.getchannel("A").tobytes()))
    check("Export: transparent background keeps alpha",
          doc9.build_export_image(1).getpixel((9, 5))[3] == 0)
    check("Export: grid is only burned in when requested",
          with_grid.tobytes() != no_grid.tobytes()
          and no_grid.size == (40, 24)
          and no_grid.tobytes() == doc9.image.resize((40, 24), NEAREST).tobytes())

    # --------------------------------------------------------------------- #
    # Packaging checks: these exercise the parts of the environment that a
    # frozen build (PyInstaller) can get wrong - Pillow's PNG codec, the
    # project file format, and the Tk/Pillow bridge used to draw the canvas.
    # --------------------------------------------------------------------- #
    temp_dir = tempfile.mkdtemp(prefix="pixel_editor_selftest_")
    try:
        # A real PNG written to and read back from disk.
        doc10 = PixelDocument(24, 18)
        doc10.write_pixels([(0, 0), (23, 17), (7, 3)], (200, 40, 90, 255))
        doc10.set_pixel(1, 1, (0, 0, 0, 0))
        png_path = os.path.join(temp_dir, "roundtrip.png")
        doc10.build_export_image(1).save(png_path, "PNG")
        with Image.open(png_path) as handle:
            handle.load()
            reloaded = handle.convert("RGBA")
        check("Files: PNG saved to disk reloads with identical pixels and size",
              reloaded.size == (24, 18)
              and reloaded.tobytes() == doc10.image.tobytes()
              and reloaded.getpixel((1, 1))[3] == 0,
              f"{reloaded.size} {os.path.getsize(png_path)} bytes")

        # The .pxproj container: JSON with a base64-encoded PNG payload.
        buffer = io.BytesIO()
        doc10.image.save(buffer, "PNG")
        project = {
            "format": "pixel_editor_project",
            "version": 1,
            "width": doc10.width,
            "height": doc10.height,
            "zoom": 8,
            "color": rgba_hex((1, 2, 3, 4)),
            "recent_colors": [rgba_hex((5, 6, 7, 8))],
            "image_png_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }
        project_path = os.path.join(temp_dir, "roundtrip" + PROJECT_EXT)
        with open(project_path, "w", encoding="utf-8") as handle:
            json.dump(project, handle)
        with open(project_path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        payload = base64.b64decode(loaded["image_png_base64"])
        with Image.open(io.BytesIO(payload)) as handle:
            handle.load()
            restored = handle.convert("RGBA")
        check("Files: project round trip keeps the canvas size and every pixel",
              loaded["format"] == "pixel_editor_project"
              and (loaded["width"], loaded["height"]) == (24, 18)
              and restored.tobytes() == doc10.image.tobytes()
              and parse_hex_color(loaded["color"]) == (1, 2, 3, 4),
              f"{os.path.getsize(project_path)} bytes")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    check("Dialogs: file, colour and message dialogs are importable",
          all(callable(func) for func in (filedialog.askopenfilename,
                                          filedialog.asksaveasfilename,
                                          colorchooser.askcolor,
                                          messagebox.showerror)))

    results.extend(run_gui_smoke_tests())
    return results


def check_imagetk_bridge(root, check):
    """Verify the Tk/Pillow bridge that render() relies on to show the canvas."""
    preview = PixelDocument(8, 6)
    preview.set_pixel(2, 2, (255, 255, 255, 255))
    scaled = preview.image.resize((8 * 16, 6 * 16), NEAREST)
    photo = ImageTk.PhotoImage(scaled, master=root)
    ok = (photo.width(), photo.height()) == (128, 96)
    del photo
    check("Display: Tk and PIL.ImageTk render a scaled canvas bitmap", ok,
          "128 x 96 PhotoImage")


def run_gui_smoke_tests():
    """Drive the real tools in an invisible window: drawing, zooming, export.

    Everything shares a single Tk root. When an editor is already running
    (Help > Run Self-Tests) only the display check runs, because building a
    second application instance inside a live one is not safe.
    """
    results = []

    def check(name, condition, detail=""):
        ok = bool(condition)
        text = f"[{'PASS' if ok else 'FAIL'}] {name}"
        if detail:
            text += f" - {detail}"
        results.append((ok, text))
        return ok

    if ImageTk is None:
        check("Display: PIL.ImageTk is available", False, "ImageTk failed to import")
        return results

    live_root = getattr(tk, "_default_root", None)
    if live_root is not None:
        try:
            check_imagetk_bridge(live_root, check)
        except tk.TclError as exc:
            check("Display: Tk and PIL.ImageTk render a scaled canvas bitmap",
                  False, f"Tcl/Tk error: {exc}")
        results.append((True, "[SKIP] GUI smoke tests (an editor is already open; "
                              "run --selftest from a command line to include them)"))
        return results

    temp_dir = tempfile.mkdtemp(prefix="pixel_editor_gui_")
    root = None
    try:
        root = tk.Tk()
        root.attributes("-alpha", 0.0)    # realised (so widgets get real sizes)
        root.geometry("1000x760+40+40")   # but fully transparent, so invisible
        check_imagetk_bridge(root, check)
        app = PixelEditorApp(root, 64, 64)
        root.update()
        app.set_zoom(8)
        app.center_view()
        root.update()
        check("GUI: editor window builds with a 64 x 64 canvas",
              app.doc.size == (64, 64) and app.canvas.winfo_width() > 200,
              f"viewport={app.canvas.winfo_width()}x{app.canvas.winfo_height()}")

        class FakeEvent:
            """Stand-in for a Tk mouse event (x, y and modifier state)."""

            def __init__(self, x, y, state=0):
                self.x, self.y, self.state, self.delta = x, y, state, 0

        def event_at(px, py):
            sx, sy = app.viewport.pixel_to_screen(px, py)
            half = app.viewport.zoom // 2
            return FakeEvent(sx + half, sy + half)

        def use_tool(tool, points):
            app.set_tool(tool)
            app.on_press(event_at(*points[0]))
            for point in points[1:]:
                app.on_drag(event_at(*point))
            app.on_release(event_at(*points[-1]))
            root.update()

        app.set_color((255, 80, 0, 255))
        use_tool("pencil", [(2, 2), (10, 2), (10, 10)])
        check("GUI: pencil paints the dragged pixels",
              app.doc.get_pixel(2, 2) == (255, 80, 0, 255)
              and app.doc.get_pixel(6, 2) == (255, 80, 0, 255)
              and app.doc.get_pixel(10, 10) == (255, 80, 0, 255))

        use_tool("line", [(0, 20), (30, 44)])
        midpoint = bresenham_line(0, 20, 30, 44)[15]
        check("GUI: line tool commits a rasterized line",
              app.doc.get_pixel(*midpoint)[3] == 255, f"mid={midpoint}")

        app.shape_fill.set(False)
        use_tool("ellipse", [(34, 4), (60, 30)])
        ring = [p for p in ellipse_outline_pixels(34, 4, 60, 30)
                if app.doc.get_pixel(*p)[3] == 255]
        check("GUI: ellipse tool commits an outline",
              len(ring) == len(ellipse_outline_pixels(34, 4, 60, 30)),
              f"{len(ring)} pixels")

        before_size = app.doc.size
        app.zoom_step(1)
        app.zoom_step(1)
        zoomed = app.viewport.percent
        app.zoom_step(-1)
        root.update()
        check("GUI: zooming in and out never changes the resolution",
              app.doc.size == before_size and zoomed > 800
              and app.viewport.percent < zoomed,
              f"{zoomed}% -> {app.viewport.percent}%, canvas {app.doc.size}")

        undo_before = app.doc.can_undo()
        app.undo()
        redo_ok = app.doc.can_redo()
        app.redo()
        check("GUI: undo and redo work on committed shapes",
              undo_before and redo_ok and app.doc.get_pixel(*midpoint)[3] == 255)

        native_path = os.path.join(temp_dir, "native.png")
        scaled_path = os.path.join(temp_dir, "scaled8x.png")
        app.doc.build_export_image(1).save(native_path, "PNG")
        app.doc.build_export_image(8).save(scaled_path, "PNG")
        with Image.open(native_path) as handle:
            handle.load()
            native = handle.convert("RGBA")
        with Image.open(scaled_path) as handle:
            handle.load()
            scaled_size = handle.size
        check("GUI: export writes a native PNG identical to the canvas",
              native.size == (64, 64) and native.tobytes() == app.doc.image.tobytes(),
              f"{native.size}")
        check("GUI: export at 8x writes 512 x 512", scaled_size == (512, 512),
              f"{scaled_size}")

        # Reopen the exported PNG the way open_png does.
        with Image.open(native_path) as handle:
            handle.load()
            reopened = handle.convert("RGBA")
        document = PixelDocument(reopened.size[0], reopened.size[1], reopened)
        app.load_document(document)
        root.update()
        check("GUI: the exported PNG reopens at its native size with its pixels",
              app.doc.size == (64, 64)
              and app.doc.image.tobytes() == native.tobytes())
    except tk.TclError as exc:
        check("GUI: editor window can be created", False, f"Tcl/Tk error: {exc}")
    finally:
        if root is not None:
            try:
                root.destroy()
            except tk.TclError:
                pass
        shutil.rmtree(temp_dir, ignore_errors=True)
    return results


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def report_startup_text(title: str, text: str) -> None:
    """Write a console report, or show a dialog when there is no console.

    A PyInstaller --windowed build has no stdout, so `--selftest` and startup
    errors would otherwise be invisible when run from PixelEditor.exe.
    """
    if sys.stdout is not None:
        print(text)
        return
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(title, text, parent=root)
    root.destroy()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv or "--self-test" in argv:
        results = run_self_tests()
        failed = sum(1 for ok, _line in results if not ok)
        report = "\n".join(line for _ok, line in results)
        report += f"\n\n{len(results) - failed} passed, {failed} failed"
        report_startup_text(f"{APP_NAME} self-tests", report)
        return 1 if failed else 0

    if ImageTk is None:
        report_startup_text(
            APP_NAME,
            "Pillow's ImageTk module is not available.\n"
            "Install/upgrade Pillow with Tk support:  pip install --upgrade pillow")
        return 2

    root = tk.Tk()
    app = PixelEditorApp(root, 32, 32)
    # Draw the main window first so the modal dialog can grab input safely.
    root.update()
    root.after(60, lambda: app.new_canvas(startup=True))
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())





