"""Screenshot -> Board."""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from ..core.board import UNKNOWN, Board
import cv2

from .cells import S, cells_from_image, classify_cells, resample_board
from .grid import Grid, channel_max, detect_grid, line_steps

MAX_UNKNOWN_FRACTION = 0.35   # above this the reading is rejected as unreliable
MIN_CELL_PX = 14              # smaller cells are refused: digits are too few pixels to read safely
CELL_MEAN_CHANGE = 4.0        # mean colour difference (0-255) that marks a cell as changed
CELL_PIXEL_CHANGE = 40        # ... or this difference on more than CELL_PIXEL_FRACTION of it
CELL_PIXEL_FRACTION = 0.02


@dataclass
class Recognition:
    ok: bool
    message: str
    grid: Optional[Grid] = None
    states: Optional[np.ndarray] = None       # [rows, cols] Board cell states
    confidence: Optional[np.ndarray] = None
    unknown_count: int = 0
    elapsed_ms: float = 0.0
    reused_grid: bool = False
    clipped: bool = False                     # grid reaches the edge of the frame
    skipped: bool = False                     # no cell changed since the last frame: nothing re-read
    changed: Optional[np.ndarray] = None      # [rows, cols] cells whose pixels changed
    kept: int = 0                             # unchanged cells whose earlier reading was kept
    features: Optional[dict] = field(default=None, repr=False)   # see cells.classify_cells

    def to_board(self, total_mines: Optional[int] = None) -> Board:
        if not self.ok or self.states is None:
            raise ValueError("no board was recognised")
        rows, cols = self.states.shape
        board = Board(rows, cols, total_mines)
        board.cells = [[int(v) for v in row] for row in self.states]
        return board


def _is_clipped(frame: np.ndarray, grid: Grid) -> bool:
    """True if the board probably continues beyond the captured picture.

    Either the grid runs right up to the frame edge, or the strip between the
    grid and the edge looks like the start of one more row / column of cells.
    """
    height, width = frame.shape[:2]
    cell_w, cell_h = grid.cell_size
    x0, x1 = grid.x_lines[0], grid.x_lines[-1]
    y0, y1 = grid.y_lines[0], grid.y_lines[-1]
    if min(x0, width - x1, y0, height - y1) < 2:
        return True
    board = frame[int(round(y0)): int(round(y1)), int(round(x0)): int(round(x1))].astype(np.int16)

    def continues(strip: np.ndarray, references: list, bands: int, axis: int) -> bool:
        """Does `strip` repeat the start of a neighbouring row/column? Checked
        per cell against the next cell in and the one after (checkerboards)."""
        best = None
        for reference in references:
            if strip.size == 0 or strip.shape != reference.shape:
                continue
            diff = np.abs(strip.astype(np.int16) - reference).mean(axis=2)
            per_cell = np.array([chunk.mean() for chunk in np.array_split(diff, bands, axis=axis)])
            best = per_cell if best is None else np.minimum(best, per_cell)
        return best is not None and float(np.mean(best < 10)) >= 0.4

    ya, yb, xa, xb = int(round(y0)), int(round(y1)), int(round(x0)), int(round(x1))
    cw, ch = int(round(cell_w)), int(round(cell_h))
    rows, cols = board.shape[0], board.shape[1]
    m = min(int(width - x1), cw - 1)        # right: strip vs the same part of the last columns
    if m >= 3 and continues(frame[ya:yb, xb: xb + m],
                            [board[:, cols - k * cw: cols - k * cw + m] for k in (1, 2)], grid.rows, 0):
        return True
    m = min(int(x0), cw - 1)                # left
    if m >= 3 and continues(frame[ya:yb, xa - m: xa],
                            [board[:, k * cw - m: k * cw] for k in (1, 2)], grid.rows, 0):
        return True
    m = min(int(height - y1), ch - 1)       # bottom
    if m >= 3 and continues(frame[yb: yb + m, xa:xb],
                            [board[rows - k * ch: rows - k * ch + m, :] for k in (1, 2)], grid.cols, 1):
        return True
    m = min(int(y0), ch - 1)                # top
    if m >= 3 and continues(frame[ya - m: ya, xa:xb],
                            [board[k * ch - m: k * ch, :] for k in (1, 2)], grid.cols, 1):
        return True
    return False


def _describe(grid: Grid, unknown: int, clipped: bool) -> str:
    message = f"{grid.cols}×{grid.rows} board recognised"
    if unknown:
        message += f", {unknown} uncertain cell{'s' if unknown != 1 else ''}"
    if clipped:
        message += ". It reaches the edge of the capture region: widen the region if rows or columns are missing"
    return message


def recognize(frame: np.ndarray, grid: Optional[Grid] = None,
              cells: Optional[np.ndarray] = None, reuse: Optional[tuple] = None) -> Recognition:
    """Detect the board in an RGB frame. Pass a previously found `grid` to skip
    detection when the capture region has not moved (and `cells`, its
    `extract_cells` output, if already computed)."""
    started = time.perf_counter()
    if grid is None:
        grid = detect_grid(frame)
        cells = None
    if grid is None:
        return Recognition(False, "No Minesweeper grid found in the capture region.",
                           elapsed_ms=(time.perf_counter() - started) * 1000)
    cell_px = min(grid.cell_size)
    if cell_px < MIN_CELL_PX - 0.5:
        return Recognition(False, f"Found a {grid.cols}×{grid.rows} grid, but its cells are only "
                                  f"{cell_px:.0f} px: too small to read reliably. Zoom the game to "
                                  f"at least {MIN_CELL_PX} px per cell.", grid,
                           elapsed_ms=(time.perf_counter() - started) * 1000)
    reading = classify_cells(frame, grid, cells, reuse)
    unknown = int((reading.states == UNKNOWN).sum())
    elapsed = (time.perf_counter() - started) * 1000
    total = grid.rows * grid.cols
    if unknown > MAX_UNKNOWN_FRACTION * total:
        return Recognition(False, f"Found a {grid.cols}×{grid.rows} grid but could not read "
                                  f"{unknown} of its {total} cells.",
                           grid, reading.states, reading.confidence, unknown, elapsed)
    clipped = _is_clipped(frame, grid)
    return Recognition(True, _describe(grid, unknown, clipped), grid, reading.states,
                       reading.confidence, unknown, elapsed, clipped=clipped,
                       features=reading.features)


def changed_cells(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """[rows, cols] bool: which cells of two `resample_board` images differ visibly."""
    diff = channel_max(cv2.absdiff(before, after))
    rows, cols = diff.shape[0] // S, diff.shape[1] // S
    # area-averaging down to one pixel per cell = the per-cell mean
    mean = cv2.resize(diff.astype(np.float32), (cols, rows), interpolation=cv2.INTER_AREA)
    strong = cv2.resize((diff > CELL_PIXEL_CHANGE).astype(np.float32), (cols, rows),
                        interpolation=cv2.INTER_AREA)
    return (mean > CELL_MEAN_CHANGE) | (strong > CELL_PIXEL_FRACTION)


class BoardReader:
    """Recognises successive frames of the same capture region, incrementally.

    - Grid detection, the expensive step, is reused while the frame size is
      unchanged and the grid lines are still where they were.
    - Each frame's cells are compared with the previous frame's. If none
      changed (a ticking timer or a hover outside the board), the previous
      reading is returned without classifying anything.
    - When some cells changed, the board is classified again (deciding which
      cells are covered is a whole-board judgement), but every cell whose
      pixels did not change keeps its earlier reading unless that reading was
      UNKNOWN and the new reading is UNKNOWN: a cell cannot change state
      without changing its picture, so this only fills gaps, while a
      confident new reading always replaces an older one.
    """

    def __init__(self):
        self.reset()
        self.frames = self.skipped = self.detections = 0

    def reset(self) -> None:
        self._grid: Optional[Grid] = None
        self._shape: Optional[tuple] = None
        self._baseline = 0.0
        self._image: Optional[np.ndarray] = None
        self._last: Optional[Recognition] = None

    def _grid_still_there(self, frame: np.ndarray) -> bool:
        steps = line_steps(frame, self._grid)
        return float((steps >= 0.3 * self._baseline).mean()) >= 0.8

    def read(self, frame: np.ndarray) -> Recognition:
        started = time.perf_counter()
        self.frames += 1
        if self._grid is not None and frame.shape == self._shape and self._grid_still_there(frame):
            image = resample_board(frame, self._grid)
            last = self._last
            changed = changed_cells(self._image, image) if last is not None else None
            if changed is not None and not changed.any():
                self.skipped += 1
                return replace(last, skipped=True, reused_grid=True, kept=0, changed=changed,
                               elapsed_ms=(time.perf_counter() - started) * 1000)
            reuse = (last.features, changed) if changed is not None and last is not None else None
            result = recognize(frame, self._grid, cells_from_image(image, self._grid), reuse)
            if result.ok:
                result.reused_grid = True
                if changed is not None and last is not None and last.states.shape == result.states.shape:
                    # an unchanged cell that now reads as UNKNOWN keeps its earlier
                    # reading; a confident new reading always wins, so that an
                    # earlier mistake cannot outlive the frame it was made in
                    keep = ~changed & (last.states != UNKNOWN) & (result.states == UNKNOWN)
                    if keep.any():
                        result.states = result.states.copy()
                        result.states[keep] = last.states[keep]
                        result.kept = int(keep.sum())
                        result.unknown_count = int((result.states == UNKNOWN).sum())
                        result.message = _describe(result.grid, result.unknown_count, result.clipped)
                result.changed = changed if changed is not None else np.ones(result.states.shape, bool)
                self._image, self._last = image, result
                result.elapsed_ms = (time.perf_counter() - started) * 1000
                return result
        self.detections += 1
        result = recognize(frame)
        if result.ok:
            self._grid, self._shape = result.grid, frame.shape
            self._baseline = float(np.median(line_steps(frame, result.grid)))
            self._image = resample_board(frame, result.grid)
            self._last = result
            result.changed = np.ones(result.states.shape, bool)
        else:
            self.reset()
        result.elapsed_ms = (time.perf_counter() - started) * 1000
        return result
