"""Finds the Minesweeper cell lattice in a screenshot.

No theme knowledge is used. A board is a regular lattice, so the columns
(rows) where many pixels differ from their neighbour repeat with a fixed
period. We measure that "edge coverage" per column and row, find the period
by autocorrelation, fit period and phase precisely, and keep the longest run of
lattice lines that are actually present.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

EDGE_THRESHOLD = 6       # colour step (0-255) across two pixels that counts as an edge
MIN_PERIOD = 8           # smallest supported cell size in pixels
MAX_PERIOD = 160
MIN_CELLS = 3
FEW_LINES = 7             # an axis with this many lines or fewer borrows the other's period
MIN_QUALITY = 0.6       # mean edge coverage of the lattice lines; real boards score ~0.9+


@dataclass
class Grid:
    x_lines: np.ndarray   # cols + 1 boundaries (pixel-edge coordinates, float)
    y_lines: np.ndarray   # rows + 1 boundaries
    quality: float        # 0..1, how much of the lattice is backed by edges

    @property
    def rows(self) -> int:
        return len(self.y_lines) - 1

    @property
    def cols(self) -> int:
        return len(self.x_lines) - 1

    @property
    def cell_size(self) -> tuple[float, float]:
        return (float(self.x_lines[1] - self.x_lines[0]), float(self.y_lines[1] - self.y_lines[0]))


def channel_max(img: np.ndarray) -> np.ndarray:
    """Largest of the three colour channels per pixel. Much faster than
    `img.max(axis=-1)`, which numpy reduces element by element."""
    return np.maximum(np.maximum(img[..., 0], img[..., 1]), img[..., 2])


@dataclass
class _Axis:
    period: float
    lines: np.ndarray     # boundaries of the detected run
    strength: float


class _Edges:
    """Colour-step maps of one frame, computed once and sliced many times.

    `wide[axis]`: step across two pixels (i-1 vs i+1) above EDGE_THRESHOLD, so
    edges softened by browser zoom still register. `narrow[axis]`: size of
    the step between neighbouring pixels (used to snap the lattice).
    """

    def __init__(self, frame: np.ndarray):
        self.frame = frame
        self._wide: dict = {}
        self._narrow: dict = {}

    def wide(self, axis: int) -> np.ndarray:
        if axis not in self._wide:
            f = self.frame
            d = cv2.absdiff(f[:, 2:], f[:, :-2]) if axis == 1 else cv2.absdiff(f[2:], f[:-2])
            self._wide[axis] = channel_max(d) > EDGE_THRESHOLD
        return self._wide[axis]

    def narrow(self, axis: int) -> np.ndarray:
        if axis not in self._narrow:
            f = self.frame
            d = cv2.absdiff(f[:, 1:], f[:, :-1]) if axis == 1 else cv2.absdiff(f[1:], f[:-1])
            self._narrow[axis] = channel_max(d).astype(np.float32)
        return self._narrow[axis]

    def coverage(self, axis: int, lo: Optional[int] = None, hi: Optional[int] = None) -> np.ndarray:
        """Fraction of pixels (between `lo` and `hi` on the other axis) that sit
        on a colour step. Entry i describes pixel i (centre at i + 0.5)."""
        wide = self.wide(axis)
        part = wide[lo:hi] if axis == 1 else wide[:, lo:hi]
        if part.size == 0:
            return np.zeros(wide.shape[1 if axis == 1 else 0] + 2)
        profile = part.mean(axis=1 - axis)
        return np.concatenate(([0.0], profile, [0.0]))


def _box(profile: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return profile
    kernel = np.ones(width) / width
    return np.convolve(profile, kernel, mode="same")


def _period_candidates(profile: np.ndarray, lo: int, hi: int) -> list[int]:
    """Autocorrelation peaks, strongest harmonics filtered, smallest lag first."""
    n = len(profile)
    hi = min(hi, n // 3)
    if hi <= lo + 1:
        return []
    centred = profile - profile.mean()
    full = np.correlate(centred, centred, mode="full")[n - 1:]
    if full[0] <= 0:
        return []
    r = full / np.arange(n, 0, -1) / (full[0] / n)
    lags = [lag for lag in range(lo, hi) if r[lag] >= r[lag - 1] and r[lag] >= r[lag + 1] and r[lag] > 0.15]
    if not lags:
        return []
    best = max(r[lag] for lag in lags)
    return [lag for lag in lags if r[lag] >= 0.7 * best]


def _fit(profile: np.ndarray, guess: float, fixed: bool = False) -> tuple[float, float]:
    """Coarse period (near `guess`, or exactly `guess` if `fixed`) and phase
    maximising the mean profile value on the lattice."""
    n = len(profile)
    best = (-1.0, guess, 0.0)
    span = 0.0 if fixed else max(1.0, guess * 0.04)
    for period in np.arange(guess - span, guess + span + 1e-9, 0.2):
        count = int((n - 1) / period) + 1
        steps = period * np.arange(count)
        phases = np.arange(0.0, period, 1.0)
        idx = np.rint(phases[:, None] + steps[None, :]).astype(int)
        valid = idx < n
        values = np.where(valid, profile[np.minimum(idx, n - 1)], 0.0)
        scores = values.sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
        i = int(np.argmax(scores))
        if scores[i] > best[0]:
            best = (float(scores[i]), float(period), float(phases[i]))
    return best[1], best[2]


def _long_baseline_slope(index: np.ndarray, centres: np.ndarray) -> float:
    """Median slope over pairs of lines at least half the run apart: robust to
    a few bad lines (Theil-Sen), but without the noise of neighbouring pairs."""
    i, j = np.triu_indices(len(index), k=1)
    span = index[j] - index[i]
    far = span >= max(1.0, (index.max() - index.min()) / 2)
    return float(np.median((centres[j][far] - centres[i][far]) / span[far]))


def _refine(energy: np.ndarray, period: float, phase: float,
            fixed: bool = False) -> tuple[float, float]:
    """Robust lattice through the centre of each boundary. A boundary may
    consist of several edges (bevels, gaps), so each line is moved to the
    centroid of the edge energy around it before fitting. With `fixed` only
    the phase is fitted."""
    n = len(energy)
    mass = np.concatenate(([0.0], np.cumsum(energy)))
    moment = np.concatenate(([0.0], np.cumsum(energy * np.arange(n))))
    # one pass only: re-centring the windows on a slightly-off estimate and
    # measuring again feeds on itself (a window that slides onto a frame edge
    # pulls the estimate further the same way)
    for _ in range(1):
        reach = max(1, int(round(period * 0.14)))
        ks = np.arange(int((n - 1 - phase) / period) + 1)
        nominal = phase + period * ks
        rounded = np.rint(nominal).astype(int)
        lo = np.maximum(0, rounded - reach)
        hi = np.minimum(n, rounded + reach + 1)
        weights = mass[hi] - mass[lo]
        centres = np.where(weights > 0, (moment[hi] - moment[lo]) / np.maximum(weights, 1e-12), nominal)
        keep = weights >= 0.3 * weights.max()
        if keep.sum() < 3:
            break
        # Theil-Sen fit (median of pairwise slopes): unaffected by the few
        # lines that a frame or panel edge just outside the board drags away
        centres = centres[keep]
        index = ks[keep].astype(float)
        if fixed:
            intercept = float(np.median(centres - period * index))
            phase = float(intercept - period * np.floor(intercept / period))
            continue
        slope = _long_baseline_slope(index, centres)
        intercept = float(np.median(centres - slope * index))
        if not (0.9 * period < slope < 1.1 * period):
            break
        period = float(slope)
        phase = float(intercept - period * np.floor(intercept / period))
    return period, phase


def _fit_inner(energy: np.ndarray, period: float, phase: float, ks: np.ndarray,
               fixed: bool) -> tuple[float, float]:
    """Refit the lattice from the lines with index in `ks` only (the inner
    lines of the board: the outer two also contain the board's frame, whose
    edges pull them off the cell boundary)."""
    n = len(energy)
    mass = np.concatenate(([0.0], np.cumsum(energy)))
    moment = np.concatenate(([0.0], np.cumsum(energy * np.arange(n))))
    reach = max(1, int(round(period * 0.14)))
    nominal = phase + period * ks
    rounded = np.rint(nominal).astype(int)
    lo = np.clip(rounded - reach, 0, n)
    hi = np.clip(rounded + reach + 1, 0, n)
    weights = mass[hi] - mass[lo]
    if (weights <= 0).any():
        return period, phase
    centres = (moment[hi] - moment[lo]) / weights
    index = ks.astype(float)
    if not fixed:
        slope = _long_baseline_slope(index, centres)
        if not (0.95 * period < slope < 1.05 * period):
            return period, phase
        period = slope
    intercept = float(np.median(centres - period * index))
    return period, intercept


def _axis(profile: np.ndarray, period_hint: Optional[float] = None,
          fixed: bool = False) -> Optional[_Axis]:
    n = len(profile)
    energy = profile ** 2          # full-length lines dominate glyph edges
    smooth = _box(energy, 3)
    if period_hint is None:
        candidates = _period_candidates(smooth, MIN_PERIOD, MAX_PERIOD)
    else:
        candidates = [period_hint]
    best: Optional[_Axis] = None
    for guess in candidates[:3]:
        period, phase = _fit(smooth, float(guess), fixed)
        period, phase = _refine(energy, period, phase, fixed)
        ks = np.arange(int((n - 1 - phase) / period) + 1)
        positions = phase + period * ks
        idx = np.rint(positions).astype(int)
        reach = max(1, int(round(period * 0.14)))
        window = np.clip(idx[:, None] + np.arange(-reach, reach + 1)[None, :], 0, n - 1)
        strength = profile[window].max(axis=1)
        if len(strength) < MIN_CELLS + 1:
            continue
        top = np.sort(strength)[::-1][: max(3, len(strength) // 4)]
        present = strength >= 0.6 * top.mean()
        # longest run of present lines
        run_start = run_len = cur_start = cur_len = 0
        for i, ok in enumerate(present):
            if ok:
                if cur_len == 0:
                    cur_start = i
                cur_len += 1
                if cur_len > run_len:
                    run_start, run_len = cur_start, cur_len
            else:
                cur_len = 0
        if run_len < MIN_CELLS + 1:
            continue
        if run_len >= 5:                     # refine from the inner lines only
            period, phase = _fit_inner(energy, period, phase,
                                       ks[run_start + 1: run_start + run_len - 1], fixed)
            positions = phase + period * ks
        lines = positions[run_start: run_start + run_len] + 0.5   # pixel index -> pixel centre
        score = float(strength[run_start: run_start + run_len].mean()) * run_len
        if best is None or score > best.strength * 1.15:   # prefer the smaller period on ties
            best = _Axis(period, lines, score)
    return best


def detect_grid(frame: np.ndarray) -> Optional[Grid]:
    """`frame` is an RGB uint8 image. Returns None if no lattice is found."""
    if frame.ndim != 3 or frame.shape[0] < 3 * MIN_PERIOD or frame.shape[1] < 3 * MIN_PERIOD:
        return None
    edges = _Edges(np.ascontiguousarray(frame))
    xs = _axis(edges.coverage(1))
    ys = _axis(edges.coverage(0))
    if xs is None or ys is None:
        return None

    # second pass: measure each axis only inside the other axis' extent, which
    # removes window chrome, counters and anything else around the board
    for _ in range(2):
        y0, y1 = int(round(ys.lines[0])), int(round(ys.lines[-1]))
        x0, x1 = int(round(xs.lines[0])), int(round(xs.lines[-1]))
        new_xs = _axis(edges.coverage(1, y0, y1))
        new_ys = _axis(edges.coverage(0, x0, x1))
        if new_xs is None or new_ys is None:
            break
        xs, ys = new_xs, new_ys

    # cells are square: if the periods disagree, one axis locked onto a
    # harmonic, so retry it with the other axis' period
    if abs(xs.period - ys.period) > 0.08 * min(xs.period, ys.period):
        small = min(xs.period, ys.period)
        y0, y1 = int(round(ys.lines[0])), int(round(ys.lines[-1]))
        x0, x1 = int(round(xs.lines[0])), int(round(xs.lines[-1]))
        if xs.period > ys.period:
            xs = _axis(edges.coverage(1, y0, y1), small) or xs
        else:
            ys = _axis(edges.coverage(0, x0, x1), small) or ys
        if abs(xs.period - ys.period) > 0.08 * min(xs.period, ys.period):
            return None

    # cells are square, and the axis with more lines measures the period more
    # precisely: on a narrow board 5 column lines can be off by a few percent,
    # which adds up to most of a cell at the far side
    if min(len(xs.lines), len(ys.lines)) <= FEW_LINES and abs(xs.period - ys.period) > 1e-6:
        y0, y1 = int(round(ys.lines[0])), int(round(ys.lines[-1]))
        x0, x1 = int(round(xs.lines[0])), int(round(xs.lines[-1]))
        if len(ys.lines) > len(xs.lines):
            xs = _axis(edges.coverage(1, y0, y1), ys.period, fixed=True) or xs
        elif len(xs.lines) > len(ys.lines):
            ys = _axis(edges.coverage(0, x0, x1), xs.period, fixed=True) or ys

    y0, y1 = int(round(ys.lines[0])), int(round(ys.lines[-1]))
    x0, x1 = int(round(xs.lines[0])), int(round(xs.lines[-1]))
    x_lines = _snap(edges.narrow(1)[y0:y1].mean(axis=0), xs.lines)
    y_lines = _snap(edges.narrow(0)[:, x0:x1].mean(axis=1), ys.lines)
    x_lines, y_lines = _trim(edges, x_lines, y_lines)
    quality = min(1.0, (xs.strength / len(xs.lines) + ys.strength / len(ys.lines)) / 2)
    if quality < MIN_QUALITY:
        return None                       # periodic clutter (text, icons), not a board
    return Grid(x_lines, y_lines, quality)


def _support(edges: _Edges, axis: int, band: tuple[float, float], lines: np.ndarray) -> float:
    """Share of the cell boundaries `lines` (along `axis`) that are visible
    inside `band` on the other axis. Every real row of cells shows its column
    boundaries (bevels, gaps, a checkerboard), a strip of page above the board
    does not."""
    inner = lines[1:-1]
    if len(inner) == 0:
        return 1.0
    wide = edges.wide(axis)
    lo, hi = int(round(band[0])) + 2, int(round(band[1])) - 2
    if hi <= lo:
        return 1.0
    part = wide[lo:hi] if axis == 1 else wide[:, lo:hi].T          # [band, position]
    profile = part.mean(axis=0)
    seen = 0
    for p in inner:
        i = int(round(p)) - 1           # wide index j describes pixel j + 1
        window = profile[max(0, i - 2): i + 3]
        seen += window.size > 0 and window.max() > 0.8   # a boundary runs the full band
    return seen / len(inner)


def _trim(edges: _Edges, x_lines: np.ndarray, y_lines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop outer rows / columns that are not really cells, e.g. a counter
    panel above the board whose edges happen to fall on lattice lines. Real
    rows always show their column boundaries; a row of the lattice that does
    not, and everything beyond it, is outside the board."""
    def cut(lines: np.ndarray, others: np.ndarray, axis: int) -> np.ndarray:
        n = len(lines) - 1
        depth = min(3, n - MIN_CELLS)
        start = 0
        for i in range(depth):                      # from the near edge inwards
            if _support(edges, axis, (lines[i], lines[i + 1]), others) < 0.3:
                start = i + 1
        end = n
        for i in range(depth):                      # from the far edge inwards
            if _support(edges, axis, (lines[n - 1 - i], lines[n - i]), others) < 0.3:
                end = n - 1 - i
        if end - start < MIN_CELLS:
            return lines
        return lines[start: end + 1]

    for _ in range(2):                              # rows, then columns, then once more
        y_lines = cut(y_lines, x_lines, 1)
        x_lines = cut(x_lines, y_lines, 0)
    return x_lines, y_lines


def _snap(step: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """Align the lattice with the actual cell seams: try a global shift and a
    per-line fit, and keep whichever sits on the stronger pixel steps."""
    magnitude = np.concatenate(([0.0], step, [0.0]))
    positions = np.arange(len(magnitude))
    shifted = _shift_snap(step, lines)
    fitted = _shift_snap(step, _line_fit(magnitude, lines))
    score = lambda ls: float(np.interp(ls, positions, magnitude).sum())
    return fitted if score(fitted) > score(shifted) * 1.01 else shifted


def _line_fit(magnitude: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """Move each line to the strongest single-pixel step within 2 px and fit a
    straight lattice through them. Corrects a period that is slightly off
    (which adds up to a pixel or two across the board)."""
    if len(lines) < 4:
        return lines
    n = len(magnitude)
    found = []
    for x in lines:
        i = int(round(x))
        lo, hi = max(0, i - 2), min(n, i + 3)
        found.append(lo + int(np.argmax(magnitude[lo:hi])) if hi > lo else x)
    found = np.array(found, float)
    index = np.arange(len(lines), dtype=float)
    period = float(lines[1] - lines[0])
    slope = _long_baseline_slope(index, found)
    if not 0.97 * period < slope < 1.03 * period:
        return lines
    return float(np.median(found - slope * index)) + slope * index


def _shift_snap(step: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """Shift the lattice by up to 1.5 px onto the strongest colour steps.

    A cell boundary drawn as a bevel or a thin line consists of several edges;
    the coverage fit lands on their centre, which can be a pixel away from
    where one cell's pixels end and the next begin. The largest step is the
    true seam (e.g. shadow of one tile against the highlight of the next).
    """
    magnitude = np.concatenate(([0.0], step, [0.0]))     # entry i = boundary before pixel i
    positions = np.arange(len(magnitude))
    best_shift, best_score = 0.0, -1.0
    for shift in sorted(np.arange(-1.5, 1.51, 0.25), key=abs):
        score = float(np.interp(lines + shift, positions, magnitude).sum())
        if score > best_score * 1.02:                     # prefer the smaller shift on ties
            best_shift, best_score = float(shift), score
    return lines + best_shift


def line_steps(frame: np.ndarray, grid: Grid) -> np.ndarray:
    """Mean colour step along each interior grid line (columns then rows).
    Lets a caller notice cheaply that a previously detected grid no longer
    matches the picture (window moved, page scrolled, game closed)."""
    x0, x1 = int(round(grid.x_lines[0])), int(round(grid.x_lines[-1]))
    y0, y1 = int(round(grid.y_lines[0])), int(round(grid.y_lines[-1]))
    img = frame[max(0, y0): y1, max(0, x0): x1]
    if img.shape[0] < 2 or img.shape[1] < 2:
        return np.zeros(1)
    out = []
    for axis, lines, origin in ((1, grid.x_lines, x0), (0, grid.y_lines, y0)):
        d = cv2.absdiff(img[:, 1:], img[:, :-1]) if axis == 1 else cv2.absdiff(img[1:], img[:-1])
        step = cv2.reduce(channel_max(d), 1 - axis, cv2.REDUCE_AVG, dtype=cv2.CV_32F).ravel()
        magnitude = np.concatenate(([0.0], step, [0.0]))
        for position in lines[1:-1] - max(0, origin):
            i = int(round(position))
            out.append(magnitude[max(0, i - 1): i + 2].max(initial=0.0))
    return np.array(out) if out else np.zeros(1)
