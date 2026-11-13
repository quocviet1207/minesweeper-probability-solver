"""Classifies every cell of a detected grid without theme-specific colours.

Self-calibrating per frame:
1. Each cell is resampled to SxS and described by its background colour and a
   bevel measure (top-left edge brighter than bottom-right = raised tile).
2. Cells are clustered by that description into visual styles.
3. Which styles are "covered" and which "revealed" is decided by game logic,
   not colour: a revealed blank cell can never touch a covered cell, glyphs on
   revealed cells look like digits, and at least one cell is always covered.
4. Glyphs on revealed cells are read as digits by shape, with a vote among
   glyphs of the same colour; anything that cannot be read confidently becomes
   UNKNOWN rather than a guess.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..core.board import FLAG, MINE, UNKNOWN, UNREVEALED
from .glyphs import CANVAS, count_holes, digit_scores, normalize_mask
from .grid import Grid, channel_max

S = 32                   # resampled cell size
RING = (4, 7)            # background is sampled between these insets
GLYPH_INSET = 4          # glyph window = cell minus this inset
FG_DISTANCE = 40         # colour distance from background that counts as glyph
MIN_GLYPH_FRACTION = 0.035
STYLE_DISTANCE = 24      # max colour distance inside one visual style
COLOR_DISTANCE = 45      # max distance between glyph colours of the same digit
MAX_STYLES = 8

DIGIT_OK = 0.60          # minimum shape score to accept a digit at all
DIGIT_SURE = 0.80        # score at which a digit needs no colour vote
DIGIT_MARGIN = 0.05      # lead over the runner-up digit
VOTE_MARGIN = 0.03       # mean lead a colour group needs to settle on one digit


@dataclass
class CellReading:
    states: np.ndarray       # [rows, cols] int, Board cell states incl. UNKNOWN
    confidence: np.ndarray   # [rows, cols] float 0..1
    styles: int              # number of visual styles found
    scores: np.ndarray       # [rows, cols, 9] digit shape scores (diagnostics)
    features: dict = None    # per-cell glyph measurements, reusable for unchanged cells


GLYPH_FEATURES = ("has_glyph", "scores", "glyph_color", "blob_like", "holes", "aspect", "purity",
                  "masks")


def extract_cells(frame: np.ndarray, grid: Grid) -> np.ndarray:
    """[rows, cols, S, S, 3] float32 tensor of resampled cells."""
    return cells_from_image(resample_board(frame, grid), grid)


def cells_from_image(image: np.ndarray, grid: Grid) -> np.ndarray:
    """Split a `resample_board` image into a [rows, cols, S, S, 3] float32 tensor."""
    return image.reshape(grid.rows, S, grid.cols, S, 3).transpose(0, 2, 1, 3, 4).astype(np.float32)


def resample_board(frame: np.ndarray, grid: Grid) -> np.ndarray:
    """The board resampled to S x S pixels per cell, as a uint8 image (sub-pixel
    accurate: the lattice may start between pixels and have a fractional period)."""
    x0, x1 = float(grid.x_lines[0]), float(grid.x_lines[-1])
    y0, y1 = float(grid.y_lines[0]), float(grid.y_lines[-1])
    out_w, out_h = grid.cols * S, grid.rows * S
    sx, sy = (x1 - x0) / out_w, (y1 - y0) / out_h
    # destination pixel centre (u + 0.5) maps to source x0 + (u + 0.5) * sx
    matrix = np.array([[sx, 0, x0 + 0.5 * sx - 0.5], [0, sy, y0 + 0.5 * sy - 0.5]], np.float32)
    shrink = sx > 1.0
    return cv2.warpAffine(frame, matrix, (out_w, out_h),
                          flags=(cv2.INTER_AREA if shrink else cv2.INTER_LINEAR) | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def _leader_clusters(points: np.ndarray, threshold: float) -> np.ndarray:
    """Greedy clustering with the max-abs metric; returns a label per point."""
    leaders: list[np.ndarray] = []
    labels = np.zeros(len(points), int)
    for i, point in enumerate(points):
        for j, leader in enumerate(leaders):
            if np.abs(point - leader).max() <= threshold:
                labels[i] = j
                break
        else:
            leaders.append(point)
            labels[i] = len(leaders) - 1
    # one refinement pass around the cluster means
    centres = np.array([points[labels == j].mean(axis=0) for j in range(len(leaders))])
    dist = np.abs(points[:, None, :] - centres[None]).max(axis=2)
    return dist.argmin(axis=1)


def _has_neighbor(mask: np.ndarray) -> np.ndarray:
    """True where any of the 8 neighbours is set in `mask`."""
    padded = np.pad(mask, 1)
    out = np.zeros_like(mask)
    rows, cols = mask.shape
    for dr in (0, 1, 2):
        for dc in (0, 1, 2):
            if (dr, dc) != (1, 1):
                out |= padded[dr: dr + rows, dc: dc + cols]
    return out


def classify_cells(frame: np.ndarray, grid: Grid, cells: Optional[np.ndarray] = None,
                   reuse: Optional[tuple[dict, np.ndarray]] = None) -> CellReading:
    """`cells` may be passed if the caller already ran `extract_cells`.

    `reuse` = (features of an earlier reading of the same grid, [rows, cols]
    mask of cells whose pixels changed since). The glyph analysis, the
    expensive per-cell part, is then only run for changed cells; the
    board-wide decisions are always made afresh.
    """
    if cells is None:
        cells = extract_cells(frame, grid)
    rows, cols = grid.rows, grid.cols

    # --- background colour from a ring that avoids both border and glyph ----
    a, b = RING
    ring_mask = np.zeros((S, S), bool)
    ring_mask[a: S - a, a: S - a] = True
    ring_mask[b: S - b, b: S - b] = False
    bg = np.median(cells[:, :, ring_mask], axis=2)                       # [R, C, 3]

    # --- bevel: top/left edge brightness minus bottom/right ----------------
    gray = cells.mean(axis=4)
    tl = (gray[:, :, :3, :].mean(axis=(2, 3)) + gray[:, :, :, :3].mean(axis=(2, 3))) / 2
    br = (gray[:, :, -3:, :].mean(axis=(2, 3)) + gray[:, :, :, -3:].mean(axis=(2, 3))) / 2
    bevel = tl - br

    # --- glyph mask ---------------------------------------------------------
    g = GLYPH_INSET
    window = cells[:, :, g: S - g, g: S - g]                              # [R, C, W, W, 3]
    dist = channel_max(np.abs(window - bg[:, :, None, None, :]))
    fg = dist > FG_DISTANCE
    width = S - 2 * g
    for edge in (0, 1, 2, width - 3, width - 2, width - 1):              # drop leaked borders
        row_full = fg[:, :, edge, :].mean(axis=2) > 0.7
        fg[:, :, edge, :] &= ~row_full[:, :, None]
        col_full = fg[:, :, :, edge].mean(axis=2) > 0.7
        fg[:, :, :, edge] &= ~col_full[:, :, None]
    has_glyph = fg[:, :, 2:-2, 2:-2].mean(axis=(2, 3)) > MIN_GLYPH_FRACTION

    # --- per-glyph shape and colour ----------------------------------------
    glyph_cells = np.argwhere(has_glyph)
    scores = np.zeros((rows, cols, 9), np.float32)
    glyph_color = np.zeros((rows, cols, 3), np.float32)
    blob_like = np.zeros((rows, cols), bool)
    holes = np.zeros((rows, cols), int)
    aspect = np.ones((rows, cols), np.float32)
    purity = np.ones((rows, cols), np.float32)
    masks = np.zeros((rows, cols, CANVAS, CANVAS), np.float32)   # normalised glyph shapes
    reused = np.zeros((rows, cols), bool)
    if reuse is not None and reuse[0] is not None and reuse[1].shape == (rows, cols):
        old, changed = reuse
        reused = ~changed
        has_glyph[reused] = old["has_glyph"][reused]
        scores[reused] = old["scores"][reused]
        glyph_color[reused] = old["glyph_color"][reused]
        blob_like[reused] = old["blob_like"][reused]
        holes[reused] = old["holes"][reused]
        aspect[reused] = old["aspect"][reused]
        purity[reused] = old["purity"][reused]
        masks[reused] = old["masks"][reused]
        glyph_cells = np.argwhere(has_glyph & ~reused)
    if len(glyph_cells):
        normalized, keep = [], []
        centre = (width - 1) / 2
        for r, c in glyph_cells:
            # keep the connected piece nearest the middle: digits are one
            # piece, and this discards border lines that leak into the window
            count, comp, stats, centroids = cv2.connectedComponentsWithStats(
                fg[r, c].astype(np.uint8), connectivity=8)
            chosen, chosen_area = 0, 0
            for label in range(1, count):
                x, y, w, h, area = stats[label]
                off_centre = max(abs(centroids[label][0] - centre), abs(centroids[label][1] - centre))
                if off_centre > 0.3 * width or area <= chosen_area:
                    continue
                chosen, chosen_area = label, area
            if not chosen:
                has_glyph[r, c] = False
                continue
            x, y, w, h, area = stats[chosen]
            mask = comp == chosen
            if h < 6 or area < MIN_GLYPH_FRACTION * width * width:
                has_glyph[r, c] = False
                continue
            strong = dist[r, c] > 0.6 * dist[r, c][mask].max()
            pixels = window[r, c][strong & mask]
            glyph_color[r, c] = np.median(pixels, axis=0)
            # digits are drawn in one colour; flags usually combine two
            purity[r, c] = (np.abs(pixels - glyph_color[r, c]).max(axis=1) < 60).mean()
            holes[r, c] = count_holes(mask[y: y + h, x: x + w])
            aspect[r, c] = w / h
            # a mine is a large, dark, colourless shape that is solid in the
            # middle or looks the same mirrored both ways; no digit is
            colour = glyph_color[r, c]
            if (0.75 <= w / h <= 1.34 and min(h, w) >= 0.45 * width
                    and colour.max() < 90 and colour.max() - colour.min() < 40):
                box = mask[y: y + h, x: x + w]
                mirrored = min((box & box[:, ::-1]).sum() / (box | box[:, ::-1]).sum(),
                               (box & box[::-1]).sum() / (box | box[::-1]).sum())
                core = box[h // 4: h - h // 4, w // 4: w - w // 4].mean()   # solid middle
                blob_like[r, c] = mirrored >= 0.7 or core >= 0.85
            masks[r, c] = normalize_mask(mask)
            normalized.append(masks[r, c])
            keep.append((r, c))
        if keep:
            result = digit_scores(np.array(normalized))
            for (r, c), row in zip(keep, result):
                # topology is font independent: a glyph with a hole is a 4, 6
                # or 8, and one with two holes is an 8
                if holes[r, c] >= 2:
                    row[[1, 2, 3, 4, 5, 6, 7]] -= 0.15
                elif holes[r, c] == 1:
                    row[[1, 2, 3, 5, 7]] -= 0.15
                else:
                    row[8] -= 0.08
                    row[6] -= 0.04
                if aspect[r, c] >= 0.95:
                    row[1] -= 0.15            # a 1 is never as wide as it is tall
                scores[r, c] = row
    features = {"has_glyph": has_glyph.copy(), "scores": scores.copy(), "glyph_color": glyph_color,
                "blob_like": blob_like.copy(), "holes": holes, "aspect": aspect, "purity": purity, "masks": masks}
    best = scores.max(axis=2)
    margin = best - np.sort(scores, axis=2)[:, :, -2]
    # +1 where a glyph looks like a single-colour digit, -1 where it clearly
    # does not (poor shape match, or two colours as on most flags), else 0
    digit_evidence = np.zeros((rows, cols), np.float32)
    digit_evidence[has_glyph & (best >= DIGIT_OK) & (margin >= 0.02) & (purity >= 0.75)] = 1.0
    digit_evidence[has_glyph & ((best < DIGIT_OK) | (purity < 0.6))] = -1.0
    digit_evidence[blob_like] = 0.0        # mines shown after a loss say nothing either way

    # --- visual styles ------------------------------------------------------
    descriptor = np.concatenate([bg, bevel[:, :, None] * 0.5], axis=2).reshape(-1, 4)
    labels = _leader_clusters(descriptor, STYLE_DISTANCE).reshape(rows, cols)
    sizes = np.bincount(labels.ravel())
    order = np.argsort(sizes)[::-1]
    main = [int(s) for s in order[:MAX_STYLES] if sizes[s] > 0]
    style_bevel = {s: float(bevel[labels == s].mean()) for s in main}
    style_bg = {s: bg[labels == s].mean(axis=0) for s in main}
    raised_pairs = [(i, j) for i in main for j in main
                    if style_bevel[i] - style_bevel[j] > 30
                    and np.abs(style_bg[i] - style_bg[j]).max() < STYLE_DISTANCE]

    # --- which styles are revealed? enumerate and score by game logic -------
    best_score, best_revealed = -np.inf, np.zeros((rows, cols), bool)
    in_main = np.isin(labels, main)
    for bits in range(1 << len(main)):
        chosen = [s for i, s in enumerate(main) if bits >> i & 1]
        revealed = np.isin(labels, chosen)
        covered = in_main & ~revealed
        if not covered.any():
            continue                      # a board always has covered cells (the mines)
        # a revealed blank never touches an unflagged covered cell (the flood
        # fill would have opened it). Next to a flag it rarely does, but it
        # can: a wrong flag stays in place while everything around it opens,
        # so that only counts weakly
        near_any = _has_neighbor(covered)
        near_unflagged = _has_neighbor(covered & ~has_glyph)
        blank = revealed & ~has_glyph
        violations = (blank & near_unflagged).sum() + 0.2 * (blank & near_any & ~near_unflagged).sum()
        score = -10.0 * violations
        score += 0.5 * digit_evidence[revealed].sum() - 0.25 * digit_evidence[covered].sum()
        score -= 0.5 * (revealed & has_glyph & ~blob_like & ~near_any).sum()
        for hi, lo in raised_pairs:       # same colour, one clearly more raised
            if hi in chosen and lo not in chosen:
                score -= 15.0             # raised tiles are the covered ones
        if score > best_score:
            best_score, best_revealed = score, revealed
    revealed = best_revealed
    covered = in_main & ~revealed

    # --- final states -------------------------------------------------------
    states = np.full((rows, cols), UNKNOWN, int)
    confidence = np.zeros((rows, cols), np.float32)

    states[covered & ~has_glyph] = UNREVEALED
    confidence[covered & ~has_glyph] = 1.0
    states[covered & has_glyph] = FLAG
    confidence[covered & has_glyph] = 0.9
    states[revealed & ~has_glyph] = 0
    confidence[revealed & ~has_glyph] = 1.0

    # an exploded mine usually sits on a red background of its own
    red_bg = (bg[:, :, 0] > bg[:, :, 1] + 60) & (bg[:, :, 0] > bg[:, :, 2] + 60)
    lonely = sizes[labels] <= max(2, rows * cols // 100)
    exploded = has_glyph & red_bg & lonely
    states[exploded] = MINE
    confidence[exploded] = 0.8

    # revealed mines (shown when the game is lost) are taken out before the
    # digits are read, so that they cannot sway a colour group's vote
    shown_mine = revealed & has_glyph & blob_like & ~exploded & (best < DIGIT_SURE + 0.1)
    states[shown_mine] = MINE
    confidence[shown_mine] = 0.7
    blob_like &= shown_mine

    todo = revealed & has_glyph & ~exploded & ~shown_mine
    coords = np.argwhere(todo)
    if len(coords):
        colors = glyph_color[todo]
        color_label = _leader_clusters(colors, COLOR_DISTANCE)
        own = scores[todo]                                   # [G, 9]
        own_best = own.argmax(axis=1)
        own_top = own.max(axis=1)
        runner_up = np.sort(own, axis=1)[:, -2]
        for group in np.unique(color_label):
            members = np.nonzero(color_label == group)[0]
            vote = own[members].mean(axis=0)
            digit = int(vote.argmax())
            agree = (own_best[members] == digit).mean()
            # glyphs of one colour are one digit, provided the group agrees
            lead = vote[digit] - np.sort(vote)[-2]
            consistent = len(members) >= 2 and agree >= 0.75 and vote[digit] >= DIGIT_OK and (
                lead >= VOTE_MARGIN
                # a large, unanimous group settles look-alikes such as 1 and 7
                or (len(members) >= 4 and agree >= 0.9 and lead >= 0.01))
            for i in members:
                r, c = coords[i]
                clear = own_top[i] - runner_up[i] >= DIGIT_MARGIN
                if blob_like[r, c]:
                    continue                                   # already reported as a mine
                if purity[r, c] < 0.6:
                    states[r, c], confidence[r, c] = UNKNOWN, float(own_top[i])
                    continue
                if consistent and own[i, digit] >= DIGIT_OK - 0.05 and (
                        own_best[i] == digit or not (clear and own_top[i] >= DIGIT_SURE)):
                    states[r, c], confidence[r, c] = digit, float(own[i, digit])
                elif not consistent and clear and own_top[i] >= DIGIT_SURE:
                    states[r, c], confidence[r, c] = int(own_best[i]), float(own_top[i])
                else:
                    states[r, c], confidence[r, c] = UNKNOWN, float(own_top[i])
    return CellReading(states, confidence, len(main), scores, features)
