"""Plain-language explanation of the analysis of one cell.

Returns a list of (title, text) sections; the UI decides how to show them.
"""
from __future__ import annotations

from .board import FLAG, MINE, UNKNOWN, UNREVEALED, Board, Cell, cell_name
from .recommend import Recommendation
from .solver import ESTIMATED, EXACT, PRIOR, UNCERTAIN, SolveResult

QUALITY = {
    EXACT: "exact",
    ESTIMATED: "estimated by sampling (its region is too large to enumerate)",
    PRIOR: "exact for an assumed mine density (the total mine count is unknown)",
    UNCERTAIN: "approximate (some cells were not recognised)",
}


def _names(cells, limit: int = 6) -> str:
    cells = sorted(cells)
    text = ", ".join(cell_name(c) for c in cells[:limit])
    return text + (f" and {len(cells) - limit} more" if len(cells) > limit else "")


def _pct(p: float) -> str:
    if 0 < p < 0.001:
        return "<0.1%"
    if 0.999 < p < 1:
        return ">99.9%"
    return f"{p * 100:.1f}%"


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def clue_line(board: Board, result: SolveResult, clue: Cell, focus: Cell | None = None) -> str:
    """'C3 shows 2: 1 flagged, needs 1 more among B2, B4, D4' (with what the solver proved)."""
    value = board[clue]
    neighbours = board.neighbors(clue)
    flagged = [n for n in neighbours if board[n] in (FLAG, MINE)]
    covered = [n for n in neighbours if board[n] in (UNREVEALED, UNKNOWN)]
    need = value - len(flagged)
    proven_mines = [n for n in covered if result.probs.get(n) == 1.0]
    proven_safe = [n for n in covered if result.probs.get(n) == 0.0]
    parts = [f"{cell_name(clue)} shows {value}"]
    if flagged:
        parts.append(f"{len(flagged)} flagged")
    if need < 0:
        parts.append(f"too many flags ({len(flagged)})")
    elif need == 0:
        parts.append("all its mines are flagged" + (f", so {_names(covered)} are safe" if covered else ""))
    else:
        where = ", ".join(cell_name(c) + (" (this cell)" if c == focus else "") for c in sorted(covered))
        parts.append(f"needs {need} more among {len(covered)} covered: {where}")
        if proven_mines:
            parts.append(f"proven mines: {_names(proven_mines)}")
        if proven_safe:
            parts.append(f"proven safe: {_names(proven_safe)}")
    return " · ".join(parts)


def explain_cell(board: Board, result: SolveResult, rec: Recommendation | None,
                 cell: Cell) -> list[tuple[str, str]]:
    state = board[cell]
    name = cell_name(cell)
    if state >= 0:
        return [("Clue", clue_line(board, result, cell))]
    if state in (FLAG, MINE):
        clues = [n for n in board.neighbors(cell) if board[n] >= 0]
        text = ("Counted as a mine. Flags are trusted, so a wrong flag distorts the analysis."
                if state == FLAG else "An exploded mine: the game is over.")
        if state == FLAG:
            contradicted = [c for c in clues if board[c] - sum(
                board[n] in (FLAG, MINE) for n in board.neighbors(c)) < 0]
            if contradicted:
                text += f" The clue at {_names(contradicted)} has too many flags around it."
        return [("Flag" if state == FLAG else "Mine", text)]
    if not result.valid:
        return [("No analysis", result.error or "The board is inconsistent.")]
    if state == UNKNOWN or cell not in result.probs:
        return [("Not recognised",
                 "The capture of this cell could not be read with confidence, so it is not "
                 "guessed. It is treated as a covered cell that may or may not hold a mine, "
                 "which keeps every proven result valid.")]

    p = result.probs[cell]
    quality = result.quality.get(cell, EXACT)
    sections = [("Probability",
                 f"{_pct(p)} mine, {_pct(1 - p)} safe — {QUALITY.get(quality, quality)}.")]
    sections.append(("Why", result.reasons.get(cell, "")))

    clues = [n for n in board.neighbors(cell) if board[n] >= 0]
    if clues:
        sections.append(("Clues around it", "\n".join(clue_line(board, result, c, cell) for c in clues)))
    else:
        sections.append(("Clues around it",
                         "None: no revealed number touches this cell, so only the overall "
                         "mine count says anything about it."))

    region = result.region_of.get(cell)
    if region is not None:
        info = result.regions[region]
        text = f"Part of a region of {info['cells']} covered cells tied together by {info['clues']} clues"
        if info["arrangements"] is not None:
            count = info["arrangements"]
            text += (f"; the clues allow {count:,} different mine arrangement"
                     f"{'s' if count != 1 else ''} there (before weighing in the global mine count).")
        else:
            text += "; too many arrangements to count, so they were sampled."
        sections.append(("Region", text))

    if result.remaining_mines is not None and result.unrevealed_count:
        undecided = [c for c, q in result.probs.items() if 0 < q < 1]
        pending = result.remaining_mines - len(result.mine_cells)
        text = (f"{result.remaining_mines} mine{'s' if result.remaining_mines != 1 else ''} not yet "
                f"flagged among {result.unrevealed_count} covered cells")
        if undecided:
            avg = max(pending, 0) / len(undecided)
            text += (f"; leaving out the proven ones, {pending} among {len(undecided)} undecided "
                     f"cells, an average of {_pct(avg)}")
            if 0 < p < 1:
                text += (". This cell is " + ("safer" if p < avg else "riskier" if p > avg else "as risky")
                         + " than average")
        sections.append(("Whole board", text + "."))

    if 0 < p < 1:
        ranked = sorted(q for q in result.probs.values() if 0 < q < 1)
        safer = sum(q < p - 1e-12 for q in ranked)
        tied = sum(abs(q - p) <= 1e-12 for q in ranked)
        text = f"{_ordinal(safer + 1)} safest of {len(ranked)} undecided cells"
        if tied > 1:
            text += f" (tied with {tied - 1} other{'s' if tied > 2 else ''})"
        if result.safe_cells:
            text += f"; {len(result.safe_cells)} proven safe cell{'s' if len(result.safe_cells) != 1 else ''} should be revealed first"
        sections.append(("Ranking", text + "."))
    if rec is not None and rec.cell == cell:
        sections.append(("Recommendation", rec.headline + "."))
    return sections
