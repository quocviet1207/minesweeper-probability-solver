"""Turns solver output into a next-move recommendation with an explanation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .board import FLAG, MINE, UNREVEALED, Board, Cell, cell_name
from .solver import ESTIMATED, EXACT, PRIOR, UNCERTAIN, SolveResult

SIMILAR_RISK = 0.01  # guesses within one percentage point count as equally risky


@dataclass
class Recommendation:
    kind: str                        # "safe" | "guess" | "flag" | "done" | "invalid"
    cell: Optional[Cell] = None      # the cell to reveal (None for flag/done/invalid)
    mine_prob: Optional[float] = None
    safe_cells: list[Cell] = field(default_factory=list)
    mine_cells: list[Cell] = field(default_factory=list)
    alternatives: list[tuple[Cell, float]] = field(default_factory=list)
    headline: str = ""
    explanation: str = ""


def opening_chance(board: Board, result: SolveResult, cell: Cell) -> float:
    """Heuristic information gain: chance that revealing `cell` shows a 0 and
    cascades open (treats neighbouring mine probabilities as independent)."""
    chance = 1.0
    for nb in board.neighbors(cell):
        state = board[nb]
        if state in (FLAG, MINE):
            return 0.0
        if state == UNREVEALED:
            chance *= 1.0 - result.probs.get(nb, 0.0)
    return chance


def _info_score(board: Board, result: SolveResult, cell: Cell) -> tuple:
    covered_neighbors = sum(board[nb] == UNREVEALED for nb in board.neighbors(cell))
    return (opening_chance(board, result, cell), covered_neighbors)


def recommend(board: Board, result: SolveResult) -> Recommendation:
    if not result.valid:
        return Recommendation(
            kind="invalid",
            headline="Board is inconsistent",
            explanation=(result.error or "") + " Check the highlighted clues, flags and the total mine count.",
        )
    if board.count(MINE):
        return Recommendation(kind="done", headline="Game over",
                              explanation="A mine has been revealed on this board.")
    if not result.probs:
        return Recommendation(kind="done", headline="Nothing left to reveal",
                              explanation="Every covered cell is flagged or revealed.")

    safe, mines = result.safe_cells, result.mine_cells
    rec = Recommendation(kind="safe", safe_cells=safe, mine_cells=mines)

    if safe:
        best = max(safe, key=lambda c: _info_score(board, result, c))
        rec.cell, rec.mine_prob = best, 0.0
        rec.headline = f"Reveal {cell_name(best)} — guaranteed safe"
        parts = [result.reasons.get(best, "")]
        if len(safe) > 1:
            parts.append(
                f"{len(safe)} cells are guaranteed safe in total; {cell_name(best)} is listed "
                f"first because it is the most likely to open up new area."
            )
        if mines:
            parts.append(f"{len(mines)} cell{'s are' if len(mines) != 1 else ' is'} a proven mine and can be flagged.")
        rec.explanation = " ".join(p for p in parts if p)
        return rec

    uncertain = sorted(
        ((c, p) for c, p in result.probs.items() if p < 1.0), key=lambda cp: (cp[1], cp[0])
    )
    if not uncertain:
        rec.kind = "flag"
        rec.headline = "Only mines remain"
        rec.explanation = "Every covered cell left is a proven mine. Flag them to finish."
        return rec

    lowest = uncertain[0][1]
    similar = [c for c, p in uncertain if p - lowest <= SIMILAR_RISK]
    best = max(similar, key=lambda c: (_info_score(board, result, c), (-c[0], -c[1])))
    p = result.probs[best]
    rec.kind = "guess"
    rec.cell, rec.mine_prob = best, p
    rec.alternatives = [(c, q) for c, q in uncertain if c != best][:5]
    rec.headline = f"Best guess: {cell_name(best)} — {p * 100:.1f}% mine risk"

    quality = result.quality.get(best, EXACT)
    parts = [f"No cell is guaranteed safe, so a guess is required. {cell_name(best)} has the "
             f"lowest mine probability on the board ({p * 100:.1f}%, {(1 - p) * 100:.1f}% safe)."]
    if len(similar) > 1:
        parts.append(
            f"{len(similar)} cells are within one percentage point of that risk; "
            f"{cell_name(best)} was preferred because it has the best chance "
            f"({opening_chance(board, result, best) * 100:.0f}%, heuristic) of revealing a 0 and opening new area."
        )
    if quality == ESTIMATED:
        parts.append("This probability is a sampled estimate, not an exact value.")
    elif quality == UNCERTAIN:
        parts.append("Some cells were not recognised, so this probability is approximate.")
    elif quality == PRIOR:
        parts.append("This probability depends on an assumed mine density because the total mine count is unknown.")
    if mines:
        parts.append(f"{len(mines)} cell{'s are' if len(mines) != 1 else ' is'} a proven mine and can be flagged first.")
    rec.explanation = " ".join(parts)
    return rec
