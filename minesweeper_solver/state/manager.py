"""Merges each recognised frame into the current board and reports what changed.

Within one game Minesweeper only moves forward: covered cells get revealed or
flagged, flags get removed, and a revealed number never changes. That gives
two jobs beyond copying the latest reading:

* **Stabilising.** If a revealed cell suddenly reads as covered, flagged or
  unknown (a hover effect, a dialog sliding over the board, a misread), the
  number is kept. Only if the new reading persists for several frames is it
  accepted, so an early misread of a covered cell as "revealed" cannot stick.
* **New games.** When a large share of the revealed cells turn back into
  covered cells, or nothing is revealed any more, a new game has started and
  the board is replaced rather than merged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.board import FLAG, MINE, UNREVEALED, Board, Cell

FIRST, CHANGED, UNCHANGED, NEW_GAME, RESIZED = "first", "changed", "unchanged", "new_game", "resized"

STANDARD_MINES = {(9, 9): 10, (16, 16): 40, (16, 30): 99, (30, 16): 99}
HOLD_FRAMES = 3            # readings that contradict a revealed cell before they are believed
NEW_GAME_FRACTION = 0.5    # share of revealed cells turning covered that means "new game"
NEW_GAME_MIN_CELLS = 3


def standard_mines(rows: int, cols: int) -> Optional[int]:
    """Mine count of the classic board sizes, None for anything else."""
    return STANDARD_MINES.get((rows, cols))


@dataclass
class SyncUpdate:
    event: str                                     # FIRST, CHANGED, UNCHANGED, NEW_GAME, RESIZED
    board: Board
    changed: list[Cell] = field(default_factory=list)
    revealed: int = 0                              # cells newly revealed by this update
    flagged: int = 0                               # flags added
    unflagged: int = 0                             # flags removed
    held: list[Cell] = field(default_factory=list) # readings rejected to keep a revealed cell
    game_over: bool = False

    def summary(self) -> str:
        if self.event == UNCHANGED:
            return "no change"
        if self.event in (FIRST, RESIZED):
            return f"{self.board.cols}×{self.board.rows} board read"
        if self.event == NEW_GAME:
            return "new game detected"
        parts = []
        if self.revealed:
            parts.append(f"{self.revealed} revealed")
        if self.flagged:
            parts.append(f"{self.flagged} flagged")
        if self.unflagged:
            parts.append(f"{self.unflagged} unflagged")
        other = len(self.changed) - self.revealed - self.flagged - self.unflagged
        if other > 0:
            parts.append(f"{other} other")
        return f"{len(self.changed)} cell{'s' if len(self.changed) != 1 else ''} changed" + (
            f" ({', '.join(parts)})" if parts else "")


class GameStateManager:
    def __init__(self, hold_frames: int = HOLD_FRAMES):
        self.hold_frames = hold_frames
        self.board: Optional[Board] = None
        self._disagree: Optional[np.ndarray] = None   # frames a held cell has been contradicted

    def reset(self) -> None:
        self.board = None
        self._disagree = None

    def apply(self, states: np.ndarray, total_mines: Optional[int] = None) -> SyncUpdate:
        """Merge a recognised [rows, cols] state array.

        `total_mines` is the count to keep for a board of the current size
        (None = unknown). When the board size changes or a game is first read,
        the standard count for that size is used if there is one.
        """
        new = np.asarray(states, dtype=int)
        rows, cols = new.shape
        previous = self.board
        if previous is None or (previous.rows, previous.cols) != (rows, cols):
            event = FIRST if previous is None else RESIZED
            total = standard_mines(rows, cols)
            if total is None and total_mines is not None and total_mines < rows * cols:
                total = total_mines
            return self._replace(new, total, event)

        old = np.array(previous.cells, dtype=int)
        was_revealed = old >= 0
        reopened = was_revealed & (new == UNREVEALED)
        if reopened.any() and (
                reopened.sum() >= max(NEW_GAME_MIN_CELLS, NEW_GAME_FRACTION * was_revealed.sum())
                or not (new >= 0).any()):
            return self._replace(new, previous.total_mines if total_mines is None else total_mines,
                                 NEW_GAME)

        merged = new.copy()
        # a revealed number does not become covered, flagged, unknown or a mine
        contradicts = was_revealed & (new < 0)
        self._disagree = np.where(contradicts, self._disagree + 1, 0)
        hold = contradicts & (self._disagree < self.hold_frames)
        merged[hold] = old[hold]

        diff = merged != old
        changed = [tuple(int(v) for v in rc) for rc in np.argwhere(diff)]
        board = previous
        board.total_mines = previous.total_mines if total_mines is None else total_mines
        if changed:
            board.cells = merged.tolist()
        return SyncUpdate(
            event=CHANGED if changed else UNCHANGED,
            board=board,
            changed=changed,
            revealed=int((diff & (merged >= 0) & (old < 0)).sum()),
            flagged=int((diff & (merged == FLAG)).sum()),
            unflagged=int((diff & (old == FLAG) & (merged == UNREVEALED)).sum()),
            held=[tuple(int(v) for v in rc) for rc in np.argwhere(hold)],
            game_over=bool((merged == MINE).any()),
        )

    def _replace(self, new: np.ndarray, total: Optional[int], event: str) -> SyncUpdate:
        rows, cols = new.shape
        board = Board(rows, cols, total)
        board.cells = new.tolist()
        self.board = board
        self._disagree = np.zeros(new.shape, int)
        return SyncUpdate(
            event=event,
            board=board,
            changed=[(r, c) for r in range(rows) for c in range(cols)],
            revealed=int((new >= 0).sum()),
            flagged=int((new == FLAG).sum()),
            game_over=bool((new == MINE).any()),
        )
