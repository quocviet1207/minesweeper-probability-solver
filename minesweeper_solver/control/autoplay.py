"""Which move to make next, and where on the screen it is."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from ..core.board import FLAG, MINE, UNREVEALED, Board, Cell, cell_name
from ..core.recommend import Recommendation
from ..core.solver import SolveResult

REVEAL, FLAG_CELL = "reveal", "flag"


@dataclass
class Move:
    cell: Cell
    action: str                 # REVEAL or FLAG_CELL
    reason: str
    risk: float = 0.0           # mine probability of the cell (only a guess is > 0)

    @property
    def button(self) -> str:
        return "left" if self.action == REVEAL else "right"


@dataclass
class Stop:
    reason: str
    finished: bool = False      # the game is over (won or lost), not just stuck


def choose_move(board: Board, result: SolveResult, rec: Recommendation,
                guess: bool = False, flag_mines: bool = False) -> Union[Move, Stop]:
    """Next move for auto-play. Only proven-safe cells are revealed unless
    `guess` is set; proven mines are flagged only if `flag_mines` is set."""
    if board.count(MINE):
        return Stop("Game over: a mine was revealed.", finished=True)
    if not result.valid:
        return Stop("The board reads as inconsistent, so auto-play will not act on it. "
                    + (result.error or ""))
    covered = [c for c in board.all_cells() if board[c] == UNREVEALED]
    if not covered:
        return Stop("Every cell is revealed or flagged.", finished=True)
    if result.safe_cells:
        cell = rec.cell if rec.kind == "safe" and rec.cell in result.safe_cells else result.safe_cells[0]
        return Move(cell, REVEAL, f"{cell_name(cell)} is guaranteed safe")
    undecided = [c for c, p in result.probs.items() if 0 < p < 1]
    if not undecided:                                   # only proven mines are left
        if flag_mines and result.mine_cells:
            cell = result.mine_cells[0]
            return Move(cell, FLAG_CELL, f"{cell_name(cell)} is a proven mine")
        return Stop("Solved: every remaining covered cell is a mine.", finished=True)
    if flag_mines and result.mine_cells:
        cell = result.mine_cells[0]
        return Move(cell, FLAG_CELL, f"{cell_name(cell)} is a proven mine")
    if guess and rec.kind == "guess" and rec.cell is not None:
        p = result.probs.get(rec.cell, 1.0)
        return Move(rec.cell, REVEAL,
                    f"no safe cell left, guessing {cell_name(rec.cell)} ({p * 100:.1f}% mine risk)", p)
    return Stop("No guaranteed-safe move left. Make the next move yourself (the best guess is "
                f"{cell_name(rec.cell)}), or allow guessing in Settings." if rec.cell is not None
                else "No guaranteed-safe move left.")


def cell_center(region_left: int, region_top: int, x_lines, y_lines, cell: Cell) -> tuple[int, int]:
    """Screen position (physical pixels) of the middle of `cell`, given the
    capture region's origin and the grid lines found in the captured frame."""
    r, c = cell
    x = region_left + (x_lines[c] + x_lines[c + 1]) / 2
    y = region_top + (y_lines[r] + y_lines[r + 1]) / 2
    return int(round(x)), int(round(y))
