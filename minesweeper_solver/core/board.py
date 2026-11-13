"""Board model shared by every module.

Cell states are plain ints so boards are cheap to copy, diff and serialise:

    0..8        revealed cell showing that many adjacent mines
    UNREVEALED  covered cell, nothing known
    FLAG        covered cell the player has flagged (treated as a mine)
    MINE        revealed / exploded mine (game over; treated as a mine)
    UNKNOWN     recognition could not tell what the cell is. The solver treats
                it as a covered cell that may or may not hold a mine, which is
                sound whatever the cell really is (covered, flag or number).
"""
from __future__ import annotations

import random
from typing import Iterator, Optional

UNREVEALED = -1
FLAG = -2
MINE = -3
UNKNOWN = -4

Cell = tuple[int, int]  # (row, col), zero based

_CHARS = {UNREVEALED: ".", FLAG: "F", MINE: "*", UNKNOWN: "?"}
_FROM_CHARS = {v: k for k, v in _CHARS.items()}


def cell_name(cell: Cell) -> str:
    """Spreadsheet-style name: column letters + 1-based row, e.g. (0, 0) -> 'A1'."""
    r, c = cell
    letters = ""
    c += 1
    while c:
        c, rem = divmod(c - 1, 26)
        letters = chr(ord("A") + rem) + letters
    return f"{letters}{r + 1}"


class Board:
    def __init__(self, rows: int, cols: int, total_mines: Optional[int] = None):
        if rows < 1 or cols < 1:
            raise ValueError("board must have at least one row and one column")
        self.rows = rows
        self.cols = cols
        self.total_mines = total_mines  # None = unknown
        self.cells = [[UNREVEALED] * cols for _ in range(rows)]

    # -- access ---------------------------------------------------------
    def __getitem__(self, cell: Cell) -> int:
        return self.cells[cell[0]][cell[1]]

    def __setitem__(self, cell: Cell, value: int) -> None:
        if not (UNKNOWN <= value <= 8):
            raise ValueError(f"invalid cell state {value}")
        self.cells[cell[0]][cell[1]] = value

    def in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.rows and 0 <= cell[1] < self.cols

    def all_cells(self) -> Iterator[Cell]:
        for r in range(self.rows):
            for c in range(self.cols):
                yield (r, c)

    def neighbors(self, cell: Cell) -> list[Cell]:
        r, c = cell
        return [
            (rr, cc)
            for rr in range(max(0, r - 1), min(self.rows, r + 2))
            for cc in range(max(0, c - 1), min(self.cols, c + 2))
            if (rr, cc) != (r, c)
        ]

    def count(self, state: int) -> int:
        return sum(row.count(state) for row in self.cells)

    def copy(self) -> "Board":
        b = Board(self.rows, self.cols, self.total_mines)
        b.cells = [row[:] for row in self.cells]
        return b

    def resized(self, rows: int, cols: int) -> "Board":
        """New board of the given size keeping the overlapping top-left cells."""
        b = Board(rows, cols, self.total_mines)
        for r in range(min(rows, self.rows)):
            for c in range(min(cols, self.cols)):
                b.cells[r][c] = self.cells[r][c]
        return b

    # -- text form (tests, debugging, save files) -------------------------
    @classmethod
    def from_strings(cls, lines: list[str], total_mines: Optional[int] = None) -> "Board":
        """'.' unrevealed, 'F' flag, '*' mine, '?' unknown, '0'-'8' numbers (' ' also means 0)."""
        lines = [ln for ln in lines if ln != ""]
        b = cls(len(lines), len(lines[0]), total_mines)
        for r, line in enumerate(lines):
            if len(line) != b.cols:
                raise ValueError("all rows must have the same length")
            for c, ch in enumerate(line):
                if ch in _FROM_CHARS:
                    b.cells[r][c] = _FROM_CHARS[ch]
                elif ch == " ":
                    b.cells[r][c] = 0
                else:
                    b.cells[r][c] = int(ch)
        return b

    def to_strings(self) -> list[str]:
        return ["".join(_CHARS.get(v, str(v)) for v in row) for row in self.cells]

    def __repr__(self) -> str:
        return "Board(\n  " + "\n  ".join(self.to_strings()) + f"\n  mines={self.total_mines})"


def random_position(rows: int, cols: int, mines: int, seed: Optional[int] = None) -> Board:
    """A plausible mid-game position: random mines, then one opening click
    (flood-filled like a real game). Used for demos and tests."""
    rng = random.Random(seed)
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    board = Board(rows, cols, mines)
    for _ in range(200):
        mine_set = set(rng.sample(cells, mines))

        def number(cell: Cell) -> int:
            return sum(n in mine_set for n in board.neighbors(cell))

        zeros = [c for c in cells if c not in mine_set and number(c) == 0]
        if not zeros:
            continue
        center = (rows / 2, cols / 2)
        start = min(zeros, key=lambda c: (c[0] - center[0]) ** 2 + (c[1] - center[1]) ** 2)
        stack, seen = [start], {start}
        while stack:
            cur = stack.pop()
            n = number(cur)
            board[cur] = n
            if n == 0:
                for nb in board.neighbors(cur):
                    if nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        return board
    return board
