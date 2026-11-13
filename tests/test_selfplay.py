"""The solver plays whole games against a hidden mine layout (no screen involved).

Every cell the solver calls safe must be safe, every cell it calls a mine must
be a mine, and the probabilities must always sum to the remaining mine count.
"""
import random

import pytest

from minesweeper_solver.core import UNREVEALED, Board, recommend, solve


def play(rows, cols, mines, seed):
    rng = random.Random(seed)
    board = Board(rows, cols, mines)
    start = (rows // 2, cols // 2)
    forbidden = set(board.neighbors(start)) | {start}
    mine_set = set(rng.sample([c for c in board.all_cells() if c not in forbidden], mines))

    def reveal(cell):
        stack = [cell]
        while stack:
            cur = stack.pop()
            if board[cur] != UNREVEALED:
                continue
            board[cur] = sum(n in mine_set for n in board.neighbors(cur))
            if board[cur] == 0:
                stack.extend(board.neighbors(cur))

    reveal(start)
    while True:
        result = solve(board)
        assert result.valid, (board, result.error)
        assert not set(result.safe_cells) & mine_set, board
        assert set(result.mine_cells) <= mine_set, board
        if not result.estimated:
            assert sum(result.probs.values()) == pytest.approx(mines, abs=1e-6)
        rec = recommend(board, result)
        if rec.kind == "safe":
            for cell in result.safe_cells:
                reveal(cell)
        elif rec.kind == "guess":
            if rec.cell in mine_set:
                return False
            reveal(rec.cell)
        else:
            return True


@pytest.mark.parametrize("rows,cols,mines,games", [(9, 9, 10, 40), (16, 16, 40, 15), (16, 30, 99, 8)])
def test_solver_never_makes_a_false_claim(rows, cols, mines, games):
    wins = sum(play(rows, cols, mines, seed) for seed in range(games))
    assert wins > 0
