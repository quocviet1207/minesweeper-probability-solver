"""Game state manager: merging readings, change reports, new games."""
import numpy as np

from minesweeper_solver.core import FLAG, MINE, UNKNOWN, UNREVEALED, Board
from minesweeper_solver.state import (CHANGED, FIRST, NEW_GAME, RESIZED, UNCHANGED,
                                      GameStateManager)
from minesweeper_solver.state.manager import HOLD_FRAMES


def grid(*lines):
    return np.array(Board.from_strings(list(lines)).cells)


START = grid("....", ".21.", ".10.", "....")


def test_first_reading_and_standard_mine_counts():
    m = GameStateManager()
    up = m.apply(START, total_mines=7)
    assert up.event == FIRST and len(up.changed) == 16 and up.board.total_mines == 7
    up = m.apply(np.full((9, 9), UNREVEALED))
    assert up.event == RESIZED and up.board.total_mines == 10      # beginner
    m.reset()
    assert m.apply(np.full((16, 30), UNREVEALED), total_mines=3).board.total_mines == 99
    m.reset()
    assert m.apply(np.full((5, 5), UNREVEALED), total_mines=40).board.total_mines is None  # impossible


def test_reports_exactly_the_cells_that_changed():
    m = GameStateManager()
    m.apply(START, 5)
    assert m.apply(START.copy()).event == UNCHANGED
    nxt = grid("....", ".21F", "110.", "....")
    up = m.apply(nxt)
    assert up.event == CHANGED
    assert sorted(up.changed) == [(1, 3), (2, 0)]
    assert (up.revealed, up.flagged, up.unflagged) == (1, 1, 0)
    assert up.board.total_mines == 5 and up.board.cells == nxt.tolist()
    up = m.apply(grid("....", ".21.", "110.", "...."))
    assert up.unflagged == 1 and up.changed == [(1, 3)]
    assert "1 unflagged" in up.summary()


def test_revealed_cells_survive_glitches_but_not_forever():
    m = GameStateManager()
    m.apply(START, 5)
    glitch = START.copy()
    glitch[1, 1] = UNKNOWN          # hover over the "2"
    glitch[2, 2] = UNREVEALED       # one cell misread as covered
    glitch[0, 0] = FLAG             # and a real change elsewhere
    up = m.apply(glitch)
    assert up.changed == [(0, 0)] and sorted(up.held) == [(1, 1), (2, 2)]
    assert up.board[(1, 1)] == 2 and up.board[(2, 2)] == 0
    # a reading that keeps contradicting the board is eventually believed
    for _ in range(HOLD_FRAMES - 2):
        assert m.apply(glitch).event == UNCHANGED
    up = m.apply(glitch)
    assert sorted(up.changed) == [(1, 1), (2, 2)] and up.board[(1, 1)] == UNKNOWN
    # and the counter resets once readings agree again
    m.apply(START)
    assert m.apply(glitch).held


def test_new_game_is_detected_not_merged():
    m = GameStateManager()
    m.apply(grid("....", ".21.", ".10.", "....", "...."), 5)
    up = m.apply(np.full((5, 4), UNREVEALED))
    assert up.event == NEW_GAME and up.board.total_mines == 5
    assert all(v == UNREVEALED for row in up.board.cells for v in row)

    big = grid("0001..", "0012..", "112...", "......")
    m.reset()
    m.apply(big, 6)
    restarted = grid("......", "......", "...01.", "...01.")     # new game, first click elsewhere
    up = m.apply(restarted)
    assert up.event == NEW_GAME and up.board.cells == restarted.tolist()


def test_game_over_is_reported():
    m = GameStateManager()
    m.apply(START, 5)
    lost = START.copy()
    lost[0, 0] = MINE
    up = m.apply(lost)
    assert up.event == CHANGED and up.game_over
