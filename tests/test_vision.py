"""Recognition on synthetic screenshots (see themes.py for what these are and
are not). The key requirement: a cell may be reported UNKNOWN, never wrong."""
import numpy as np
import pytest

from minesweeper_solver.core import MINE, UNKNOWN, solve
from minesweeper_solver.vision import BoardReader, detect_grid, recognize

from themes import THEMES, make_game, render, render_app_board

# (rows, cols, mines), cell size in px, zoom factor
CONFIGS = [
    ((9, 9, 10), 24, 1.0),
    ((16, 16, 40), 16, 1.0),
    ((16, 30, 99), 20, 1.0),
    ((16, 30, 99), 24, 0.77),    # fractional cell size, as with browser zoom
    ((9, 9, 10), 30, 1.23),
    ((12, 20, 40), 18, 1.5),
    ((16, 30, 99), 32, 1.0),
]
ROUNDS = [0, 1, 3, 6]            # just opened ... nearly solved


@pytest.mark.parametrize("theme", list(THEMES) + ["app"])
def test_theme_is_read_without_wrong_cells(theme, app):
    cells = unknown = 0
    for (rows, cols, mines), cell, scale in CONFIGS:
        for seed, rounds in enumerate(ROUNDS):
            board = make_game(rows, cols, mines, seed, rounds=rounds, lose=(seed == 3))
            frame = render_app_board(board) if theme == "app" else render(board, theme, cell, scale)
            result = recognize(frame)
            where = (theme, rows, cols, cell, scale, seed)
            assert result.ok, (where, result.message)
            assert result.states.shape == (rows, cols), where
            truth = np.array(board.cells)
            mismatch = result.states != truth
            is_unknown = result.states == UNKNOWN
            assert not (mismatch & ~is_unknown).any(), (where, truth[mismatch], result.states[mismatch])
            cells += truth.size
            unknown += int(is_unknown.sum())
            # the recognised board must be usable by the solver as is
            assert solve(result.to_board(mines)).valid, where
    assert unknown <= 0.02 * cells, f"{unknown} of {cells} cells unknown"


def test_grid_position_is_accurate(app):
    for theme in THEMES:
        for scale in (1.0, 0.77, 1.5):
            frame = render(make_game(16, 30, 99, 1), theme, 20, scale)
            grid = detect_grid(frame)
            assert (grid.rows, grid.cols) == (16, 30)
            cell = 20 * scale
            assert abs(grid.x_lines[0] - 14 * scale) <= 0.1 * cell, (theme, scale)
            assert abs(grid.y_lines[0] - 60 * scale) <= 0.1 * cell, (theme, scale)
            assert abs(grid.cell_size[0] - cell) <= 0.01 * cell, (theme, scale)
            assert abs(grid.x_lines[-1] - (14 + 30 * 20) * scale) <= 0.1 * cell, (theme, scale)


def test_exploded_mine_is_reported(app):
    for theme in THEMES:
        board = make_game(9, 9, 10, 3, rounds=2, lose=True)
        assert board.count(MINE) == 1
        result = recognize(render(board, theme, 24))
        assert result.ok and (result.states == MINE).sum() == 1, theme


def test_images_without_a_board_are_rejected(app):
    rng = np.random.default_rng(0)
    blank = np.full((300, 400, 3), 200, np.uint8)
    noise = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
    tiny = np.zeros((10, 10, 3), np.uint8)
    text = blank.copy()
    text[40:60, 30:370] = 20
    text[100:104, 30:200] = 20
    for image in (blank, noise, tiny, text):
        result = recognize(image)
        assert not result.ok and result.message


def test_reader_reuses_grid_and_recovers_when_the_board_moves(app):
    reader = BoardReader()
    board = make_game(16, 16, 40, 1, rounds=1)
    frame = render(board, "classic", 20)
    first = reader.read(frame)
    assert first.ok and not first.reused_grid

    later = make_game(16, 16, 40, 1, rounds=4)
    second = reader.read(render(later, "classic", 20))
    assert second.ok and second.reused_grid
    assert (second.states == np.array(later.cells)).all()

    moved = np.roll(render(later, "classic", 20), (7, 9), axis=(0, 1))   # window dragged
    third = reader.read(moved)
    assert third.ok and not third.reused_grid
    readable = third.states != UNKNOWN
    assert readable.mean() > 0.98
    assert (third.states == np.array(later.cells))[readable].all()

    assert not reader.read(np.full_like(frame, 128)).ok                  # game closed
    assert reader.read(frame).ok                                         # and back again


@pytest.mark.parametrize("size,cell", [
    ((30, 4, 20), 20),      # tall and narrow: few column lines to measure the period
    ((4, 30, 20), 20),      # short and wide
    ((5, 5, 3), 24),        # smaller than the counter panel above it
    ((9, 9, 10), 16),       # counter panel edge falls exactly on a lattice line
    ((30, 50, 300), 16),    # large custom board
    ((9, 9, 10), 64),       # very large cells
])
def test_unusual_board_and_cell_sizes(app, size, cell):
    rows, cols, mines = size
    for theme in THEMES:
        for seed in (1, 2):
            board = make_game(rows, cols, mines, seed, rounds=2 + seed)
            result = recognize(render(board, theme, cell))
            assert result.ok and result.states.shape == (rows, cols), (theme, seed, result.message)
            truth = np.array(board.cells)
            assert not ((result.states != truth) & (result.states != UNKNOWN)).any(), (theme, seed)


def test_tiny_cells_are_refused_not_misread(app):
    for theme in THEMES:
        for cell in (8, 10, 12):
            result = recognize(render(make_game(16, 30, 99, 1, rounds=3), theme, cell))
            assert not result.ok, (theme, cell)
            if result.grid is not None:
                assert "too small" in result.message


def test_board_cut_off_by_the_region_is_flagged(app):
    board = make_game(9, 9, 10, 1, rounds=1)
    frame = render(board, "classic", 24)
    assert not recognize(frame).clipped
    cut = frame[:, : 14 + 24 * 7 + 10]              # region ends inside column 8
    result = recognize(cut)
    assert result.ok and result.clipped and result.states.shape[1] < 9
    assert "edge of the capture region" in result.message
