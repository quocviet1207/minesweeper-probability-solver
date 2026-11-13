"""Digit templates from real samples and from teaching in the app."""
import numpy as np
import pytest

import themes
from minesweeper_solver.core import UNKNOWN
from minesweeper_solver.vision import BoardReader, glyphs, recognize
from minesweeper_solver.vision.cells import classify_cells


@pytest.fixture
def taught(tmp_path):
    path = tmp_path / "taught.npz"
    glyphs.set_user_samples_file(path)
    yield path
    glyphs.set_user_samples_file(None)          # module-level state: restore


def decorative(seed, font="MV Boli"):
    """A board drawn with a decorative font whose digits look like other digits."""
    board = themes.make_game(16, 16, 40, seed, rounds=3)
    spec = themes.THEMES["classic"]
    saved, spec["font"] = spec["font"], font
    try:
        return board, themes.render(board, "classic", 24)
    finally:
        spec["font"] = saved


def misread(board, states, unknown_too=True):
    truth = np.array(board.cells)
    digits = truth > 0
    off = digits & (states != truth)
    if not unknown_too:
        off &= states != UNKNOWN
    return int(off.sum())


def test_sample_store(app, taught):
    base = len(glyphs.load_templates()[1])
    mask = np.zeros((glyphs.CANVAS, glyphs.CANVAS), np.float32)
    mask[3:17, 8:12] = 1
    assert glyphs.add_user_samples(mask[None], [1]) == 1
    assert glyphs.add_user_samples(mask[None], [1]) == 0          # duplicate
    assert glyphs.user_sample_count() == 1 and taught.exists()
    assert len(glyphs.load_templates()[1]) == base + 1
    glyphs.clear_user_samples()
    assert glyphs.user_sample_count() == 0 and len(glyphs.load_templates()[1]) == base


def test_teaching_a_few_cells_fixes_a_font_it_could_not_read(app, taught):
    boards = [decorative(seed) for seed in (4, 5, 6)]
    assert sum(misread(b, recognize(f).states) for b, f in boards) > 50

    board, frame = decorative(1)                 # the user corrects one cell per digit
    result = recognize(frame)
    masks = classify_cells(frame, result.grid).features["masks"]
    truth = np.array(board.cells)
    for digit in range(1, 9):
        bad = np.argwhere((truth == digit) & (result.states != digit))
        if len(bad):
            glyphs.add_user_samples(masks[tuple(bad[0])][None], [digit])
    assert glyphs.user_sample_count() >= 1

    after = [recognize(f).states for _, f in boards]                     # other boards
    assert sum(misread(b, s, unknown_too=False) for (b, _), s in zip(boards, after)) == 0
    assert sum(misread(b, s) for (b, _), s in zip(boards, after)) <= 2
    for theme in ("classic", "flat"):            # normal styles are unaffected
        b = themes.make_game(16, 30, 99, 2, rounds=3)
        r = recognize(themes.render(b, theme, 22))
        truth = np.array(b.cells)
        assert not ((r.states != truth) & (r.states != UNKNOWN)).any()


def test_settings_dialog_shows_and_forgets_taught_digits(app, taught):
    from minesweeper_solver.ui.preferences import Preferences
    from minesweeper_solver.ui.settings_dialog import SettingsDialog

    mask = np.zeros((glyphs.CANVAS, glyphs.CANVAS), np.float32)
    mask[2:18, 4:16] = 1
    glyphs.add_user_samples(mask[None], [5])
    dialog = SettingsDialog(Preferences())
    assert dialog.taught.text() == "1 taught shape" and dialog.forget.isEnabled()
    dialog.forget.click()
    assert glyphs.user_sample_count() == 0
    assert dialog.taught.text() == "0 taught shapes" and not dialog.forget.isEnabled()


def test_typing_a_digit_on_an_unread_cell_teaches_it(app, taught):
    from minesweeper_solver.ui.main_window import MainWindow

    win = MainWindow(taught_digits=str(taught))
    try:
        board, frame = decorative(1)
        result = BoardReader().read(frame)
        truth = np.array(board.cells)
        cell = tuple(int(v) for v in np.argwhere((truth > 0) & (result.states != truth))[0])
        win.capture_panel._last_frame = frame
        win._on_recognition(result, win.worker.generation)
        submitted = win.worker.submitted
        win.board_widget._select(cell)
        win.board_widget.digitTyped.emit(cell, int(truth[cell]))
        assert glyphs.user_sample_count() == 1
        assert win.worker.submitted == submitted + 1            # board read again
        assert "Learned" in win.statusBar().currentMessage()
        # a cell that was already read correctly teaches nothing
        right = tuple(int(v) for v in np.argwhere((truth > 0) & (result.states == truth))[0])
        win.board_widget.digitTyped.emit(right, int(truth[right]))
        assert glyphs.user_sample_count() == 1
    finally:
        win.shutdown()
