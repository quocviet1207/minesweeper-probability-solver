"""Main window behaviour that needs the event loop: background analysis."""
import random
import time

import pytest
from PySide6.QtCore import QTimer

from minesweeper_solver.core import Board, solve
from minesweeper_solver.ui.main_window import MainWindow


def wait(app, condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not condition():
        app.processEvents()
        time.sleep(0.005)
    return condition()


def hard_board(seed=3):
    """Scattered reveals on a big dense board: one region too large to enumerate."""
    rng = random.Random(seed)
    board = Board(30, 50, 300)
    cells = list(board.all_cells())
    mines = set(rng.sample(cells, 300))
    for c in rng.sample([c for c in cells if c not in mines], int(0.35 * 1200)):
        board[c] = sum(n in mines for n in board.neighbors(c))
    return board


@pytest.fixture
def window(app):
    win = MainWindow()
    yield win
    win.shutdown()
    win.deleteLater()


def settle(app):
    for _ in range(5):
        app.processEvents()
        time.sleep(0.01)


def test_window_can_be_made_small(app, window):
    window.show()
    settle(app)
    screen = app.primaryScreen().availableGeometry()
    assert window.width() <= screen.width() and window.height() <= screen.height()
    assert window.minimumSizeHint().width() <= 600          # all three panes showing
    window.resize(900, 600)
    settle(app)
    assert window.size().toTuple() == (900, 600)
    window.left_toggle.setChecked(False)
    window.right_toggle.setChecked(False)
    settle(app)
    assert window.left_pane.isHidden() and window.right_pane.isHidden()
    assert window.minimumSizeHint().width() <= 300          # board only
    window.resize(400, 500)
    settle(app)
    assert window.width() <= 400
    window.right_toggle.setChecked(True)
    settle(app)
    assert not window.right_pane.isHidden() and window.splitter.sizes()[2] > 0


def test_dragging_a_pane_shut_updates_its_button(app, window):
    window.show()
    settle(app)
    sizes = window.splitter.sizes()
    window.splitter.setSizes([0, sizes[0] + sizes[1], sizes[2]])
    window._sync_pane_buttons()
    assert not window.left_toggle.isChecked()
    window.left_toggle.setChecked(True)                      # reopens it
    settle(app)
    assert window.splitter.sizes()[0] > 0


def test_window_size_and_panes_are_remembered(app, tmp_path):
    from PySide6.QtCore import QSettings
    settings = QSettings(str(tmp_path / "w.ini"), QSettings.IniFormat)
    first = MainWindow(settings)
    first.show()
    settle(app)
    first.resize(1000, 650)
    first.right_toggle.setChecked(False)
    settle(app)
    first.close()
    settle(app)
    second = MainWindow(settings)
    try:
        second.show()
        settle(app)
        assert second.size().toTuple() == (1000, 650)
        assert not second.right_toggle.isChecked()
    finally:
        second.shutdown()


def test_analysis_arrives_and_matches_a_direct_solve(app, window):
    assert wait(app, lambda: window.result is not None)
    expected = solve(window.board)
    assert window.result.probs == expected.probs
    assert window.recommendation is not None


def test_only_the_latest_board_is_shown(app, window):
    assert wait(app, lambda: window.result is not None)
    boards = [Board.from_strings(lines, 3) for lines in (
        ["....", ".1..", "....", "...."],
        ["....", ".2..", "....", "...."],
        ["....", ".11.", "....", "...."],
    )]
    for board in boards:                 # edits faster than analysis can finish
        window._replace_board(board)
        window._solve()
    assert wait(app, lambda: window.result is not None
                and window.result.probs == solve(boards[-1]).probs)
    window.analyse_now()                 # the synchronous path agrees
    assert window.result.probs == solve(boards[-1]).probs


def test_window_stays_responsive_while_a_hard_board_is_solved(app, window):
    assert wait(app, lambda: window.result is not None)
    gaps, last = [], [time.perf_counter()]

    def tick():
        now = time.perf_counter()
        gaps.append(now - last[0])
        last[0] = now

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(10)
    window._replace_board(hard_board())
    window._solve()
    started = time.perf_counter()
    assert wait(app, lambda: window.result is not None and window.result.estimated, timeout=30)
    elapsed = time.perf_counter() - started
    timer.stop()
    assert elapsed > 0.3                       # a genuinely slow solve ...
    gaps = sorted(gaps[1:])
    # ... during which the UI kept running: typical ticks stay near 10 ms and
    # no pause comes anywhere near the length of the solve (some slack for a
    # busy test machine)
    assert gaps[len(gaps) // 2] < 0.05
    assert gaps[-1] < max(0.4, elapsed / 3)
    assert window.analysis_state.text() == ""  # busy indicator cleared
