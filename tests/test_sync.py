"""Live synchronisation: games played move by move and read back from screenshots."""
import random
import time

import numpy as np
import pytest

from minesweeper_solver.core import FLAG, UNKNOWN, UNREVEALED, Board, SolverCache, solve
from minesweeper_solver.state import CHANGED, FIRST, UNCHANGED, GameStateManager
from minesweeper_solver.vision import BoardReader, Recognition
from minesweeper_solver.vision.cells import resample_board
from minesweeper_solver.vision.recognizer import changed_cells

from themes import make_game, render


@pytest.mark.parametrize("theme", ["classic", "flat"])
def test_game_followed_move_by_move(app, theme):
    """A player follows the solver's advice, but the solver only ever sees what
    was recognised from the screenshots. Every proven move must be right."""
    rng = random.Random(3)
    rows, cols, mines = 16, 16, 40
    board = Board(rows, cols, mines)
    forbidden = {(r, c) for r in range(6, 11) for c in range(6, 11)}
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

    reveal((8, 8))
    reader, sync, cache = BoardReader(), GameStateManager(), SolverCache()
    moves = 0
    previous = None
    for _ in range(80):
        frame = render(board, theme, 20)
        result = reader.read(frame)
        assert result.ok, result.message
        update = sync.apply(result.states, mines)
        assert update.event in (FIRST, CHANGED, UNCHANGED)
        truth = np.array(board.cells)
        seen = np.array(update.board.cells)
        assert not ((seen != truth) & (seen != UNKNOWN)).any(), (moves, np.argwhere(seen != truth))
        if previous is not None:            # only cells that really changed are reported
            assert set(update.changed) <= {tuple(rc) for rc in np.argwhere(truth != previous)}
        previous = truth
        analysis = solve(update.board, cache=cache)
        assert analysis.valid

        again = reader.read(frame)          # the same picture again: nothing is re-read
        assert again.skipped and sync.apply(again.states, mines).event == UNCHANGED

        if analysis.mine_cells and rng.random() < 0.4:
            cell = analysis.mine_cells[0]
            assert cell in mine_set
            board[cell] = FLAG
        elif analysis.safe_cells:
            cell = analysis.safe_cells[0]
            assert cell not in mine_set
            reveal(cell)
        else:
            break
        moves += 1
    assert moves >= 10
    assert cache.hits > 0
    assert reader.detections == 1           # the grid was found once and reused


def test_reader_skips_changes_outside_the_board(app):
    board = make_game(9, 9, 10, 2, rounds=2)
    frame = render(board, "classic", 24)
    reader = BoardReader()
    first = reader.read(frame)
    assert first.ok and not first.skipped and first.changed.all()

    ticking = frame.copy()
    ticking[12:40, 14:70] = 255 - ticking[12:40, 14:70]     # the timer in the header changes
    r = reader.read(ticking)
    assert r.skipped and not r.changed.any()
    assert (r.states == first.states).all()

    covered = [c for c in board.all_cells() if board[c] == UNREVEALED]
    board[covered[0]] = FLAG
    r = reader.read(render(board, "classic", 24))
    assert not r.skipped and r.changed.sum() == 1 and r.changed[covered[0]]
    assert r.states[covered[0]] == FLAG
    assert reader.detections == 1


def test_wrong_flags_inside_an_opened_area_are_read_correctly(app):
    """A player can flag a cell that is not a mine; the game then opens
    everything around it, so revealed blanks end up next to a flag."""
    for theme in ("classic", "flat", "blue", "mono"):
        board = make_game(16, 16, 40, 2, rounds=3)
        zeros = [c for c in board.all_cells() if board[c] == 0
                 and all(board[n] >= 0 for n in board.neighbors(c))]
        for cell in zeros[:3]:
            board[cell] = FLAG
        result = BoardReader().read(render(board, theme, 22))
        truth = np.array(board.cells)
        assert result.ok
        assert not ((result.states != truth) & (result.states != UNKNOWN)).any(), theme


def test_an_earlier_misreading_does_not_outlive_its_frame(app):
    board = make_game(9, 9, 10, 2, rounds=2)
    reader = BoardReader()
    reader.read(render(board, "classic", 24))
    revealed = [c for c in board.all_cells() if board[c] >= 0][:5]
    for cell in revealed:                       # pretend the last frame misread these
        reader._last.states[cell] = FLAG
    covered = [c for c in board.all_cells() if board[c] == UNREVEALED]
    board[covered[0]] = FLAG                    # something changes elsewhere
    result = reader.read(render(board, "classic", 24))
    assert not result.skipped
    assert all(result.states[c] == board[c] for c in revealed)


def test_cell_change_detection_ignores_noise(app):
    board = make_game(9, 9, 10, 1, rounds=1)
    frame = render(board, "flat", 24)
    grid = BoardReader().read(frame).grid
    a = resample_board(frame, grid)
    assert not changed_cells(a, a.copy()).any()
    noisy = frame.astype(np.int16) + np.random.default_rng(0).integers(-2, 3, frame.shape)
    b = resample_board(np.clip(noisy, 0, 255).astype(np.uint8), grid)
    assert not changed_cells(a, b).any()                     # compression-like noise


class _SlowReader:
    def reset(self):
        pass

    def read(self, frame):
        time.sleep(0.05)
        return Recognition(False, str(int(frame[0, 0, 0])))


def _wait(app, condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not condition():
        app.processEvents()
        time.sleep(0.01)
    return condition()


def test_worker_keeps_only_the_newest_frame(app):
    from minesweeper_solver.ui.recognition_worker import RecognitionWorker

    worker = RecognitionWorker(reader_factory=_SlowReader)
    results = []
    worker.resultReady.connect(lambda r, g: results.append((r.message, g)))
    worker.start()
    for value in range(10):
        worker.submit(np.full((2, 2, 3), value, np.uint8))
    assert _wait(app, lambda: results and results[-1][0] == "9")   # newest frame always read
    assert len(results) < 10 and worker.dropped > 0                # stale ones skipped

    worker.reset()
    worker.submit(np.full((2, 2, 3), 42, np.uint8))
    assert _wait(app, lambda: results[-1][0] == "42")
    assert results[-1][1] == worker.generation
    worker.stop()
    assert not worker.isRunning()
