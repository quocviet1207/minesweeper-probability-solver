"""Auto-play end to end against a simulated game and a fake mouse.

The game is drawn with themes.render and read back through the real
recognition pipeline; the fake mouse turns each click (screen coordinates)
back into a cell of the simulated game. No real mouse is touched.
"""
import random
import time

import numpy as np
import pytest

import themes
from minesweeper_solver.capture import Region
from minesweeper_solver.control import FLAG_CELL, REVEAL, Move, Stop, choose_move
from minesweeper_solver.core import FLAG, MINE, UNREVEALED, Board, recommend, solve
from minesweeper_solver.ui import autoplay_controller
from minesweeper_solver.ui.main_window import MainWindow
from minesweeper_solver.ui.preferences import Preferences
from minesweeper_solver.vision import BoardReader

CELL = 24
ORIGIN = (14, 60)            # where themes.render puts the board inside the frame


class Game:
    def __init__(self, rows=9, cols=9, mines=10, seed=0, react=True):
        rng = random.Random(seed)
        self.board = Board(rows, cols, mines)
        start = (rows // 2, cols // 2)
        safe_zone = set(self.board.neighbors(start)) | {start}
        self.mines = set(rng.sample([c for c in self.board.all_cells() if c not in safe_zone], mines))
        self.react = react
        self.lost = False
        self.reveal(start)

    def reveal(self, cell):
        if cell in self.mines:
            self.board[cell] = MINE
            self.lost = True
            return
        stack = [cell]
        while stack:
            cur = stack.pop()
            if self.board[cur] != UNREVEALED:
                continue
            self.board[cur] = sum(n in self.mines for n in self.board.neighbors(cur))
            if self.board[cur] == 0:
                stack.extend(self.board.neighbors(cur))

    def frame(self):
        return themes.render(self.board, "classic", CELL)


class FakeMouse:
    def __init__(self, game, region):
        self.game, self.region = game, region
        self.pos = (0, 0)
        self.clicks = []

    def position(self):
        return self.pos

    def click(self, x, y, button="left"):
        self.pos = (x, y)
        fx, fy = x - self.region.left, y - self.region.top
        cell = (int((fy - ORIGIN[1]) // CELL), int((fx - ORIGIN[0]) // CELL))
        self.clicks.append((cell, button))
        if not self.game.react or not self.game.board.in_bounds(cell):
            return
        if button == "left":
            self.game.reveal(cell)
        elif self.game.board[cell] == UNREVEALED:
            self.game.board[cell] = FLAG


def wait(app, condition, timeout=8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not condition():
        app.processEvents()
        time.sleep(0.005)
    return condition()


@pytest.fixture
def setup(app):
    windows = []

    def make(game, prefs=None, region=Region(1000, 500, 300, 320)):
        mouse = FakeMouse(game, region)
        win = MainWindow(mouse=mouse)
        win.apply_preferences(prefs or Preferences(autoplay_delay_ms=0), save=False)
        controller = win.capture_panel.controller
        controller._region = region            # pretend capture is running on this region
        controller.state = "running"
        reader = BoardReader()

        def show_screen():                     # what the capture + recognition would deliver
            win.capture_panel._last_frame = game.frame()
            win._on_recognition(reader.read(game.frame()), win.worker.generation)

        windows.append(win)
        return win, mouse, show_screen

    yield make
    for win in windows:
        win.autoplay.stop()
        win.shutdown()


def run(app, win, game, mouse, show_screen, limit=200):
    """Play until auto-play stops: after each click, show the new screen."""
    show_screen()
    assert wait(app, win.analysis_current)
    assert win.autoplay.start()
    seen = 0
    for _ in range(limit):
        if not wait(app, lambda: len(mouse.clicks) > seen or not win.autoplay.running):
            break
        if not win.autoplay.running:
            break
        seen = len(mouse.clicks)
        show_screen()
    return win.autoplay


def test_safe_only_auto_play_never_clicks_a_mine(app, setup):
    for seed in range(3):
        game = Game(seed=seed)
        win, mouse, show = setup(game)
        auto = run(app, win, game, mouse, show)
        assert not game.lost
        assert mouse.clicks and all(cell not in game.mines for cell, _ in mouse.clicks)
        assert not auto.running
        text = win.autoplay_label.text()
        assert "No guaranteed-safe move" in text or "every remaining covered cell is a mine" in text.lower() \
            or "Every cell" in text


def test_guessing_plays_a_game_to_the_end(app, setup):
    finished = 0
    for seed in range(4):
        game = Game(seed=10 + seed)
        win, mouse, show = setup(game, Preferences(autoplay_delay_ms=0, autoplay_guess=True,
                                                   autoplay_flag=True))
        run(app, win, game, mouse, show, limit=300)
        text = win.autoplay_label.text()
        assert not win.autoplay.running
        assert "Game over" in text or "Solved" in text or "Every cell" in text, text
        finished += 1
        if not game.lost:                       # won: every mine flagged, nothing else covered
            assert all(game.board[c] == FLAG for c in game.mines)
    assert finished == 4


def test_moving_the_mouse_hands_control_back(app, setup):
    game = Game(seed=1)
    win, mouse, show = setup(game)
    show()
    assert wait(app, win.analysis_current)
    win.autoplay.start()
    assert wait(app, lambda: len(mouse.clicks) == 1)
    mouse.pos = (mouse.pos[0] + 40, mouse.pos[1])    # the user grabs the mouse
    show()
    assert wait(app, lambda: not win.autoplay.running)
    assert len(mouse.clicks) == 1 and "moved the mouse" in win.autoplay_label.text()
    assert not win.autoplay_button.isChecked()


def test_a_click_without_effect_stops_auto_play(app, setup, monkeypatch):
    monkeypatch.setattr(autoplay_controller, "CLICK_TIMEOUT_S", 0.3)
    game = Game(seed=2, react=False)                  # e.g. the game window is covered
    win, mouse, show = setup(game)
    show()
    assert wait(app, win.analysis_current)
    win.autoplay.start()
    assert wait(app, lambda: not win.autoplay.running, timeout=3)
    assert len(mouse.clicks) == 1 and "did not react" in win.autoplay_label.text()


def test_does_not_start_without_live_capture(app):
    win = MainWindow(mouse=FakeMouse(None, Region(0, 0, 10, 10)))
    try:
        assert not win.autoplay.start()
        assert "capture" in win.autoplay_label.text().lower()
        win.autoplay_button.setChecked(True)          # the button falls back to off
        assert not win.autoplay_button.isChecked()
    finally:
        win.shutdown()


def test_never_clicks_on_its_own_window(app, setup):
    game = Game(seed=3)
    win, mouse, show = setup(game)
    win.show()
    wait(app, lambda: False, timeout=0.1)
    ratio = win.devicePixelRatioF()
    g = win.frameGeometry()                            # put the "game" under the app window
    region = Region(int(g.x() * ratio), int(g.y() * ratio), 300, 320)
    mouse.region = region
    win.capture_panel.controller._region = region
    show()
    assert wait(app, win.analysis_current)
    win.autoplay.start()
    assert wait(app, lambda: not win.autoplay.running)
    assert not mouse.clicks and "covering the game" in win.autoplay_label.text()


def test_move_choice():
    board = Board.from_strings(["...", "121"], 2)
    result = solve(board)
    move = choose_move(board, result, recommend(board, result))
    assert isinstance(move, Move) and move.cell == (0, 1) and move.action == REVEAL
    board[(0, 1)] = 2
    result = solve(board)
    rec = recommend(board, result)
    assert isinstance(choose_move(board, result, rec), Stop)                  # only mines left
    flag = choose_move(board, result, rec, flag_mines=True)
    assert isinstance(flag, Move) and flag.action == FLAG_CELL and flag.button == "right"
    guess_board = Board.from_strings(["....", ".1..", "....", "...."], 3)
    result = solve(guess_board)
    rec = recommend(guess_board, result)
    assert isinstance(choose_move(guess_board, result, rec), Stop)            # no guessing by default
    guess = choose_move(guess_board, result, rec, guess=True)
    assert isinstance(guess, Move) and 0 < guess.risk < 1
    lost = Board.from_strings(["*1.", "11.", "..."], 2)
    assert choose_move(lost, solve(lost), recommend(lost, solve(lost))).finished
