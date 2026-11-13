"""Runs auto-play inside the main window: one move per fresh analysis.

Cycle: the latest analysis belongs to the board on screen -> choose a move ->
click it -> wait until the screen shows that cell changed -> next move.

It stops by itself when the game is over, the board reads as inconsistent,
no safe move is left (unless guessing is allowed), a click has no visible
effect, capture or reading is switched off, the app window covers the game,
or the user moves the mouse (the pointer is no longer where auto-play left it).
"""
from __future__ import annotations

import math
import time
from typing import Optional

from PySide6.QtCore import QObject, QPoint, QTimer, Signal

from ..control import Move, Stop, cell_center, choose_move
from ..control.mouse import Mouse, system_mouse
from ..core.board import cell_name

CLICK_TIMEOUT_S = 4.0       # the game must show the click's effect within this time
TAKEOVER_PX = 4             # pointer moved further than this since our click = user took over


class AutoPlayController(QObject):
    runningChanged = Signal(bool)
    status = Signal(str, str)        # text, kind: "running" | "stopped" | "finished" | "error"

    def __init__(self, window, mouse: Optional[Mouse] = None):
        super().__init__(window)
        self.window = window
        self.mouse = mouse if mouse is not None else system_mouse()
        self.running = False
        self.moves = 0
        self._pending = None              # (cell, state before, time of click)
        self._pointer: Optional[tuple[int, int]] = None
        self._scheduled = False
        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self.tick)

    @property
    def available(self) -> bool:
        return self.mouse is not None

    # -- start / stop ------------------------------------------------------------
    def start(self) -> bool:
        w = self.window
        problem = None
        if self.mouse is None:
            problem = "Auto-play needs mouse control, which is not available on this system."
        elif not w._reading():
            problem = "Turn on “Read board from capture” first."
        elif w.capture_panel.controller.state != "running":
            problem = "Select the game's region and press Start capture first."
        elif w._last_recognition is None or w._last_recognition.grid is None:
            problem = "No board has been read from the screen yet."
        if problem:
            self.status.emit(problem, "error")
            return False
        self.running, self.moves = True, 0
        self._pending, self._pointer, self._scheduled = None, None, False
        self._timer.start()
        self.runningChanged.emit(True)
        self.status.emit("Auto-play on. Move the mouse or click Stop to take over.", "running")
        self.tick()
        return True

    def stop(self, reason: str = "Auto-play stopped.", kind: str = "stopped") -> None:
        if not self.running:
            return
        self.running = False
        self._timer.stop()
        self._pending = None
        self.runningChanged.emit(False)
        self.status.emit(f"{reason} ({self.moves} move{'s' if self.moves != 1 else ''} made)", kind)

    # -- cycle ---------------------------------------------------------------------
    def tick(self) -> None:
        """Called on a timer and whenever a new analysis arrives."""
        if not self.running or self._scheduled:
            return
        w = self.window
        if self._pending is not None:
            cell, before, clicked_at = self._pending
            if w.board.in_bounds(cell) and w.board[cell] != before:
                self._pending = None                      # the game reacted
            elif time.monotonic() - clicked_at > CLICK_TIMEOUT_S:
                self.stop(f"The game did not react to the click on {cell_name(cell)}. "
                          "Is the game window visible and not covered?", "error")
                return
            else:
                return
        if not w.analysis_current():
            return                                        # wait for the analysis of this board
        decision = choose_move(w.board, w.result, w.recommendation,
                               guess=w.prefs.autoplay_guess, flag_mines=w.prefs.autoplay_flag)
        if isinstance(decision, Stop):
            self.stop(decision.reason, "finished" if decision.finished else "stopped")
            return
        self._scheduled = True
        QTimer.singleShot(max(0, w.prefs.autoplay_delay_ms), lambda: self._perform(decision))

    def _perform(self, move: Move) -> None:
        self._scheduled = False
        if not self.running:
            return
        w = self.window
        if not w.analysis_current() or w.board[move.cell] != (-1):
            self.tick()                                    # the board moved on meanwhile
            return
        if self._pointer is not None:
            x, y = self.mouse.position()
            if math.hypot(x - self._pointer[0], y - self._pointer[1]) > TAKEOVER_PX:
                self.stop("You moved the mouse, so auto-play handed control back to you.")
                return
        target = self._target(move)
        if target is None:
            return
        self.mouse.click(*target, button=move.button)
        self._pointer = target
        self._pending = (move.cell, w.board[move.cell], time.monotonic())
        self.moves += 1
        verb = "Revealed" if move.action == "reveal" else "Flagged"
        self.status.emit(f"Auto-play: {verb.lower()} {cell_name(move.cell)}, {move.reason.split(' ', 1)[1]} · "
                         f"{self.moves} move{'s' if self.moves != 1 else ''} · move the mouse to take over",
                         "running")

    def _target(self, move: Move) -> Optional[tuple[int, int]]:
        w = self.window
        recognition = w._last_recognition
        region = w.capture_panel.controller.region
        if recognition is None or recognition.grid is None or region is None:
            self.stop("The board is no longer being read.", "error")
            return None
        grid = recognition.grid
        if (grid.rows, grid.cols) != (w.board.rows, w.board.cols):
            self.stop("The board on screen changed size.", "error")
            return None
        x, y = cell_center(region.left, region.top, grid.x_lines, grid.y_lines, move.cell)
        if not (region.left <= x < region.right and region.top <= y < region.bottom):
            self.stop("That cell is outside the capture region.", "error")
            return None
        # never click on this app's own window
        ratio = w.devicePixelRatioF()
        frame = w.frameGeometry()
        if w.isVisible() and frame.contains(QPoint(int(x / ratio), int(y / ratio))):
            self.stop("This app's window is covering the game. Move it aside and start again.", "error")
            return None
        return x, y
