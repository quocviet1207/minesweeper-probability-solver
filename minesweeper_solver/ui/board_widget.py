"""Custom-painted, editable board with probability overlays."""
from __future__ import annotations

import time
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..core import EXACT, FLAG, MINE, UNKNOWN, UNREVEALED, Board, Recommendation, SolveResult
from ..core.board import Cell, cell_name
from . import theme

VIEW_NORMAL, VIEW_MINE, VIEW_SAFE, VIEW_RECOMMEND = range(4)
TOOL_SELECT = "select"   # other tools are cell states (ints)
FLASH_SECONDS = 1.6       # how long a changed cell stays highlighted


class BoardWidget(QWidget):
    boardEdited = Signal()
    digitTyped = Signal(object, int)    # cell, digit 1-8 typed on it (before the edit)
    selectionChanged = Signal(object)   # Cell or None

    def __init__(self, board: Board, parent=None):
        super().__init__(parent)
        self.board = board
        self.result: Optional[SolveResult] = None
        self.recommendation: Optional[Recommendation] = None
        self.view = VIEW_MINE
        self.show_percent = True
        self.decimals = 0
        self.flash_seconds = FLASH_SECONDS     # 0 disables the changed-cell highlight
        self.tool = TOOL_SELECT
        self.selected: Optional[Cell] = None
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(200, 200)
        self._flashes: dict[Cell, float] = {}     # cell -> time it changed
        self._flash_timer = QTimer(self)
        self._flash_timer.setInterval(40)
        self._flash_timer.timeout.connect(self._fade)

    def flash(self, cells) -> None:
        """Briefly highlight cells that just changed in the live game."""
        if self.flash_seconds <= 0:
            return
        now = time.monotonic()
        for cell in cells:
            self._flashes[cell] = now
        if self._flashes and not self._flash_timer.isActive():
            self._flash_timer.start()
        self.update()

    def _fade(self) -> None:
        now = time.monotonic()
        self._flashes = {c: t for c, t in self._flashes.items() if now - t < self.flash_seconds}
        if not self._flashes:
            self._flash_timer.stop()
        self.update()

    # -- public API -------------------------------------------------------
    def set_board(self, board: Board) -> None:
        self.board = board
        if self.selected and not board.in_bounds(self.selected):
            self.selected = None
            self.selectionChanged.emit(None)
        self.update()

    def set_analysis(self, result: SolveResult, recommendation: Recommendation) -> None:
        self.result = result
        self.recommendation = recommendation
        self.update()

    def set_view(self, view: int) -> None:
        self.view = view
        self.update()

    def set_show_percent(self, on: bool) -> None:
        self.show_percent = on
        self.update()

    # -- geometry -----------------------------------------------------------
    def _layout(self) -> tuple[float, float, float]:
        """(cell size, x origin, y origin) of the grid, leaving room for headers."""
        header = 22
        size = min((self.width() - header - 8) / self.board.cols,
                   (self.height() - header - 8) / self.board.rows, 64)
        size = max(size, 4)
        x0 = header + (self.width() - header - size * self.board.cols) / 2
        y0 = header + (self.height() - header - size * self.board.rows) / 2
        return size, x0, y0

    def _cell_at(self, pos: QPointF) -> Optional[Cell]:
        size, x0, y0 = self._layout()
        c = int((pos.x() - x0) // size)
        r = int((pos.y() - y0) // size)
        return (r, c) if self.board.in_bounds((r, c)) else None

    # -- editing ------------------------------------------------------------
    def _select(self, cell: Optional[Cell]) -> None:
        if cell != self.selected:
            self.selected = cell
            self.selectionChanged.emit(cell)
            self.update()

    def _set_cell(self, cell: Cell, state: int) -> None:
        if self.board[cell] != state:
            self.board[cell] = state
            self.boardEdited.emit()
            self.update()

    def mousePressEvent(self, event):
        self.setFocus()
        cell = self._cell_at(event.position())
        if cell is None:
            return
        self._select(cell)
        if event.button() == Qt.RightButton:
            self._set_cell(cell, UNREVEALED if self.board[cell] == FLAG else FLAG)
        elif event.button() == Qt.LeftButton and self.tool != TOOL_SELECT:
            self._set_cell(cell, self.tool)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.tool != TOOL_SELECT:
            cell = self._cell_at(event.position())
            if cell is not None:
                self._select(cell)
                self._set_cell(cell, self.tool)

    def keyPressEvent(self, event):
        key = event.key()
        moves = {Qt.Key_Left: (0, -1), Qt.Key_Right: (0, 1), Qt.Key_Up: (-1, 0), Qt.Key_Down: (1, 0)}
        if key in moves:
            r, c = self.selected or (0, 0)
            dr, dc = moves[key] if self.selected else (0, 0)
            target = (min(max(r + dr, 0), self.board.rows - 1), min(max(c + dc, 0), self.board.cols - 1))
            self._select(target)
            return
        if self.selected is None:
            return super().keyPressEvent(event)
        if Qt.Key_0 <= key <= Qt.Key_8:
            if key != Qt.Key_0:
                self.digitTyped.emit(self.selected, key - Qt.Key_0)
            self._set_cell(self.selected, key - Qt.Key_0)
        elif key == Qt.Key_F:
            self._set_cell(self.selected, FLAG)
        elif key == Qt.Key_M:
            self._set_cell(self.selected, MINE)
        elif key in (Qt.Key_U, Qt.Key_Space, Qt.Key_Delete, Qt.Key_Backspace, Qt.Key_Period):
            self._set_cell(self.selected, UNREVEALED)
        else:
            super().keyPressEvent(event)

    # -- painting -----------------------------------------------------------
    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(theme.BG))
        size, x0, y0 = self._layout()
        board, result, rec = self.board, self.result, self.recommendation
        probs = result.probs if result and result.valid else {}

        header_font = QFont("Segoe UI")
        header_font.setPixelSize(int(min(12, max(7, size * 0.38))))
        p.setFont(header_font)
        p.setPen(QColor(theme.MUTED))
        for c in range(board.cols):
            p.drawText(QRectF(x0 + c * size, y0 - 20, size, 18), Qt.AlignCenter,
                       cell_name((0, c))[:-1])
        for r in range(board.rows):
            p.drawText(QRectF(x0 - 26, y0 + r * size, 22, size), Qt.AlignVCenter | Qt.AlignRight, str(r + 1))

        big = QFont("Segoe UI")
        big.setPixelSize(max(6, int(size * 0.56)))
        big.setBold(True)
        small = QFont("Segoe UI")
        small.setPixelSize(max(6, int(size * 0.33)))
        small.setBold(True)
        gap = 1.0 if size >= 10 else 0.5
        radius = min(4.0, size * 0.12)

        for cell in board.all_cells():
            r, c = cell
            rect = QRectF(x0 + c * size + gap, y0 + r * size + gap, size - 2 * gap, size - 2 * gap)
            state = board[cell]
            p.setPen(Qt.NoPen)

            if state >= 0:
                p.setBrush(theme.CELL_REVEALED)
                p.drawRoundedRect(rect, radius, radius)
                if state:
                    p.setFont(big)
                    p.setPen(theme.NUMBER_COLORS[state])
                    p.drawText(rect, Qt.AlignCenter, str(state))
                continue
            if state == MINE:
                p.setBrush(QColor("#b3261e"))
                p.drawRoundedRect(rect, radius, radius)
                p.setFont(big)
                p.setPen(QColor("#ffffff"))
                p.drawText(rect, Qt.AlignCenter, "✹")
                continue
            if state == FLAG:
                p.setBrush(theme.CELL_COVERED)
                p.drawRoundedRect(rect, radius, radius)
                p.setFont(big)
                p.setPen(QColor("#ff5d5d"))
                p.drawText(rect, Qt.AlignCenter, "⚑")
                continue

            if state == UNKNOWN:
                p.setBrush(QColor("#6b5a2a"))
                p.drawRoundedRect(rect, radius, radius)
                p.setPen(QPen(QColor("#f5c542"), 1, Qt.DashLine))
                p.setBrush(Qt.NoBrush)
                p.drawRoundedRect(rect.adjusted(1, 1, -1, -1), radius, radius)
                p.setFont(big)
                p.setPen(QColor("#ffe08a"))
                p.drawText(rect, Qt.AlignCenter, "?")
                continue

            # covered, unflagged
            prob = probs.get(cell)
            fill, label, label_color = theme.CELL_COVERED, "", None
            if prob is not None and self.view in (VIEW_MINE, VIEW_SAFE):
                fill = theme.risk_color(prob)
                shown = prob if self.view == VIEW_MINE else 1.0 - prob
                label = theme.format_percent(shown, self.decimals)
                if result.quality.get(cell) != EXACT:
                    label = "~" + label
                label_color = theme.text_on(fill)
            elif prob is not None and self.view == VIEW_RECOMMEND:
                if prob == 0.0:
                    fill, label = theme.RISK_BANDS[0][1], "✓"
                elif prob == 1.0:
                    fill, label = theme.RISK_BANDS[-1][1], "✕"
                elif rec and rec.cell == cell:
                    fill, label = theme.risk_color(prob), theme.format_percent(prob, self.decimals)
                else:
                    fill = QColor("#3a4254")
                label_color = theme.text_on(fill)
            p.setBrush(fill)
            p.drawRoundedRect(rect, radius, radius)
            if fill == theme.CELL_COVERED and size >= 10:
                p.setPen(QPen(theme.CELL_COVERED_EDGE, 1))
                p.drawLine(rect.topLeft() + QPointF(radius, 0.5), rect.topRight() + QPointF(-radius, 0.5))
            if label and (self.show_percent or self.view == VIEW_RECOMMEND) and size >= 14:
                p.setFont(big if label in ("✓", "✕") else small)
                p.setPen(label_color)
                p.drawText(rect, Qt.AlignCenter, label)

        # cells that just changed in the live game
        now = time.monotonic()
        for (r, c), started in self._flashes.items():
            if not board.in_bounds((r, c)):
                continue
            alpha = max(0.0, 1.0 - (now - started) / max(self.flash_seconds, 1e-6))
            color = QColor(theme.ACCENT)
            color.setAlphaF(alpha)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(color, max(2.0, size * 0.1)))
            p.drawRoundedRect(QRectF(x0 + c * size + 1.5, y0 + r * size + 1.5, size - 3, size - 3),
                              radius, radius)

        # recommended move ring
        if rec and rec.cell is not None and self.view != VIEW_NORMAL:
            r, c = rec.cell
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor("#ffffff"), max(2.0, size * 0.09)))
            p.drawRoundedRect(QRectF(x0 + c * size + 2, y0 + r * size + 2, size - 4, size - 4), radius, radius)
        if self.selected is not None:
            r, c = self.selected
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(theme.ACCENT), max(2.0, size * 0.07), Qt.DashLine))
            p.drawRect(QRectF(x0 + c * size, y0 + r * size, size, size))
        p.end()
