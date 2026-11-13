"""Synthetic Minesweeper screenshots in several visual styles, for vision tests.

These are hand-made approximations of common looks (bevelled classic, flat
checkerboard, gradient tiles, the app's own dark board). They are not copies of
any real game's artwork, so passing here does not prove a real site works.
"""
import random

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter, QPen, QPolygonF

from minesweeper_solver.core import FLAG, MINE, UNREVEALED, Board, solve

CLASSIC_NUMBERS = {1: (0, 0, 255), 2: (0, 128, 0), 3: (255, 0, 0), 4: (0, 0, 128),
                   5: (128, 0, 0), 6: (0, 128, 128), 7: (0, 0, 0), 8: (128, 128, 128)}
FLAT_NUMBERS = {1: (25, 118, 210), 2: (56, 142, 60), 3: (211, 47, 47), 4: (123, 31, 162),
                5: (255, 143, 0), 6: (0, 151, 167), 7: (66, 66, 66), 8: (158, 158, 158)}

THEMES = {
    "classic": dict(font="Trebuchet MS", numbers=CLASSIC_NUMBERS),
    "flat": dict(font="Comic Sans MS", numbers=FLAT_NUMBERS),
    "blue": dict(font="Bahnschrift", numbers=CLASSIC_NUMBERS),
    "mono": dict(font="Verdana", numbers={n: (235, 235, 235) for n in range(1, 9)}),
}


def make_game(rows, cols, mines, seed, rounds=3, flag_rate=0.6, lose=False):
    """A mid-game position reached by playing `rounds` rounds of safe moves."""
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
    for _ in range(rounds):
        result = solve(board)
        for cell in result.mine_cells:
            if rng.random() < flag_rate:
                board[cell] = FLAG
        for cell in result.safe_cells:
            reveal(cell)
    if lose:
        hidden = [c for c in mine_set if board[c] == UNREVEALED]
        if hidden:
            board[rng.choice(hidden)] = MINE
    return board


def _to_array(image: QImage) -> np.ndarray:
    image = image.convertToFormat(QImage.Format_RGB888)
    buf = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.bytesPerLine())
    return buf[:, : image.width() * 3].reshape(image.height(), image.width(), 3).copy()


def _flag(p, rect, pole=(0, 0, 0), cloth=(230, 30, 30)):
    x, y, s = rect.x(), rect.y(), rect.width()
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(*cloth))
    p.drawPolygon(QPolygonF([QPointF(x + .55 * s, y + .2 * s), QPointF(x + .55 * s, y + .55 * s),
                             QPointF(x + .22 * s, y + .375 * s)]))
    p.setBrush(QColor(*pole))
    p.drawRect(QRectF(x + .52 * s, y + .2 * s, max(1.0, .07 * s), .5 * s))
    p.drawRect(QRectF(x + .3 * s, y + .7 * s, .42 * s, max(1.0, .09 * s)))


def _mine(p, rect):
    c, s = rect.center(), rect.width()
    p.setPen(QPen(QColor(0, 0, 0), max(1.0, s * .08)))
    for dx, dy in ((1, 0), (0, 1), (.7, .7), (.7, -.7)):
        p.drawLine(QPointF(c.x() - dx * s * .34, c.y() - dy * s * .34),
                   QPointF(c.x() + dx * s * .34, c.y() + dy * s * .34))
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(0, 0, 0))
    p.drawEllipse(c, s * .24, s * .24)
    p.setBrush(QColor(255, 255, 255))
    p.drawEllipse(QPointF(c.x() - s * .08, c.y() - s * .08), s * .05, s * .05)


def render(board: Board, theme: str, cell: int = 24, scale: float = 1.0) -> np.ndarray:
    """RGB screenshot of `board` in the given theme, with surrounding clutter."""
    spec = THEMES[theme]
    margin, header = 14, 46
    ox, oy = margin, margin + header
    width = board.cols * cell + 2 * margin
    height = board.rows * cell + 2 * margin + header
    image = QImage(width, height, QImage.Format_RGB888)
    page = {"classic": (192, 192, 192), "flat": (255, 255, 255), "blue": (236, 240, 248),
            "mono": (18, 18, 20)}[theme]
    image.fill(QColor(*page))
    p = QPainter(image)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setRenderHint(QPainter.TextAntialiasing, True)

    # header clutter: counters and a title, like a real game window
    head_font = QFont("Consolas")
    head_font.setPixelSize(26)
    head_font.setBold(True)
    p.setFont(head_font)
    if theme == "flat":
        p.fillRect(0, 0, width, header + margin - 6, QColor(74, 117, 44))
        p.setPen(QColor(255, 255, 255))
        p.drawText(QRectF(margin, 6, 200, 40), Qt.AlignLeft | Qt.AlignVCenter, "⚑ 10   ⏱ 042")
    else:
        for x in (margin, width - margin - 62):
            p.fillRect(QRectF(x, 10, 62, 34), QColor(0, 0, 0))
            p.setPen(QColor(255, 0, 0))
            p.drawText(QRectF(x, 10, 62, 34), Qt.AlignCenter, "042")
    if theme == "classic":   # sunken frame around the grid
        p.fillRect(QRectF(ox - 3, oy - 3, board.cols * cell + 6, board.rows * cell + 6), QColor(128, 128, 128))
        p.fillRect(QRectF(ox, oy, board.cols * cell + 3, board.rows * cell + 3), QColor(255, 255, 255))

    font = QFont(spec["font"])
    font.setPixelSize(int(cell * 0.72))
    font.setBold(True)
    bevel = max(2, cell // 8)

    for (r, c) in board.all_cells():
        state = board[(r, c)]
        x, y = ox + c * cell, oy + r * cell
        rect = QRectF(x, y, cell, cell)
        covered = state in (UNREVEALED, FLAG)
        p.setPen(Qt.NoPen)
        p.setRenderHint(QPainter.Antialiasing, False)
        if theme == "classic":
            p.fillRect(rect, QColor(192, 192, 192))
            if covered:
                p.fillRect(QRectF(x, y, cell, bevel), QColor(255, 255, 255))
                p.fillRect(QRectF(x, y, bevel, cell), QColor(255, 255, 255))
                p.fillRect(QRectF(x, y + cell - bevel, cell, bevel), QColor(128, 128, 128))
                p.fillRect(QRectF(x + cell - bevel, y, bevel, cell), QColor(128, 128, 128))
            else:
                if state == MINE:
                    p.fillRect(rect, QColor(255, 0, 0))
                p.fillRect(QRectF(x, y, cell, 1), QColor(128, 128, 128))
                p.fillRect(QRectF(x, y, 1, cell), QColor(128, 128, 128))
        elif theme == "flat":
            odd = (r + c) % 2
            if covered:
                p.fillRect(rect, QColor(*((162, 209, 73) if odd else (170, 215, 81))))
            elif state == MINE:
                p.fillRect(rect, QColor(219, 50, 54))
            else:
                p.fillRect(rect, QColor(*((215, 184, 153) if odd else (229, 194, 159))))
        elif theme == "blue":
            if covered:
                grad = QLinearGradient(x, y, x, y + cell)
                grad.setColorAt(0, QColor(110, 160, 235))
                grad.setColorAt(1, QColor(60, 100, 200))
                p.fillRect(rect, grad)
                p.setPen(QColor(40, 70, 150))
            else:
                p.fillRect(rect, QColor(240, 90, 90) if state == MINE else QColor(215, 225, 240))
                p.setPen(QColor(170, 185, 210))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(x, y, cell - 1, cell - 1))
        elif theme == "mono":
            p.fillRect(rect, QColor(18, 18, 20))
            inner = QRectF(x + 1, y + 1, cell - 2, cell - 2)
            p.fillRect(inner, QColor(70, 74, 84) if covered else
                       (QColor(150, 40, 40) if state == MINE else QColor(34, 36, 42)))
        p.setRenderHint(QPainter.Antialiasing, True)
        if state == FLAG:
            _flag(p, rect, pole=(230, 30, 30) if theme == "flat" else
                  ((240, 240, 240) if theme == "mono" else (0, 0, 0)))
        elif state == MINE:
            _mine(p, rect)
        elif state > 0:
            p.setFont(font)
            p.setPen(QColor(*spec["numbers"][state]))
            p.drawText(rect, Qt.AlignCenter, str(state))
    p.end()
    frame = _to_array(image)
    if scale != 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    return frame


def render_app_board(board: Board, size=(900, 640)) -> np.ndarray:
    """The app's own board widget (normal view), including its row/column labels."""
    from minesweeper_solver.ui.board_widget import VIEW_NORMAL, BoardWidget
    widget = BoardWidget(board.copy())
    widget.set_view(VIEW_NORMAL)
    widget.resize(*size)
    return _to_array(widget.grab().toImage())
