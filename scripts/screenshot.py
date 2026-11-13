"""Render the main window to PNG files.

    python scripts/screenshot.py <output-dir>     # UI smoke test shots
    python scripts/screenshot.py --readme         # the images used in README.md
"""
import os
import sys

if "--readme" not in sys.argv:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PySide6.QtWidgets import QApplication  # noqa: E402

from minesweeper_solver.core.board import random_position  # noqa: E402
from minesweeper_solver.ui import theme  # noqa: E402
from minesweeper_solver.ui.main_window import MainWindow  # noqa: E402


def settle(app) -> None:
    for _ in range(8):
        app.processEvents()


def interesting_cell(win):
    """An undecided cell next to some clues, so the explanation panel has content."""
    result = win.result
    candidates = [c for c, p in result.probs.items() if 0 < p < 1 and c in result.frontier]
    return max(candidates, key=lambda c: (sum(win.board[n] >= 0 for n in win.board.neighbors(c)), c),
               default=None)


def readme_images() -> None:
    out_dir = os.path.join(ROOT, "docs", "images")
    os.makedirs(out_dir, exist_ok=True)
    app = QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet(theme.STYLESHEET)
    win = MainWindow()
    win.resize(1400, 800)
    win.show()
    shots = [("main", random_position(16, 16, 40, seed=12), 1),
             ("recommendation", random_position(16, 30, 99, seed=3), 3)]
    for name, board, view in shots:
        win._replace_board(board)
        win.view_combo.setCurrentIndex(view)
        win.analyse_now()
        win.board_widget._select(interesting_cell(win))
        win.analyse_now()
        settle(app)
        win.grab().save(os.path.join(out_dir, name + ".png"))
        print("wrote", name, "-", win.recommendation.headline)
    win.shutdown()


def main(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    app = QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet(theme.STYLESHEET)
    win = MainWindow()
    win.show()
    shots = [
        ("beginner_mine", random_position(9, 9, 10, seed=4), 1, (2, 2)),
        ("expert_safe", random_position(16, 30, 99, seed=3), 2, (5, 9)),
        ("expert_recommend", random_position(16, 30, 99, seed=3), 3, None),
    ]
    for name, board, view, selected in shots:
        win._replace_board(board)
        win.view_combo.setCurrentIndex(view)
        win.board_widget._select(selected)
        win.analyse_now()
        settle(app)
        win.grab().save(os.path.join(out_dir, name + ".png"))
        print(name, win.recommendation.headline)
    win.shutdown()


if __name__ == "__main__":
    if "--readme" in sys.argv:
        readme_images()
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "screenshots")
