"""Entry point: python -m minesweeper_solver"""
import sys

from pathlib import Path

from PySide6.QtCore import QSettings, QStandardPaths
from PySide6.QtWidgets import QApplication

from .ui import theme
from .ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(theme.STYLESHEET)
    app.setApplicationName("MinesweeperSolver")
    data = Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation))
    window = MainWindow(QSettings("MinesweeperSolver", "MinesweeperSolver"),
                        taught_digits=str(data / "taught_digits.npz"))
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
