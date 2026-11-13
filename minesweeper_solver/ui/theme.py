"""Colours, risk bands and the application stylesheet."""
from __future__ import annotations

from PySide6.QtGui import QColor

BG = "#161a22"
PANEL = "#1e232e"
PANEL_ALT = "#262c39"
BORDER = "#343c4d"
TEXT = "#e6e9ef"
MUTED = "#8d96a8"
ACCENT = "#4da3ff"

CELL_REVEALED = QColor("#232936")
CELL_COVERED = QColor("#566179")
CELL_COVERED_EDGE = QColor("#6f7b96")
GRID_LINE = QColor("#12151c")

NUMBER_COLORS = {
    1: QColor("#5aa9ff"), 2: QColor("#5fd17a"), 3: QColor("#ff6b6b"), 4: QColor("#b08cff"),
    5: QColor("#ff9f5a"), 6: QColor("#4fd6d6"), 7: QColor("#e6e9ef"), 8: QColor("#a0a8b8"),
}

# (upper bound inclusive, colour, label) for mine probability
_LABELS = ["0% · safe", "1–20%", "21–40%", "41–60%", "61–80%", "81–99%", "100% · mine"]
_UPPER = [0.0, 0.20, 0.40, 0.60, 0.80, 0.999999, 1.0]
PALETTES = {
    "standard": ["#1fa55b", "#8fd694", "#f2d94e", "#f5a142", "#f0703a", "#e03535", "#7d1414"],
    # blue -> yellow -> red: distinguishable with red-green colour blindness
    "colourblind": ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#f46d43", "#d73027", "#67001f"],
}
RISK_BANDS: list = []


def set_palette(name: str) -> None:
    """Switch the risk colours in place (modules holding RISK_BANDS see it)."""
    colors = PALETTES.get(name, PALETTES["standard"])
    RISK_BANDS[:] = [(u, QColor(c), label) for u, c, label in zip(_UPPER, colors, _LABELS)]


set_palette("standard")


def risk_color(mine_prob: float) -> QColor:
    if mine_prob <= 0.0:
        return RISK_BANDS[0][1]
    if mine_prob >= 1.0:
        return RISK_BANDS[-1][1]
    for upper, color, _ in RISK_BANDS[1:-1]:
        if mine_prob <= upper + 1e-9:
            return color
    return RISK_BANDS[-2][1]


def text_on(color: QColor) -> QColor:
    """Readable text colour for a given fill."""
    luminance = 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
    return QColor("#10131a") if luminance > 140 else QColor("#ffffff")


def format_percent(p: float, decimals: int = 0) -> str:
    """Never rounds an uncertain value to 0% or 100%."""
    pct = p * 100
    if p <= 0.0:
        return "0%"
    if p >= 1.0:
        return "100%"
    smallest = 10 ** -decimals
    if pct < smallest:
        return f"<{smallest:g}%"
    if pct > 100 - smallest:
        return f">{100 - smallest:g}%"
    return f"{pct:.{decimals}f}%"


STYLESHEET = f"""
QWidget {{ background: {BG}; color: {TEXT}; font-family: 'Segoe UI', sans-serif; font-size: 13px; }}
QGroupBox {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px;
             margin-top: 14px; padding: 12px 10px 10px 10px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px; color: {MUTED};
                    background: {BG}; }}
QGroupBox QWidget {{ background: transparent; }}
QLabel {{ background: transparent; }}
QLabel[role="muted"] {{ color: {MUTED}; }}
QLabel[role="value"] {{ font-size: 20px; font-weight: 700; }}
QLabel[role="headline"] {{ font-size: 15px; font-weight: 700; }}
QPushButton, QToolButton {{ background: {PANEL_ALT}; border: 1px solid {BORDER}; border-radius: 6px;
               padding: 6px 10px; }}
QPushButton:hover, QToolButton:hover {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: {MUTED}; border-color: {PANEL_ALT}; }}
QToolButton:checked {{ background: {ACCENT}; color: #0b1220; border-color: {ACCENT}; font-weight: 700; }}
QComboBox, QSpinBox {{ background: {PANEL_ALT}; border: 1px solid {BORDER}; border-radius: 6px;
                       padding: 4px 8px; min-height: 22px; }}
QComboBox QAbstractItemView {{ background: {PANEL_ALT}; selection-background-color: {ACCENT};
                               selection-color: #0b1220; }}
QCheckBox {{ background: transparent; spacing: 6px; }}
QScrollArea {{ border: none; }}
QSplitter::handle {{ background: transparent; }}
QSplitter::handle:hover {{ background: {BORDER}; }}
QStatusBar {{ color: {MUTED}; }}
QToolButton {{ padding: 6px 2px; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
"""
