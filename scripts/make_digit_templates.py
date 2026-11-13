"""Regenerates minesweeper_solver/vision/digit_templates.npz.

Renders the digits 1-8 in a range of installed fonts and stores their
normalised masks. Run on a machine with the listed fonts (Windows):

    python scripts/make_digit_templates.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFont, QFontInfo, QGuiApplication, QImage, QPainter  # noqa: E402

from minesweeper_solver.vision.glyphs import TEMPLATE_FILE, normalize_mask  # noqa: E402

FONTS = ["Arial", "Arial Black", "Verdana", "Tahoma", "Segoe UI", "Courier New", "Consolas",
         "Times New Roman", "Calibri", "Impact", "Lucida Console", "Franklin Gothic Medium"]


PIXEL_FONTS = ["Arial", "Verdana", "Tahoma", "Segoe UI"]
SLAB_ONE = {"Courier New", "Lucida Console", "Consolas"}   # their heavy 1 resembles a 2
PIXEL_SIZES = [9, 10, 11, 12, 13]     # rendered without smoothing: blocky, like sprite digits


def render(digit: int, family: str, bold: bool, size: int = 72, smooth: bool = True) -> np.ndarray:
    image = QImage(size * 2, size * 2, QImage.Format_Grayscale8)
    image.fill(0)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.TextAntialiasing, smooth)
    font = QFont(family)
    font.setPixelSize(size)
    font.setBold(bold)
    if not smooth:
        font.setStyleStrategy(QFont.NoAntialias)
    painter.setFont(font)
    painter.setPen(QColor(255, 255, 255))
    painter.drawText(QRectF(0, 0, size * 2, size * 2), Qt.AlignCenter, str(digit))
    painter.end()
    buf = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.bytesPerLine())
    return buf[:, : image.width()].copy() > 110


def fill_holes(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.uint8), 1)
    _, labels = cv2.connectedComponents(1 - padded, connectivity=4)
    outside = labels == labels[0, 0]
    return ~outside[1:-1, 1:-1]


def main() -> None:
    app = QGuiApplication([])  # noqa: F841
    masks, digits, used = [], [], []
    for family in FONTS:
        if QFontInfo(QFont(family)).family() != family:
            print("skipping missing font:", family)
            continue
        used.append(family)
        for bold in (True, False):
            for digit in range(1, 9):
                glyph = render(digit, family, bold)
                variants = [glyph]
                if bold and not (digit == 1 and family in SLAB_ONE):
                    # small pixel-art digits are heavier than any font and
                    # their counters (the holes in 4, 6, 8) are often filled in
                    heavy = cv2.dilate(glyph.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
                    variants.append(heavy)
                    if digit in (4, 6, 8):
                        variants.append(fill_holes(glyph))
                        variants.append(fill_holes(heavy))
                for variant in variants:
                    masks.append(normalize_mask(variant))
                    digits.append(digit)
    for family in PIXEL_FONTS:
        if QFontInfo(QFont(family)).family() != family:
            continue
        for size in PIXEL_SIZES:
            for digit in range(1, 9):
                glyph = render(digit, family, True, size, smooth=False)
                if glyph.any():
                    masks.append(normalize_mask(glyph))
                    digits.append(digit)
    np.savez_compressed(TEMPLATE_FILE, masks=np.array(masks, np.float32), digits=np.array(digits, np.int8))
    print(f"wrote {len(masks)} templates from {len(used)} fonts to {TEMPLATE_FILE}")


if __name__ == "__main__":
    main()
