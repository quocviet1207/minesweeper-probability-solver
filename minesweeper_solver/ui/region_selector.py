"""Full-screen overlay for dragging out a capture region.

Each monitor gets its own overlay showing a frozen screenshot of that monitor,
so mouse positions convert to physical pixels with a simple per-monitor scale
regardless of display scaling.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from ..capture import Region, ScreenGrabber
from ..capture.region import MIN_SIZE
from . import theme


def frame_to_image(frame: np.ndarray) -> QImage:
    height, width, _ = frame.shape
    return QImage(frame.data, width, height, 3 * width, QImage.Format_RGB888).copy()


class _Overlay(QWidget):
    selected = Signal(object)   # Region in physical pixels
    cancelled = Signal()

    def __init__(self, screen, monitor: Region, pixmap: QPixmap):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.monitor = monitor
        self.pixmap = pixmap
        self._start: Optional[QPointF] = None
        self._end: Optional[QPointF] = None
        self.setCursor(Qt.CrossCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setGeometry(screen.geometry())

    # logical widget coordinates -> physical desktop pixels
    def _to_region(self) -> Optional[Region]:
        if self._start is None or self._end is None:
            return None
        sx = self.monitor.width / max(1, self.width())
        sy = self.monitor.height / max(1, self.height())
        region = Region.from_corners(self._start.x() * sx, self._start.y() * sy,
                                     self._end.x() * sx, self._end.y() * sy)
        return Region(self.monitor.left + region.left, self.monitor.top + region.top,
                      region.width, region.height)

    def _clamp(self, pos: QPointF) -> QPointF:
        return QPointF(min(max(pos.x(), 0), self.width()), min(max(pos.y(), 0), self.height()))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._start = self._end = self._clamp(event.position())
            self.update()
        elif event.button() == Qt.RightButton:
            self.cancelled.emit()

    def mouseMoveEvent(self, event):
        if self._start is not None and event.buttons() & Qt.LeftButton:
            self._end = self._clamp(event.position())
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton or self._start is None:
            return
        self._end = self._clamp(event.position())
        region = self._to_region()
        if region is not None and region.width >= MIN_SIZE and region.height >= MIN_SIZE:
            self.selected.emit(region)
        else:                      # a click without a drag: start over
            self._start = self._end = None
            self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.cancelled.emit()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        full = QRectF(self.rect())
        p.drawPixmap(full, self.pixmap, QRectF(self.pixmap.rect()))
        p.fillRect(full, QColor(0, 0, 0, 120))

        region = self._to_region()
        if region is not None and self._start != self._end:
            sel = QRectF(self._start, self._end).normalized()
            sx = self.pixmap.width() / max(1, self.width())
            sy = self.pixmap.height() / max(1, self.height())
            source = QRectF(sel.x() * sx, sel.y() * sy, sel.width() * sx, sel.height() * sy)
            p.drawPixmap(sel, self.pixmap, source)
            p.setPen(QPen(QColor(theme.ACCENT), 2))
            p.setBrush(Qt.NoBrush)
            p.drawRect(sel)
            self._pill(p, f"{region.width} × {region.height} px",
                       QPointF(sel.center().x(), min(sel.bottom() + 22, self.height() - 18)))
        self._pill(p, "Drag a rectangle around the Minesweeper board  ·  Esc to cancel",
                   QPointF(self.width() / 2, 36))
        p.end()

    @staticmethod
    def _pill(p: QPainter, text: str, center: QPointF) -> None:
        font = QFont("Segoe UI")
        font.setPixelSize(14)
        font.setBold(True)
        p.setFont(font)
        width = p.fontMetrics().horizontalAdvance(text) + 28
        rect = QRectF(center.x() - width / 2, center.y() - 15, width, 30)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(22, 26, 34, 235))
        p.drawRoundedRect(rect, 15, 15)
        p.setPen(QColor("#ffffff"))
        p.drawText(rect, Qt.AlignCenter, text)


class RegionSelector(QObject):
    """Shows one overlay per monitor and reports the dragged region."""
    selected = Signal(object)   # Region
    cancelled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.overlays: list[_Overlay] = []

    def start(self) -> None:
        with ScreenGrabber() as grabber:
            monitors = grabber.monitors()
            shots = [grabber.grab(m) for m in monitors]
        screens = QGuiApplication.screens()
        for index, screen in enumerate(screens):
            origin = screen.geometry().topLeft()
            match = next((i for i, m in enumerate(monitors)
                          if (m.left, m.top) == (origin.x(), origin.y())), None)
            if match is None:
                match = index if index < len(monitors) else None
            if match is None:
                continue
            overlay = _Overlay(screen, monitors[match], QPixmap.fromImage(frame_to_image(shots[match])))
            overlay.selected.connect(self._finish)
            overlay.cancelled.connect(lambda: self._finish(None))
            self.overlays.append(overlay)
        if not self.overlays:
            self.cancelled.emit()
            return
        for overlay in self.overlays:
            overlay.show()
        self.overlays[0].activateWindow()
        self.overlays[0].setFocus()

    def _finish(self, region: Optional[Region]) -> None:
        for overlay in self.overlays:
            overlay.close()
            overlay.deleteLater()
        self.overlays.clear()
        if region is None:
            self.cancelled.emit()
        else:
            self.selected.emit(region)
