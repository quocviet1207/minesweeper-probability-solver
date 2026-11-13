"""Left-panel group: region selection, live preview, start / pause / stop."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QCheckBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSpinBox,
                               QVBoxLayout)

from ..capture import Region, ScreenGrabber
from ..capture.controller import PAUSED, RUNNING, STOPPED, CaptureController
from ..capture.region import MIN_SIZE
from . import theme
from .region_selector import RegionSelector, frame_to_image

PREVIEW_HEIGHT = 150


class CapturePanel(QGroupBox):
    frameChanged = Signal(object)   # RGB ndarray whenever the captured picture changes
    frameSettled = Signal(object)   # RGB ndarray once the picture has stopped changing
    message = Signal(str)           # short status-bar notices

    def __init__(self, parent=None, settings: Optional[QSettings] = None):
        super().__init__("Game capture", parent)
        self.settings = settings
        self.controller = CaptureController(self)
        self.controller.frameChanged.connect(self._on_frame)
        self.controller.frameSettled.connect(self.frameSettled)
        self.controller.tick.connect(self._on_tick)
        self.controller.stateChanged.connect(lambda _s: self._refresh_controls())
        self.controller.failed.connect(self._on_failed)
        self._selector: Optional[RegionSelector] = None
        self._stats = (0, 0, 0.0)
        self._last_frame = None
        self._grid = None
        self._error = ""
        try:
            self._grabber: Optional[ScreenGrabber] = ScreenGrabber()
            self._bounds = self._grabber.virtual_bounds()
        except Exception as exc:
            self._grabber, self._bounds = None, Region(0, 0, 1, 1)
            self._error = f"Screen capture unavailable: {exc}"

        v = QVBoxLayout(self)
        self.preview = QLabel("No capture region selected")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setFixedHeight(PREVIEW_HEIGHT)
        self.preview.setStyleSheet(
            f"border: 1px dashed {theme.BORDER}; border-radius: 6px; color: {theme.MUTED};")
        v.addWidget(self.preview)

        self.select_button = QPushButton("Select capture region…")
        self.select_button.clicked.connect(self.select_region)
        v.addWidget(self.select_button)

        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        b = self._bounds
        self.spins = {}
        for i, (key, label, lo, hi) in enumerate([
            ("left", "X", b.left, b.right - MIN_SIZE), ("top", "Y", b.top, b.bottom - MIN_SIZE),
            ("width", "W", MIN_SIZE, b.width), ("height", "H", MIN_SIZE, b.height),
        ]):
            spin = QSpinBox(minimum=lo, maximum=max(lo, hi))
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(self._spins_changed)
            self.spins[key] = spin
            caption = QLabel(label)
            caption.setProperty("role", "muted")
            grid.addWidget(caption, i // 2, (i % 2) * 2)
            grid.addWidget(spin, i // 2, (i % 2) * 2 + 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        v.addLayout(grid)

        row = QHBoxLayout()
        caption = QLabel("Capture every")
        caption.setProperty("role", "muted")
        row.addWidget(caption)
        self.interval = QSpinBox(minimum=20, maximum=5000, singleStep=50, suffix=" ms")
        self.interval.setValue(200)
        self.interval.valueChanged.connect(self._interval_changed)
        row.addWidget(self.interval, 1)
        v.addLayout(row)

        row = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        self.start_button.clicked.connect(self.controller.start)
        self.pause_button.clicked.connect(self.controller.pause)
        self.stop_button.clicked.connect(self.controller.stop)
        for button in (self.start_button, self.pause_button, self.stop_button):
            row.addWidget(button)
        v.addLayout(row)

        self.status = QLabel()
        self.status.setProperty("role", "muted")
        self.status.setWordWrap(True)
        v.addWidget(self.status)

        self.read_board = QCheckBox("Read board from capture")
        self.read_board.setChecked(True)
        self.read_board.setToolTip("Recognise the grid and cells in each captured frame and "
                                   "replace the board with them. Untick to edit by hand.")
        self.read_board.toggled.connect(self._read_toggled)
        v.addWidget(self.read_board)
        self.recognition = QLabel()
        self.recognition.setProperty("role", "muted")
        self.recognition.setWordWrap(True)
        self.recognition.hide()
        v.addWidget(self.recognition)

        self._load_settings()
        self._refresh_controls()

    # -- settings -----------------------------------------------------------
    def _load_settings(self) -> None:
        if self.settings is None:
            return
        self.interval.setValue(int(self.settings.value("capture/interval_ms", 200)))
        saved = self.settings.value("capture/region")
        if saved:
            try:
                self.set_region(Region(*[int(x) for x in str(saved).split(",")]))
            except (TypeError, ValueError):
                pass

    def _save_settings(self) -> None:
        if self.settings is None:
            return
        self.settings.setValue("capture/interval_ms", self.interval.value())
        r = self.controller.region
        if r is not None:
            self.settings.setValue("capture/region", f"{r.left},{r.top},{r.width},{r.height}")

    # -- region ---------------------------------------------------------------
    def set_region(self, region: Region) -> None:
        region = region.clamped(self._bounds)
        for key, spin in self.spins.items():
            spin.blockSignals(True)
            spin.setValue(getattr(region, key))
            spin.blockSignals(False)
        self._error = ""
        self.controller.set_region(region)
        self._save_settings()
        if self.controller.state != RUNNING:
            self._snapshot()
        self._refresh_controls()

    def _spins_changed(self) -> None:
        self.set_region(Region(*(self.spins[k].value() for k in ("left", "top", "width", "height"))))

    def _interval_changed(self, value: int) -> None:
        self.controller.set_interval(value)
        self._save_settings()

    def select_region(self) -> None:
        if self._grabber is None or self._selector is not None:
            return
        self.window().hide()          # so the app does not cover the game
        QTimer.singleShot(300, self._open_selector)

    def _open_selector(self) -> None:
        self._selector = RegionSelector(self)
        self._selector.selected.connect(self._selector_done)
        self._selector.cancelled.connect(lambda: self._selector_done(None))
        try:
            self._selector.start()
        except Exception as exc:
            self._error = f"Could not capture the screen: {exc}"
            self._selector_done(None)

    def _selector_done(self, region: Optional[Region]) -> None:
        if self._selector is not None:
            self._selector.deleteLater()
            self._selector = None
        window = self.window()
        window.show()
        window.raise_()
        window.activateWindow()
        if region is not None:
            self.set_region(region)
            self.message.emit(f"Capture region set to {region.width} × {region.height} px.")
        self._refresh_controls()

    # -- frames ---------------------------------------------------------------
    def _snapshot(self) -> None:
        """One-off grab so the preview follows region edits while not monitoring."""
        region = self.controller.region
        if self._grabber is None or region is None:
            return
        try:
            frame = self._grabber.grab(region)
        except Exception as exc:
            self._error = f"Screen capture failed: {exc}"
            return
        self._on_frame(frame)
        self.frameSettled.emit(frame)        # a still picture: read it straight away

    def _on_frame(self, frame) -> None:
        self._last_frame = frame
        self._show_frame(frame)
        self.frameChanged.emit(frame)

    def _read_toggled(self, on: bool) -> None:
        self._grid = None
        self.recognition.setVisible(False)
        if self._last_frame is not None:
            self._show_frame(self._last_frame)
            if on:
                self.frameSettled.emit(self._last_frame)   # read the current picture now

    def show_recognition(self, result) -> None:
        """Display the outcome of reading the last frame (a vision.Recognition)."""
        self._grid = result.grid if result.ok else None
        if self._last_frame is not None:
            self._show_frame(self._last_frame)
        text = f"{'✓' if result.ok else '✕'} {result.message} · {result.elapsed_ms:.0f} ms"
        self.recognition.setText(text)
        color = "#ff6b6b" if not result.ok else ("#f5a142" if result.unknown_count or result.clipped else "#5fd17a")
        self.recognition.setStyleSheet(f"color: {color};")
        self.recognition.show()

    def _show_frame(self, frame) -> None:
        pixmap = QPixmap.fromImage(frame_to_image(frame))
        if self._grid is not None:            # outline the detected cells
            painter = QPainter(pixmap)
            painter.setPen(QPen(QColor(77, 163, 255), max(1.0, pixmap.width() / 400)))
            xs, ys = self._grid.x_lines, self._grid.y_lines
            for x in xs:
                painter.drawLine(int(x), int(ys[0]), int(x), int(ys[-1]))
            for y in ys:
                painter.drawLine(int(xs[0]), int(y), int(xs[-1]), int(y))
            painter.end()
        dpr = self.devicePixelRatioF()
        target = self.preview.contentsRect().size() * dpr
        pixmap = pixmap.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pixmap.setDevicePixelRatio(dpr)
        self.preview.setPixmap(pixmap)

    def _on_tick(self, captured: int, changed: int, grab_ms: float) -> None:
        self._stats = (captured, changed, grab_ms)
        self._refresh_status()

    def _on_failed(self, message: str) -> None:
        self._error = message
        self.message.emit(message)
        self._refresh_controls()

    # -- state ----------------------------------------------------------------
    def _refresh_controls(self) -> None:
        state = self.controller.state
        has_region = self.controller.region is not None and self._grabber is not None
        self.select_button.setEnabled(self._grabber is not None)
        for spin in self.spins.values():
            spin.setEnabled(has_region)
        self.start_button.setText("Resume" if state == PAUSED else "Start")
        self.start_button.setEnabled(has_region and state != RUNNING)
        self.pause_button.setEnabled(state == RUNNING)
        self.stop_button.setEnabled(state != STOPPED)
        self._refresh_status()

    def _refresh_status(self) -> None:
        state = self.controller.state
        region = self.controller.region
        captured, changed, grab_ms = self._stats
        if self._error:
            text, color = self._error, "#ff6b6b"
        elif region is None:
            text, color = "No region selected — manual input only.", theme.MUTED
        elif state == RUNNING:
            text = (f"● Monitoring {region.width} × {region.height} px · {captured} frames, "
                    f"{changed} changed · {grab_ms:.1f} ms per grab")
            color = "#5fd17a"
        elif state == PAUSED:
            text, color = f"❚❚ Paused · {captured} frames, {changed} changed", "#f5a142"
        else:
            text, color = f"Stopped · region {region.width} × {region.height} px ready", theme.MUTED
        self.status.setText(text)
        self.status.setStyleSheet(f"color: {color};")

    def shutdown(self) -> None:
        self.controller.stop()
        if self._grabber is not None:
            self._grabber.close()
            self._grabber = None
