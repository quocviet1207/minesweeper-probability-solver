"""Continuous capture on a background thread, exposed through Qt signals."""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from PySide6.QtCore import QObject, QThread, Signal

from .frames import FrameMonitor
from .grabber import ScreenGrabber
from .region import Region

STOPPED, RUNNING, PAUSED = "stopped", "running", "paused"
MIN_INTERVAL_MS = 20
MAX_SETTLE_MS = 1000     # longest wait for a still picture before reading it anyway


class _Worker(QThread):
    frameChanged = Signal(object)        # RGB ndarray, only when the picture changed
    frameSettled = Signal(object)        # RGB ndarray, once the picture has stopped changing
    tick = Signal(int, int, float)       # frames captured, frames changed, grab time (ms)
    failed = Signal(str)

    def __init__(self, region: Region, interval_ms: int, grabber_factory: Callable, parent=None):
        super().__init__(parent)
        self.region = region
        self.interval_ms = interval_ms
        self.paused = False
        self.settle = True
        self.settle_roi = None           # (top, bottom, left, right) judged for stillness
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._grabber_factory = grabber_factory

    def request_stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def wake(self) -> None:
        self._wake.set()

    def run(self) -> None:
        monitor = FrameMonitor()
        try:
            grabber = self._grabber_factory()
        except Exception as exc:  # no display, driver problem, ...
            self.failed.emit(f"Could not start screen capture: {exc}")
            return
        try:
            last_region = None
            while not self._stop.is_set():
                self._wake.clear()
                if self.paused:
                    self._wake.wait(0.2)
                    continue
                started = time.perf_counter()
                region = self.region
                if region != last_region:
                    monitor.reset()
                    last_region = region
                try:
                    frame = grabber.grab(region)
                except Exception as exc:
                    self.failed.emit(f"Screen capture failed: {exc}")
                    return
                grab_ms = (time.perf_counter() - started) * 1000
                monitor.roi = self.settle_roi
                # if the board never holds still, read it anyway after ~1 s
                monitor.max_wait = max(3, round(MAX_SETTLE_MS / max(self.interval_ms, 1)))
                changed = monitor.process(frame)
                if changed:
                    self.frameChanged.emit(frame)
                if not self.settle:
                    if changed:                  # read every new picture straight away
                        self.frameSettled.emit(frame)
                elif monitor.just_settled:
                    self.frameSettled.emit(frame)
                self.tick.emit(monitor.captured, monitor.changed, grab_ms)
                remaining = self.interval_ms / 1000 - (time.perf_counter() - started)
                if remaining > 0:
                    self._wake.wait(remaining)
        finally:
            grabber.close()


class CaptureController(QObject):
    """Start / pause / stop monitoring of one screen region."""
    frameChanged = Signal(object)
    frameSettled = Signal(object)
    tick = Signal(int, int, float)
    stateChanged = Signal(str)
    failed = Signal(str)

    def __init__(self, parent=None, grabber_factory: Callable = ScreenGrabber):
        super().__init__(parent)
        self._grabber_factory = grabber_factory
        self._worker: Optional[_Worker] = None
        self._region: Optional[Region] = None
        self._interval_ms = 200
        self._settle = True
        self._settle_roi = None
        self.state = STOPPED

    # -- configuration (safe to change while running) ----------------------
    @property
    def region(self) -> Optional[Region]:
        return self._region

    def set_region(self, region: Optional[Region]) -> None:
        if region != self._region:
            self.set_settle_region(None)
        self._region = region
        if self._worker and region is not None:
            self._worker.region = region
            self._worker.wake()

    @property
    def interval_ms(self) -> int:
        return self._interval_ms

    def set_interval(self, interval_ms: int) -> None:
        self._interval_ms = max(MIN_INTERVAL_MS, int(interval_ms))
        if self._worker:
            self._worker.interval_ms = self._interval_ms
            self._worker.wake()

    @property
    def settle(self) -> bool:
        return self._settle

    def set_settle_region(self, roi) -> None:
        """Judge stillness only inside `roi` = (top, bottom, left, right) in
        frame pixels, typically the board once it has been found; None = the
        whole frame."""
        self._settle_roi = roi
        if self._worker:
            self._worker.settle_roi = roi

    def set_settle(self, on: bool) -> None:
        """Wait until the picture holds still before passing it on (default),
        or pass on every changed picture immediately."""
        self._settle = bool(on)
        if self._worker:
            self._worker.settle = self._settle

    # -- control -------------------------------------------------------------
    def start(self) -> None:
        if self._region is None or not self._region.is_valid():
            self.failed.emit("Select a capture region first.")
            return
        if self._worker:
            self.resume()
            return
        worker = _Worker(self._region, self._interval_ms, self._grabber_factory, self)
        worker.settle = self._settle
        worker.settle_roi = self._settle_roi
        worker.frameChanged.connect(self.frameChanged)
        worker.frameSettled.connect(self.frameSettled)
        worker.tick.connect(self.tick)
        worker.failed.connect(self._on_failed)
        self._worker = worker
        worker.start()
        self._set_state(RUNNING)

    def pause(self) -> None:
        if self._worker and self.state == RUNNING:
            self._worker.paused = True
            self._set_state(PAUSED)

    def resume(self) -> None:
        if self._worker and self.state == PAUSED:
            self._worker.paused = False
            self._worker.wake()
            self._set_state(RUNNING)

    def stop(self) -> None:
        worker, self._worker = self._worker, None
        if worker:
            worker.request_stop()
            worker.wait(3000)
        self._set_state(STOPPED)

    def _on_failed(self, message: str) -> None:
        self.stop()
        self.failed.emit(message)

    def _set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self.stateChanged.emit(state)
