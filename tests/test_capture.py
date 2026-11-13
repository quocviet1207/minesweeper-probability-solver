"""Capture module: region maths, change detection, controller life cycle."""
import time

import numpy as np
import pytest

from minesweeper_solver.capture import FrameMonitor, Region, ScreenGrabber
from minesweeper_solver.capture.controller import PAUSED, RUNNING, STOPPED, CaptureController


def pump(app, condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


def test_region_from_corners_and_clamp():
    assert Region.from_corners(50, 60, 10, 20) == Region(10, 20, 40, 40)
    bounds = Region(0, 0, 1920, 1080)
    assert Region(1900, 1070, 100, 100).clamped(bounds) == Region(1820, 980, 100, 100)
    assert Region(-50, -50, 5000, 5000).clamped(bounds) == bounds
    assert Region(10, 10, 2, 2).clamped(bounds).is_valid()
    # monitors left of / above the primary have negative coordinates
    assert Region(-3000, 0, 100, 100).clamped(Region(-1920, 0, 3840, 1080)).left == -1920


def test_frame_monitor_only_reports_real_changes():
    monitor = FrameMonitor()
    a = np.zeros((4, 4, 3), np.uint8)
    b = a.copy()
    b[2, 2, 1] = 1
    assert monitor.process(a) is True            # first frame
    assert monitor.process(a.copy()) is False    # identical
    assert monitor.process(b) is True            # one pixel differs
    assert monitor.process(np.zeros((5, 4, 3), np.uint8)) is True   # size changed
    assert (monitor.captured, monitor.changed) == (4, 3)


class FakeGrabber:
    """Returns a new picture every third grab."""
    def __init__(self):
        self.calls = 0
        self.closed = False

    def grab(self, region):
        self.calls += 1
        return np.full((region.height, region.width, 3), (self.calls // 3) % 256, np.uint8)

    def close(self):
        self.closed = True


def test_controller_start_pause_resume_stop(app):
    grabbers = []

    def factory():
        grabbers.append(FakeGrabber())
        return grabbers[-1]

    controller = CaptureController(grabber_factory=factory)
    frames, ticks, states, errors = [], [], [], []
    controller.frameChanged.connect(frames.append)
    controller.tick.connect(lambda *a: ticks.append(a))
    controller.stateChanged.connect(states.append)
    controller.failed.connect(errors.append)

    controller.start()                                   # no region yet
    assert errors and controller.state == STOPPED

    controller.set_interval(1)                           # clamped to the minimum
    assert controller.interval_ms == 20
    controller.set_region(Region(0, 0, 16, 12))
    controller.start()
    assert controller.state == RUNNING
    assert pump(app, lambda: len(ticks) >= 9)
    captured, changed, _ = ticks[-1]
    assert frames[0].shape == (12, 16, 3)
    assert changed < captured                            # unchanged frames were skipped
    assert len(frames) == changed

    controller.pause()
    assert controller.state == PAUSED
    pump(app, lambda: False, timeout=0.1)                # drain in-flight signals
    count = len(ticks)
    pump(app, lambda: False, timeout=0.2)
    assert len(ticks) == count                           # nothing captured while paused

    controller.start()                                   # start while paused = resume
    assert controller.state == RUNNING
    assert pump(app, lambda: len(ticks) > count)

    controller.set_region(Region(0, 0, 20, 10))          # live region change
    assert pump(app, lambda: frames[-1].shape == (10, 20, 3))

    controller.stop()
    assert controller.state == STOPPED and grabbers[0].closed
    assert states == [RUNNING, PAUSED, RUNNING, STOPPED]


def test_frame_monitor_reports_when_the_picture_settles():
    monitor = FrameMonitor()
    a = np.zeros((4, 4, 3), np.uint8)
    b = a + 1
    events = []
    for frame in (a, a, a, b, b, b, b, a, b, b):
        changed = monitor.process(frame)
        events.append("C" if changed else ("S" if monitor.just_settled else "-"))
    # changed, then settled once it repeats, then quiet until the next change
    assert "".join(events) == "CS-CS--CCS"
    assert monitor.settled == 3


def test_settling_ignores_changes_outside_the_board_and_never_waits_forever():
    base = np.zeros((20, 20, 3), np.uint8)

    def frame(board_value, outside_value):
        f = base.copy()
        f[:, :] = outside_value
        f[5:15, 5:15] = board_value
        return f

    # something animates next to the board on every frame
    monitor = FrameMonitor()
    for i in range(10):
        monitor.process(frame(1, i))
        assert not monitor.just_settled               # whole frame never still
    monitor = FrameMonitor(roi=(5, 15, 5, 15))
    events = []
    for i, board in enumerate((1, 1, 1, 2, 2, 2)):
        monitor.process(frame(board, i))
        events.append(monitor.just_settled)
    assert events == [False, True, False, False, True, False]

    # the board itself never holds still: passed on anyway every max_wait frames
    monitor = FrameMonitor(roi=(5, 15, 5, 15), max_wait=3)
    events = []
    for i in range(9):
        monitor.process(frame(i, 0))
        events.append(monitor.just_settled)
    assert events == [False, False, False, True, False, False, False, True, False]
    assert monitor.forced == 2


def test_controller_emits_settled_frames_only_once_still(app):
    class Animated(FakeGrabber):
        """A new picture on each of the first 6 grabs, then still."""
        def grab(self, region):
            self.calls += 1
            return np.full((region.height, region.width, 3), min(self.calls, 6), np.uint8)

    controller = CaptureController(grabber_factory=Animated)
    changed, settled = [], []
    controller.frameChanged.connect(changed.append)
    controller.frameSettled.connect(settled.append)
    controller.set_interval(20)
    controller.set_region(Region(0, 0, 8, 8))
    controller.start()
    assert pump(app, lambda: len(settled) >= 1)
    pump(app, lambda: False, timeout=0.2)
    controller.stop()
    assert len(changed) == 6                                 # every picture previewed
    assert len(settled) == 1 and settled[0][0, 0, 0] == 6    # only the final one read


def test_controller_reports_grab_failure(app):
    class Broken(FakeGrabber):
        def grab(self, region):
            raise OSError("display went away")

    controller = CaptureController(grabber_factory=Broken)
    errors = []
    controller.failed.connect(errors.append)
    controller.set_region(Region(0, 0, 16, 16))
    controller.start()
    assert pump(app, lambda: errors)
    assert "display went away" in errors[0] and controller.state == STOPPED


def test_real_screen_grab():
    try:
        grabber = ScreenGrabber()
    except Exception as exc:                             # headless CI
        pytest.skip(f"no display: {exc}")
    with grabber:
        bounds = grabber.virtual_bounds()
        assert grabber.monitors()
        frame = grabber.grab(Region(bounds.left, bounds.top, 64, 48))
        assert frame.shape == (48, 64, 3) and frame.dtype == np.uint8
        assert frame.flags["C_CONTIGUOUS"]
