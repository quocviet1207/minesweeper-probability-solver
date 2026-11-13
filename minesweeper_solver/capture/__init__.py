"""Screen capture: region model, frame grabbing, continuous monitoring.

`region`, `grabber` and `frames` are Qt-free; `controller` adapts them to Qt
signals for the UI. Nothing here knows about boards or the solver.
"""
from .frames import FrameMonitor
from .grabber import ScreenGrabber
from .region import Region

__all__ = ["Region", "ScreenGrabber", "FrameMonitor"]
