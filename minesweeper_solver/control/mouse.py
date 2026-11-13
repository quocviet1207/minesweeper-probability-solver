"""Moving and clicking the real mouse.

Coordinates are physical desktop pixels, the same space the screen capture
uses. The Qt application is per-monitor DPI aware (Qt 6's default on
Windows), so the Win32 cursor functions use physical pixels too.
"""
from __future__ import annotations

import sys
import time
from typing import Optional


class Mouse:
    """Interface; `system_mouse()` returns the real one where available."""

    def position(self) -> tuple[int, int]:
        raise NotImplementedError

    def click(self, x: int, y: int, button: str = "left") -> None:
        raise NotImplementedError


class _WindowsMouse(Mouse):
    _DOWN_UP = {"left": (0x0002, 0x0004), "right": (0x0008, 0x0010)}

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        self._ctypes = ctypes
        self._user32 = ctypes.windll.user32
        self._point = wintypes.POINT

    def position(self) -> tuple[int, int]:
        point = self._point()
        self._user32.GetCursorPos(self._ctypes.byref(point))
        return point.x, point.y

    def click(self, x: int, y: int, button: str = "left") -> None:
        down, up = self._DOWN_UP[button]
        self._user32.SetCursorPos(int(x), int(y))
        time.sleep(0.015)                     # let the game see the pointer arrive
        self._user32.mouse_event(down, 0, 0, 0, 0)
        time.sleep(0.03)
        self._user32.mouse_event(up, 0, 0, 0, 0)


def system_mouse() -> Optional[Mouse]:
    """The real mouse, or None where this platform is not supported."""
    if sys.platform == "win32":
        try:
            return _WindowsMouse()
        except Exception:
            return None
    return None
