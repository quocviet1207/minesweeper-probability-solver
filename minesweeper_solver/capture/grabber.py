"""Thin wrapper around mss returning RGB numpy arrays.

An instance must be used from the thread that created it (mss keeps
thread-local OS handles).
"""
from __future__ import annotations

import mss
import numpy as np

from .region import Region


class ScreenGrabber:
    def __init__(self):
        self._sct = mss.MSS()

    def close(self) -> None:
        self._sct.close()

    def __enter__(self) -> "ScreenGrabber":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def virtual_bounds(self) -> Region:
        """Bounding box of all monitors."""
        m = self._sct.monitors[0]
        return Region(m["left"], m["top"], m["width"], m["height"])

    def monitors(self) -> list[Region]:
        return [Region(m["left"], m["top"], m["width"], m["height"]) for m in self._sct.monitors[1:]]

    def grab(self, region: Region) -> np.ndarray:
        """Capture `region` as a contiguous (height, width, 3) uint8 RGB array."""
        shot = self._sct.grab(region.as_dict())
        bgra = np.asarray(shot)
        return np.ascontiguousarray(bgra[:, :, 2::-1])
