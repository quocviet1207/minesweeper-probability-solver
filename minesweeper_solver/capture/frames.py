"""Frame bookkeeping: decides whether a new frame differs from the last one,
and when the picture has settled, so that downstream work (preview,
recognition, solving) only runs when it has to."""
from __future__ import annotations

from typing import Optional

import numpy as np

Rect = tuple[int, int, int, int]      # top, bottom, left, right in frame pixels


class FrameMonitor:
    """Feed every captured frame to `process`.

    - `process` returns True when the frame differs from the previous one
      (use it for the live preview).
    - `just_settled` becomes True on the first frame whose region of interest
      is identical to the previous frame's after a change, i.e. once the board
      has stopped moving for one capture interval. Recognition runs on settled
      frames only, so it never sees half-finished animations or scrolling.
    - `roi` limits the stillness test to part of the frame (the board, once it
      has been found), so a ticking timer or anything animated around the
      board cannot keep it from settling.
    - `max_wait`: if the region keeps changing for this many frames, the
      current frame is passed on anyway (0 = wait indefinitely).
    """

    def __init__(self, roi: Optional[Rect] = None, max_wait: int = 0):
        self.roi = roi
        self.max_wait = max_wait
        self.reset()

    def reset(self) -> None:
        self.captured = self.changed = self.settled = self.forced = 0
        self.just_settled = False
        self._last: Optional[np.ndarray] = None
        self._previous_view: Optional[np.ndarray] = None
        self._is_settled = False
        self._waiting = 0

    @property
    def last_frame(self) -> Optional[np.ndarray]:
        return self._last

    def _view(self, frame: np.ndarray) -> np.ndarray:
        if self.roi is None:
            return frame
        top, bottom, left, right = self.roi
        return frame[max(0, top): bottom, max(0, left): right]

    def process(self, frame: np.ndarray) -> bool:
        """Record `frame`; True if it is the first frame or differs from the last."""
        self.captured += 1
        last = self._last
        changed = not (last is not None and last.shape == frame.shape and np.array_equal(last, frame))
        if changed:
            self._last = frame
            self.changed += 1

        view, previous = self._view(frame), self._previous_view
        self._previous_view = view
        still = previous is not None and previous.shape == view.shape and np.array_equal(previous, view)
        self.just_settled = False
        if still:
            self._waiting = 0
            if not self._is_settled:
                self._is_settled = self.just_settled = True
                self.settled += 1
        else:
            self._is_settled = False
            self._waiting += 1
            if self.max_wait and self._waiting > self.max_wait:
                self._waiting = 0                     # never still: pass it on anyway
                self.just_settled = True
                self.settled += 1
                self.forced += 1
        return changed
