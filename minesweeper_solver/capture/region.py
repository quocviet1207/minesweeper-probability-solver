"""A rectangle on the virtual desktop, in physical pixels."""
from __future__ import annotations

from dataclasses import dataclass

MIN_SIZE = 8


@dataclass(frozen=True)
class Region:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def is_valid(self) -> bool:
        return self.width >= MIN_SIZE and self.height >= MIN_SIZE

    def clamped(self, bounds: "Region") -> "Region":
        """Largest part of this region that lies inside `bounds` (at least
        MIN_SIZE in each direction, shifted inwards if necessary)."""
        width = max(MIN_SIZE, min(self.width, bounds.width))
        height = max(MIN_SIZE, min(self.height, bounds.height))
        left = min(max(self.left, bounds.left), bounds.right - width)
        top = min(max(self.top, bounds.top), bounds.bottom - height)
        return Region(left, top, width, height)

    def as_dict(self) -> dict:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}

    @classmethod
    def from_corners(cls, x1: float, y1: float, x2: float, y2: float) -> "Region":
        left, right = sorted((round(x1), round(x2)))
        top, bottom = sorted((round(y1), round(y2)))
        return cls(left, top, right - left, bottom - top)
