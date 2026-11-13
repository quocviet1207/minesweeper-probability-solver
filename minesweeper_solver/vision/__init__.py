"""Computer vision: screenshot -> grid -> cell states. No GUI or capture imports."""
from .grid import Grid, detect_grid
from .recognizer import BoardReader, Recognition, recognize

__all__ = ["Grid", "detect_grid", "Recognition", "recognize", "BoardReader"]
