"""Acting on the game: mouse control and the auto-play move choice.
Qt-free; depends on core (and capture/vision types for coordinates)."""
from .autoplay import FLAG_CELL, REVEAL, Move, Stop, cell_center, choose_move
from .mouse import Mouse, system_mouse

__all__ = ["Mouse", "system_mouse", "Move", "Stop", "choose_move", "cell_center",
           "REVEAL", "FLAG_CELL"]
