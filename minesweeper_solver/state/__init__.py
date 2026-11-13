"""Game state: keeps the reconstructed board in step with successive readings.
Qt-free; depends on `core` only."""
from .manager import (CHANGED, FIRST, NEW_GAME, RESIZED, UNCHANGED, GameStateManager,
                      SyncUpdate, standard_mines)

__all__ = ["GameStateManager", "SyncUpdate", "standard_mines",
           "FIRST", "CHANGED", "UNCHANGED", "NEW_GAME", "RESIZED"]
