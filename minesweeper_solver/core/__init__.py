"""Pure-Python game logic: board model, probability solver, recommendations.

Nothing in this package may import GUI, capture or vision code.
"""
from .board import FLAG, MINE, UNKNOWN, UNREVEALED, Board, cell_name
from .explain import explain_cell
from .recommend import Recommendation, recommend
from .solver import ESTIMATED, EXACT, PRIOR, UNCERTAIN, SolverCache, SolveResult, solve

__all__ = [
    "Board", "cell_name", "UNREVEALED", "FLAG", "MINE", "UNKNOWN",
    "solve", "SolveResult", "SolverCache", "EXACT", "ESTIMATED", "PRIOR", "UNCERTAIN",
    "recommend", "Recommendation", "explain_cell",
]
