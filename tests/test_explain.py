"""Per-cell explanations."""
from minesweeper_solver.core import UNKNOWN, Board, explain_cell, recommend, solve


def explain(board, cell):
    result = solve(board)
    return dict(explain_cell(board, result, recommend(board, result), cell))


def test_region_size_and_arrangement_count():
    board = Board.from_strings(["...", ".1.", "..."], 1)
    sections = explain(board, (0, 0))
    assert "12.5% mine" in sections["Probability"] and "exact" in sections["Probability"]
    assert "region of 8 covered cells tied together by 1 clues" in sections["Region"]
    assert "8 different mine arrangements" in sections["Region"]
    assert "B2 shows 1 · needs 1 more among 8 covered: A1 (this cell)" in sections["Clues around it"]


def test_proven_cells_name_the_clues_and_what_they_prove():
    board = Board.from_strings(["...", "121"], None)
    mine = explain(board, (0, 0))
    assert mine["Probability"].startswith("100.0% mine")
    assert "A2" in mine["Why"]
    assert "proven mines: A1, C1" in mine["Clues around it"]
    assert "proven safe: B1" in mine["Clues around it"]
    safe = explain(board, (0, 1))
    assert safe["Probability"].startswith("0.0% mine") and "Ranking" not in safe


def test_whole_board_comparison_and_ranking():
    board = Board.from_strings(["....", "....", "..21", "..10"], 4)
    far = explain(board, (0, 0))
    assert "Clues around it" in far and far["Clues around it"].startswith("None")
    assert "average of" in far["Whole board"]
    assert "undecided cells" in far["Ranking"]
    revealed = explain(board, (2, 2))
    assert list(revealed) == ["Clue"] and revealed["Clue"].startswith("C3 shows 2")


def test_flags_unknown_cells_and_invalid_boards():
    board = Board.from_strings(["F1.", "11.", "..."], 1)
    assert "Flags are trusted" in explain(board, (0, 0))["Flag"]
    board[(2, 2)] = UNKNOWN
    assert "Not recognised" in explain(board, (2, 2))
    bad = Board.from_strings(["FF.", "1..", "..."], 3)     # A2 shows 1 next to two flags
    sections = explain(bad, (0, 0))
    assert "too many flags" in sections["Flag"]
    assert "No analysis" in explain(bad, (2, 2))
