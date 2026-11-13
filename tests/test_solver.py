"""Solver correctness: every probability is checked against brute force."""
import random
from fractions import Fraction
from itertools import combinations

import pytest

from minesweeper_solver.core import (ESTIMATED, EXACT, FLAG, PRIOR, UNREVEALED, Board,
                                     recommend, solve)
from minesweeper_solver.core.board import cell_name, random_position


def brute_force(board: Board, density=None):
    """Exact mine probability per covered cell by trying every arrangement.
    Returns None if no arrangement is valid."""
    covered = [c for c in board.all_cells() if board[c] == UNREVEALED]
    clues = []
    for cell in board.all_cells():
        v = board[cell]
        if v >= 0:
            nbs = board.neighbors(cell)
            need = v - sum(board[n] in (FLAG, -3) for n in nbs)
            clues.append(([n for n in nbs if board[n] == UNREVEALED], need))
    if board.total_mines is not None:
        counts = [board.total_mines - board.count(FLAG) - board.count(-3)]
        weight = lambda k: Fraction(1)
    else:
        counts = range(len(covered) + 1)
        rho = Fraction(density).limit_denominator(1000)
        rho = rho / (1 - rho)
        weight = lambda k: rho ** k
    total = Fraction(0)
    hits = {c: Fraction(0) for c in covered}
    for k in counts:
        if k < 0 or k > len(covered):
            continue
        for mines in combinations(covered, k):
            ms = set(mines)
            if all(sum(n in ms for n in nbs) == need for nbs, need in clues):
                w = weight(k)
                total += w
                for m in mines:
                    hits[m] += w
    if total == 0:
        return None
    return {c: hits[c] / total for c in covered}


def random_board(rng, known_total=True):
    rows, cols = rng.randint(2, 5), rng.randint(2, 5)
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    n_mines = rng.randint(1, max(1, len(cells) // 3))
    mines = set(rng.sample(cells, n_mines))
    board = Board(rows, cols, n_mines if known_total else None)
    safe = [c for c in cells if c not in mines]
    for c in rng.sample(safe, rng.randint(max(0, len(safe) - 15), len(safe))):
        board[c] = sum(n in mines for n in board.neighbors(c))
    for m in mines:
        if rng.random() < 0.25:
            board[m] = FLAG
    return board


@pytest.mark.parametrize("seed", range(8))
def test_matches_brute_force_with_mine_count(seed):
    rng = random.Random(seed)
    for _ in range(60):
        board = random_board(rng)
        expected = brute_force(board)
        result = solve(board)
        assert result.valid, (board, result.error)
        assert set(result.probs) == set(expected)
        for cell, p in expected.items():
            assert result.probs[cell] == pytest.approx(float(p), abs=1e-12), (board, cell_name(cell))
            assert result.quality[cell] == EXACT
        assert result.safe_cells == sorted(c for c, p in expected.items() if p == 0)
        assert result.mine_cells == sorted(c for c, p in expected.items() if p == 1)


@pytest.mark.parametrize("seed", range(4))
def test_matches_brute_force_with_density_prior(seed):
    rng = random.Random(100 + seed)
    for _ in range(40):
        board = random_board(rng, known_total=False)
        expected = brute_force(board, density=0.2)
        result = solve(board, prior_density=0.2)
        assert result.valid
        for cell, p in expected.items():
            assert result.probs[cell] == pytest.approx(float(p), abs=1e-12), (board, cell_name(cell))
            if 0 < p < 1:
                assert result.quality[cell] == PRIOR


def test_sampling_fallback_is_close_and_never_claims_false_certainty():
    rng = random.Random(7)
    worst = 0.0
    for _ in range(40):
        board = random_board(rng)
        expected = brute_force(board)
        result = solve(board, node_budget=0, samples=20_000)
        assert result.valid
        for cell, p in expected.items():
            got = result.probs[cell]
            worst = max(worst, abs(got - float(p)))
            if got in (0.0, 1.0):          # certainty must be real
                assert p == got
            elif result.quality[cell] != EXACT:
                assert result.quality[cell] == ESTIMATED
    assert worst < 0.05


def test_inconsistent_boards_are_reported_not_guessed():
    for board in (
        Board.from_strings(["1FF", "..."], 5),     # more flags than the clue allows
        Board.from_strings(["0.2"], None),         # clue needs more mines than cells
        Board.from_strings(["1.", ".."], 3),       # clue contradicts the mine total
        Board.from_strings(["F.", ".."], 0),       # more flags than mines
    ):
        result = solve(board)
        assert not result.valid and result.error and not result.probs
        assert recommend(board, result).kind == "invalid"


def test_classic_1_2_1_pattern():
    board = Board.from_strings(["...", "121"], None)
    r = solve(board)
    assert r.mine_cells == [(0, 0), (0, 2)]
    assert r.safe_cells == [(0, 1)]
    assert all(r.quality[c] == EXACT for c in r.probs)


def test_global_mine_count_changes_probabilities():
    lines = ["....", "....", "..21", "..10"]
    low = solve(Board.from_strings(lines, 4))
    high = solve(Board.from_strings(lines, 8))
    interior = (0, 0)
    assert low.probs[interior] == pytest.approx(2 / 7)
    assert high.probs[interior] == pytest.approx(6 / 7)
    assert (1, 1) in low.safe_cells


def test_all_mines_on_frontier_makes_interior_safe():
    board = Board.from_strings(["..1.", "..1.", "111.", "...."], 1)
    r = solve(board)
    expected = brute_force(board)
    for c, p in expected.items():
        assert r.probs[c] == pytest.approx(float(p))
    assert (1, 1) in r.mine_cells
    assert (3, 3) in r.safe_cells and (0, 3) in r.safe_cells


def test_recommendation_prefers_safe_then_lowest_risk():
    board = Board.from_strings(["...", "121"], None)
    rec = recommend(board, solve(board))
    assert rec.kind == "safe" and rec.cell == (0, 1)
    assert rec.mine_cells == [(0, 0), (0, 2)]

    board = Board.from_strings(["....", ".1..", "....", "...."], 3)
    result = solve(board)
    rec = recommend(board, result)
    assert rec.kind == "guess"
    assert result.probs[rec.cell] == pytest.approx(min(result.probs.values()))
    assert rec.explanation


def test_expert_board_is_exact_and_fast():
    import time
    for seed in range(5):
        board = random_position(16, 30, 99, seed=seed)
        start = time.perf_counter()
        result = solve(board)
        elapsed = time.perf_counter() - start
        assert result.valid
        assert elapsed < 5
        expected_mines = sum(result.probs.values())
        if not result.estimated:
            assert expected_mines == pytest.approx(99, abs=1e-6)


def test_expected_mines_equals_remaining_count():
    rng = random.Random(3)
    for _ in range(50):
        board = random_board(rng)
        result = solve(board)
        assert sum(result.probs.values()) == pytest.approx(result.remaining_mines, abs=1e-9)


def test_unknown_cells_never_cause_false_claims():
    """Blank out random cells as UNKNOWN: whatever they really were, every
    remaining safe / mine claim must still hold in the true layout."""
    from minesweeper_solver.core import UNCERTAIN, UNKNOWN
    rng = random.Random(11)
    checked = 0
    for _ in range(300):
        rows, cols = rng.randint(3, 6), rng.randint(3, 6)
        cells = [(r, c) for r in range(rows) for c in range(cols)]
        mines = set(rng.sample(cells, rng.randint(1, len(cells) // 3)))
        board = Board(rows, cols, len(mines))
        for c in cells:
            if c in mines:
                if rng.random() < 0.3:
                    board[c] = FLAG
            elif rng.random() < 0.6:
                board[c] = sum(n in mines for n in board.neighbors(c))
        hidden = rng.sample(cells, rng.randint(1, 4))
        for c in hidden:
            board[c] = UNKNOWN
        result = solve(board)
        assert result.valid, board
        assert result.unknown_count == len(hidden)
        assert not set(result.probs) & set(hidden)
        assert not set(result.safe_cells) & mines, board
        assert set(result.mine_cells) <= mines, board
        assert all(q in (EXACT, UNCERTAIN) for q in result.quality.values())
        checked += len(result.safe_cells) + len(result.mine_cells)
    assert checked > 200


def test_cache_gives_identical_results_while_a_game_progresses():
    """Follow games move by move with one shared cache: every result must be
    identical to an uncached solve, and unchanged regions must be reused."""
    from minesweeper_solver.core import SolverCache

    cache = SolverCache()
    reused = solves = 0
    for seed in range(12):
        rng = random.Random(seed)
        board = Board(16, 16, 40)
        mines = set(rng.sample([c for c in board.all_cells() if max(abs(c[0] - 8), abs(c[1] - 8)) > 1], 40))

        def reveal(cell):
            stack = [cell]
            while stack:
                cur = stack.pop()
                if board[cur] != UNREVEALED:
                    continue
                board[cur] = sum(n in mines for n in board.neighbors(cur))
                if board[cur] == 0:
                    stack.extend(board.neighbors(cur))

        reveal((8, 8))
        for _ in range(60):
            plain = solve(board)
            cached = solve(board, cache=cache)
            solves += 1
            reused += cached.cached_components
            assert cached.probs == plain.probs and cached.quality == plain.quality
            assert cached.safe_cells == plain.safe_cells and cached.mine_cells == plain.mine_cells
            # one step: flag a proven mine (changes the mine count only) or reveal
            if plain.mine_cells and rng.random() < 0.5:
                board[plain.mine_cells[0]] = FLAG
            elif plain.safe_cells:
                reveal(plain.safe_cells[0])
            else:
                guess = min(plain.probs, key=plain.probs.get)
                if guess in mines:
                    break
                reveal(guess)
    assert reused > solves        # most regions were reused, not re-enumerated
    assert cache.hits > cache.misses


def test_cache_respects_a_changed_mine_count():
    from minesweeper_solver.core import SolverCache
    cache = SolverCache()
    lines = ["....", "....", "..21", "..10"]
    for total in (8, 4, 6, 3, 8):         # cached with a large limit, reused with smaller
        expected = solve(Board.from_strings(lines, total))
        got = solve(Board.from_strings(lines, total), cache=cache)
        assert got.probs == expected.probs and got.valid == expected.valid
