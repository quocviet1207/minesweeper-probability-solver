"""Minesweeper probability solver.

Pipeline
--------
1. Build one linear constraint per revealed number:
   "exactly N of these covered neighbours are mines" (flags already subtracted).
2. Constraint propagation: trivial rules (all safe / all mines) and the subset
   rule (A subset of B  =>  B-A holds the difference) until a fixpoint.
3. Split the remaining constraints into independent components. Inside each,
   cells that sit in exactly the same constraints are merged into one group, and
   every valid assignment is enumerated by backtracking. For each component we
   keep, per mine count k, the number of arrangements W(k) and for every group
   the number of mines it holds summed over those arrangements G(k).
4. Combine components with the global mine count: an arrangement using j
   frontier mines can be completed in C(I, R-j) ways, where I is the number of
   unconstrained covered cells and R the mines still unaccounted for. All
   arithmetic is exact (Python integers) until the final division.

If a component is too large to enumerate within `node_budget`, it is sampled
instead (weighted random dives through the same search tree) and every
probability that depends on it is labelled ESTIMATED. If the total mine count
is unknown, a mine-density prior is used and results are labelled PRIOR.
Cells reported as certain (0% / 100%) are always logically proven.
"""
from __future__ import annotations

import random
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from fractions import Fraction
from math import comb
from typing import Optional

from .board import FLAG, MINE, UNKNOWN, UNREVEALED, Board, Cell, cell_name

EXACT = "exact"          # mathematically exact
ESTIMATED = "estimated"  # Monte-Carlo estimate (region too large to enumerate)
PRIOR = "prior"          # exact given an *assumed* mine density (total unknown)
UNCERTAIN = "uncertain"  # board contains unrecognised cells, so the value is approximate

DEFAULT_DENSITY = 0.2


@dataclass
class SolveResult:
    valid: bool = True
    error: Optional[str] = None
    probs: dict[Cell, float] = field(default_factory=dict)     # mine probability
    quality: dict[Cell, str] = field(default_factory=dict)     # EXACT/ESTIMATED/PRIOR
    reasons: dict[Cell, str] = field(default_factory=dict)
    safe_cells: list[Cell] = field(default_factory=list)       # proven safe
    mine_cells: list[Cell] = field(default_factory=list)       # proven mines (unflagged)
    frontier: set[Cell] = field(default_factory=set)           # covered cells touching a number
    unrevealed_count: int = 0        # covered, unflagged cells
    flag_count: int = 0
    unknown_count: int = 0           # cells recognition could not identify
    total_mines: Optional[int] = None
    remaining_mines: Optional[int] = None   # total - flags (None if total unknown)
    used_mine_count: bool = False
    prior_density: Optional[float] = None
    components: int = 0
    region_of: dict[Cell, int] = field(default_factory=dict)   # cell -> index into regions
    regions: list[dict] = field(default_factory=list)          # cells, clues, arrangements, exact
    cached_components: int = 0              # regions reused from a SolverCache
    estimated_components: int = 0           # regions too large to enumerate exactly
    arrangements: Optional[int] = None      # frontier arrangements enumerated (exact only)
    estimated: bool = False
    notes: list[str] = field(default_factory=list)

    def safe_prob(self, cell: Cell) -> float:
        return 1.0 - self.probs[cell]


class _Inconsistent(Exception):
    pass


class _BudgetExceeded(Exception):
    pass


def _names(cells, limit: int = 4) -> str:
    cells = sorted(cells)
    text = ", ".join(cell_name(c) for c in cells[:limit])
    if len(cells) > limit:
        text += f" and {len(cells) - limit} more"
    return text


# ---------------------------------------------------------------------------
# 1. constraints
# ---------------------------------------------------------------------------
def _add(cons: dict, cells: frozenset, mines: int, sources: frozenset) -> None:
    if mines < 0 or mines > len(cells):
        raise _Inconsistent(
            f"The clues at {_names(sources)} contradict each other "
            f"({mines} mines required in {len(cells)} cells)."
        )
    if not cells:
        return
    old = cons.get(cells)
    if old is None:
        cons[cells] = (mines, sources)
    elif old[0] != mines:
        raise _Inconsistent(
            f"The clues at {_names(old[1] | sources)} contradict each other."
        )
    elif len(sources) < len(old[1]):
        cons[cells] = (mines, sources)


def _build_constraints(board: Board) -> dict:
    cons: dict = {}
    for cell in board.all_cells():
        value = board[cell]
        if value < 0:
            continue
        covered, known = [], 0
        for nb in board.neighbors(cell):
            state = board[nb]
            if state in (UNREVEALED, UNKNOWN):
                covered.append(nb)
            elif state in (FLAG, MINE):
                known += 1
        need = value - known
        if need < 0:
            raise _Inconsistent(
                f"{cell_name(cell)} shows {value} but has {known} flagged neighbours."
            )
        if need > len(covered):
            raise _Inconsistent(
                f"{cell_name(cell)} shows {value} but only {known + len(covered)} "
                f"of its neighbours can be mines."
            )
        _add(cons, frozenset(covered), need, frozenset([cell]))
    return cons


# ---------------------------------------------------------------------------
# 2. propagation
# ---------------------------------------------------------------------------
def _propagate(cons: dict, determined: dict[Cell, bool], reasons: dict[Cell, str]) -> dict:
    """Reduce `cons` to a fixpoint; fills `determined` (cell -> is_mine)."""
    while True:
        new: dict[Cell, bool] = {}
        for cells, (mines, src) in cons.items():
            if mines == 0:
                is_mine = False
            elif mines == len(cells):
                is_mine = True
            else:
                continue
            for c in cells:
                if new.get(c, is_mine) != is_mine:
                    raise _Inconsistent(
                        f"The clues around {cell_name(c)} contradict each other."
                    )
                if c not in new:
                    new[c] = is_mine
                    reasons[c] = _propagation_reason(src, is_mine, len(cells))
        if new:
            determined.update(new)
            rebuilt: dict = {}
            for cells, (mines, src) in cons.items():
                hit = cells & new.keys()
                if hit:
                    mines -= sum(new[c] for c in hit)
                    cells = cells - hit
                    if not cells and mines != 0:
                        raise _Inconsistent(
                            f"The clues at {_names(src)} contradict each other."
                        )
                _add(rebuilt, cells, mines, src)
            cons = rebuilt
            continue

        # subset rule: A < B  =>  replace B by B - A
        index: dict[Cell, list] = {}
        for cells in cons:
            for c in cells:
                index.setdefault(c, []).append(cells)
        reduced = False
        for a in sorted(cons, key=len):
            if a not in cons:
                continue
            mines_a, src_a = cons[a]
            for b in index[next(iter(a))]:
                if len(b) <= len(a) or b not in cons or not a < b:
                    continue
                mines_b, src_b = cons.pop(b)
                _add(cons, b - a, mines_b - mines_a, src_a | src_b)
                reduced = True
        if not reduced:
            return cons


def _propagation_reason(sources: frozenset, is_mine: bool, n_cells: int) -> str:
    if len(sources) == 1:
        name = cell_name(next(iter(sources)))
        if is_mine:
            return (f"{name} still needs {n_cells} mine{'s' if n_cells != 1 else ''} and has "
                    f"exactly {n_cells} covered neighbour{'s' if n_cells != 1 else ''} left, "
                    f"so {'they' if n_cells != 1 else 'it'} must be mined.")
        return (f"All mines around {name} are already accounted for, "
                f"so its remaining covered neighbours are safe.")
    if is_mine:
        return (f"Combining the clues at {_names(sources)}: the mines they require "
                f"can only fit here, so this cell is a mine.")
    return (f"Combining the clues at {_names(sources)}: the mines they require "
            f"must lie elsewhere, so this cell is safe.")


# ---------------------------------------------------------------------------
# 3. components and enumeration
# ---------------------------------------------------------------------------
@dataclass
class _Component:
    groups: list[list[Cell]]
    sizes: list[int]
    cons: list[tuple[list[int], int]]      # (group indices, mines needed)
    gcons: list[list[int]]                 # group -> constraint indices
    order: list[int]
    W: dict[int, int] = field(default_factory=dict)          # k -> weight
    G: list[dict[int, int]] = field(default_factory=list)    # group -> k -> mines*weight
    exact: bool = True


def _order_groups(sizes: list[int], gcons: list[list[int]], cons: list) -> list[int]:
    """Processing order that keeps few constraints "open" at a time.

    A frontier is usually a long, thin band of cells. Starting at one end
    (a pseudo-peripheral group) and always taking next the neighbour that
    closes the most constraints and opens the fewest walks along the band, so
    the dynamic programme in `_enumerate` only ever tracks a handful of
    partially satisfied clues.
    """
    n = len(sizes)
    adj: list[set] = [set() for _ in range(n)]
    for gs, _ in cons:
        for a in gs:
            adj[a].update(gs)
    for a in range(n):
        adj[a].discard(a)

    def bfs(src: int) -> tuple[list[int], int]:
        dist = [-1] * n
        dist[src] = 0
        queue, far = deque([src]), src
        while queue:
            g = queue.popleft()
            far = g
            for h in adj[g]:
                if dist[h] < 0:
                    dist[h] = dist[g] + 1
                    queue.append(h)
        return dist, far

    _, start = bfs(0)
    dist, _ = bfs(start)
    remaining = [len(gs) for gs, _ in cons]
    touched = [False] * len(cons)
    done = [False] * n
    order: list[int] = []
    frontier = {start}
    while len(order) < n:
        if not frontier:                       # not reached for connected components
            frontier = {next(g for g in range(n) if not done[g])}

        def score(g: int) -> tuple:
            opened = sum(1 for c in gcons[g] if not touched[c] and remaining[c] > 1)
            closed = sum(1 for c in gcons[g] if remaining[c] == 1)
            return (opened - closed, dist[g], g)

        g = min(frontier, key=score)
        frontier.discard(g)
        done[g] = True
        order.append(g)
        for c in gcons[g]:
            touched[c] = True
            remaining[c] -= 1
        frontier.update(h for h in adj[g] if not done[h])
    return order


def _split_components(cons: dict) -> list[_Component]:
    parent: dict[Cell, Cell] = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for cells in cons:
        first = None
        for c in cells:
            parent.setdefault(c, c)
            if first is None:
                first = c
            else:
                parent[find(c)] = find(first)

    by_root: dict[Cell, list] = {}
    for cells in cons:
        by_root.setdefault(find(next(iter(cells))), []).append(cells)

    components = []
    for con_list in by_root.values():
        con_list.sort(key=lambda cs: sorted(cs))
        membership: dict[Cell, list[int]] = {}
        for ci, cells in enumerate(con_list):
            for c in cells:
                membership.setdefault(c, []).append(ci)
        group_of: dict[tuple, int] = {}
        groups: list[list[Cell]] = []
        gcons: list[list[int]] = []
        for c in sorted(membership):
            sig = tuple(membership[c])
            gi = group_of.get(sig)
            if gi is None:
                gi = group_of[sig] = len(groups)
                groups.append([])
                gcons.append(list(sig))
            groups[gi].append(c)
        comp_cons: list[tuple[list[int], int]] = [([], cons[cells][0]) for cells in con_list]
        for gi, cis in enumerate(gcons):
            for ci in cis:
                comp_cons[ci][0].append(gi)

        order = _order_groups([len(g) for g in groups], gcons, comp_cons)
        components.append(_Component(groups, [len(g) for g in groups], comp_cons, gcons, order))
    components.sort(key=lambda comp: comp.groups[0][0])
    return components


class SolverCache:
    """Remembers the exact enumeration of each independent region of the board.

    Between two frames of a live game most regions are untouched: only the
    area around the newly revealed cells changes. A region is identified by its
    cells and their constraints, so an unchanged region is looked up instead of
    enumerated again. The global mine count only enters afterwards, when the
    regions are combined, so a cached region stays valid when flags or
    revelations elsewhere change the number of mines left.
    """

    def __init__(self, max_entries: int = 512):
        self.max_entries = max_entries
        self._entries: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    @staticmethod
    def _key(comp: "_Component") -> tuple:
        return (tuple(tuple(g) for g in comp.groups),
                tuple((tuple(gs), m) for gs, m in comp.cons))

    def load(self, comp: "_Component", kmax: int) -> bool:
        """Fill `comp.W` / `comp.G` from the cache; False if not available."""
        key = self._key(comp)
        entry = self._entries.get(key)
        # an enumeration capped at `limit` mines is reusable for any kmax up
        # to that limit, or for any kmax at all if the cap never bit
        if entry is None or not (entry[3] or entry[2] >= kmax):
            self.misses += 1
            return False
        W, G, _, _ = entry
        comp.W = {k: v for k, v in W.items() if k <= kmax}
        comp.G = [{k: v for k, v in g.items() if k <= kmax} for g in G]
        comp.exact = True
        self._entries.move_to_end(key)
        self.hits += 1
        return True

    def store(self, comp: "_Component", kmax: int) -> None:
        if not comp.exact:
            return                       # sampled results depend on the random draw
        complete = kmax >= sum(comp.sizes)
        self._entries[self._key(comp)] = (dict(comp.W), [dict(g) for g in comp.G], kmax, complete)
        self._entries.move_to_end(self._key(comp))
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)


def _enumerate(comp: _Component, kmax: int, budget: list[int]) -> None:
    """Exact count of every valid arrangement of a region, by dynamic programming.

    Groups are processed in `comp.order`. After each step the only thing that
    matters for the rest of the region is how many mines each clue that is
    still "open" (touched by processed and unprocessed groups) still needs.
    Arrangements leading to the same open-clue state are merged, keeping a
    polynomial in the number of mines used so far: {mines: weighted count}.
    A forward pass counts prefixes, a backward pass counts completions, and
    their product gives, for every group, the weighted number of mines it
    holds per total mine count (`comp.G`). Cost grows with the number of open
    clues, which stays small along a typical frontier. `budget` is a shared
    one-element work counter; running out raises `_BudgetExceeded`.
    """
    sizes, gcons, order = comp.sizes, comp.gcons, comp.order
    n = len(order)
    need0 = [m for _, m in comp.cons]
    pos = [0] * n
    for i, g in enumerate(order):
        pos[g] = i
    first = [min(pos[g] for g in gs) for gs, _ in comp.cons]
    last = [max(pos[g] for g in gs) for gs, _ in comp.cons]
    active = [[c for c in range(len(comp.cons)) if first[c] < i <= last[c]] for i in range(n + 1)]
    # capacity each clue of the group at position i still has after that group
    after = [{c: sum(sizes[h] for h in comp.cons[c][0] if pos[h] > i) for c in gcons[order[i]]}
             for i in range(n)]

    def spend(units: int) -> None:
        budget[0] -= units
        if budget[0] < 0:
            raise _BudgetExceeded

    def transitions(i: int, state: tuple) -> list:
        g = order[i]
        s = sizes[g]
        needs = dict(zip(active[i], state))
        for c in gcons[g]:
            if first[c] == i:
                needs[c] = need0[c]
        lo, hi = 0, s
        for c in gcons[g]:
            r = needs[c]
            lo = max(lo, r - after[i][c])
            hi = min(hi, r)
        mine = set(gcons[g])
        return [(j, tuple(needs[c] - (j if c in mine else 0) for c in active[i + 1]))
                for j in range(lo, hi + 1)]

    forward: list[dict] = [{(): {0: 1}}]
    steps: list[dict] = []
    for i in range(n):
        s = sizes[order[i]]
        nxt: dict = {}
        layer: dict = {}
        for state, poly in forward[i].items():
            options = transitions(i, state)
            layer[state] = options
            for j, state2 in options:
                spend(len(poly))
                w = comb(s, j)
                target = nxt.setdefault(state2, {})
                for k, v in poly.items():
                    kk = k + j
                    if kk <= kmax:
                        target[kk] = target.get(kk, 0) + v * w
        steps.append(layer)
        forward.append({st: p for st, p in nxt.items() if p})

    backward: list[dict] = [{} for _ in range(n)] + [{(): {0: 1}}]
    for i in range(n - 1, -1, -1):
        s = sizes[order[i]]
        for state, options in steps[i].items():
            acc: dict = {}
            for j, state2 in options:
                b = backward[i + 1].get(state2)
                if not b:
                    continue
                spend(len(b))
                w = comb(s, j)
                for k, v in b.items():
                    kk = k + j
                    if kk <= kmax:
                        acc[kk] = acc.get(kk, 0) + v * w
            if acc:
                backward[i][state] = acc

    G: list[dict[int, int]] = [{} for _ in range(n)]
    for i in range(n):
        g = order[i]
        s = sizes[g]
        Gg = G[g]
        for state, options in steps[i].items():
            f = forward[i].get(state)
            if not f:
                continue
            for j, state2 in options:
                b = backward[i + 1].get(state2) if j else None
                if not b:
                    continue
                spend(len(f) * len(b))
                w = j * comb(s, j)
                for k1, v1 in f.items():
                    for k2, v2 in b.items():
                        kk = k1 + j + k2
                        if kk <= kmax:
                            Gg[kk] = Gg.get(kk, 0) + w * v1 * v2
    comp.W, comp.G, comp.exact = dict(forward[n].get((), {})), G, True


def _sample(comp: _Component, kmax: int, samples: int, rng: random.Random) -> None:
    """Importance-sampled estimate of W and G (same scale factor for both, so
    the factor cancels in every probability)."""
    sizes, gcons, order = comp.sizes, comp.gcons, comp.order
    need0 = [m for _, m in comp.cons]
    cap0 = [sum(sizes[g] for g in gs) for gs, _ in comp.cons]
    n = len(order)
    W: dict[int, int] = {}
    G: list[dict[int, int]] = [{} for _ in range(n)]
    assign = [0] * n
    for _ in range(samples):
        need, cap = need0[:], cap0[:]
        weight, k = 1, 0
        for g in order:
            s = sizes[g]
            lo, hi = 0, min(s, kmax - k)
            for ci in gcons[g]:
                cap[ci] -= s
                d = need[ci] - cap[ci]
                if d > lo:
                    lo = d
                if need[ci] < hi:
                    hi = need[ci]
            if lo > hi:
                weight = 0
                break
            if lo == hi:
                j = lo
                weight *= comb(s, j)
            else:
                choices = [comb(s, x) for x in range(lo, hi + 1)]
                weight *= sum(choices)
                j = rng.choices(range(lo, hi + 1), choices)[0]
            for ci in gcons[g]:
                need[ci] -= j
            assign[g] = j
            k += j
        if weight:
            W[k] = W.get(k, 0) + weight
            for g in range(n):
                if assign[g]:
                    G[g][k] = G[g].get(k, 0) + weight * assign[g]
    comp.W, comp.G, comp.exact = W, G, False


# ---------------------------------------------------------------------------
# 4. combination
# ---------------------------------------------------------------------------
def _convolve(a: list[int], b: list[int], limit: int) -> list[int]:
    out = [0] * min(limit + 1, len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                if i + j > limit:
                    break
                if y:
                    out[i + j] += x * y
    return out or [0]


def _poly(d: dict[int, int], limit: int) -> list[int]:
    out = [0] * (min(max(d, default=0), limit) + 1)
    for k, v in d.items():
        if k <= limit:
            out[k] = v
    return out


def solve(
    board: Board,
    *,
    node_budget: int = 3_000_000,
    samples: int = 4_000,
    prior_density: float = DEFAULT_DENSITY,
    seed: int = 0,
    cache: Optional[SolverCache] = None,
) -> SolveResult:
    """Mine probability for every covered, unflagged cell of `board`.

    Pass the same `cache` on every call while a game is being followed to
    reuse the enumeration of regions that did not change."""
    res = SolveResult(total_mines=board.total_mines)
    covered = [c for c in board.all_cells() if board[c] in (UNREVEALED, UNKNOWN)]
    unknown = {c for c in covered if board[c] == UNKNOWN}
    res.unrevealed_count = len(covered) - len(unknown)
    res.unknown_count = len(unknown)
    res.flag_count = board.count(FLAG)
    known_mines = res.flag_count + board.count(MINE)
    if board.total_mines is not None:
        res.remaining_mines = board.total_mines - known_mines

    try:
        _solve_into(res, board, covered, known_mines, node_budget, samples, prior_density, seed,
                    cache)
    except _Inconsistent as exc:
        res.valid = False
        res.error = str(exc)
        res.probs.clear()
        res.quality.clear()
        res.reasons.clear()
        res.safe_cells.clear()
        res.mine_cells.clear()
        res.regions.clear()
        res.region_of.clear()
    if unknown and res.valid:
        # Unknown cells took part as "covered, may hold a mine" so every proof
        # stays sound, but they get no verdict of their own and the remaining
        # percentages are only approximate.
        for cell in unknown:
            res.probs.pop(cell, None)
            res.quality.pop(cell, None)
            res.reasons.pop(cell, None)
        res.safe_cells = [c for c in res.safe_cells if c not in unknown]
        res.mine_cells = [c for c in res.mine_cells if c not in unknown]
        for cell, p in res.probs.items():
            if 0.0 < p < 1.0 and res.quality[cell] == EXACT:
                res.quality[cell] = UNCERTAIN
        res.notes.append(
            f"{len(unknown)} cell{'s were' if len(unknown) != 1 else ' was'} not recognised. "
            f"Proven safe cells and mines are still reliable; other percentages are approximate."
        )
    return res


def _solve_into(res, board, covered, known_mines, node_budget, samples, prior_density, seed,
                cache=None):
    total = board.total_mines
    if total is not None:
        if total < known_mines:
            raise _Inconsistent(
                f"{known_mines} mines are flagged but the board only has {total}."
            )
        if total - known_mines > len(covered):
            raise _Inconsistent(
                f"{total - known_mines} mines remain but only {len(covered)} covered cells are left."
            )

    cons = _build_constraints(board)
    res.frontier = set().union(*cons) if cons else set()
    determined: dict[Cell, bool] = {}
    cons = _propagate(cons, determined, res.reasons)

    for cell, is_mine in determined.items():
        res.probs[cell] = 1.0 if is_mine else 0.0
        res.quality[cell] = EXACT
    determined_mines = sum(determined.values())

    constrained = set().union(*cons) if cons else set()
    interior = [c for c in covered if c not in determined and c not in constrained]
    n_int = len(interior)

    use_count = total is not None
    res.used_mine_count = use_count
    if use_count:
        R = total - known_mines - determined_mines
        if R < 0:
            raise _Inconsistent(
                "The clues require more mines than the board's total mine count."
            )
        kmax = R
    else:
        R = None
        kmax = len(constrained)
        res.prior_density = prior_density

    components = _split_components(cons)
    res.components = len(components)
    budget = [node_budget]
    rng = random.Random(seed)
    arrangements = 1
    # small regions first: a single huge region must not use up the budget
    # and push every region after it into sampling
    for comp in sorted(components, key=lambda c: (sum(c.sizes), len(c.cons))):
        if cache is not None and cache.load(comp, kmax):
            res.cached_components += 1
        else:
            try:
                _enumerate(comp, kmax, budget)
                if cache is not None:
                    cache.store(comp, kmax)
            except _BudgetExceeded:
                budget[0] = 0
                _sample(comp, kmax, samples, rng)
                res.estimated = True
                res.estimated_components += 1
        if not comp.W:
            if comp.exact:
                raise _Inconsistent(
                    f"No mine arrangement satisfies the clues around {_names(comp.groups[0])}."
                )
            raise _Inconsistent(
                "The region around " + _names(comp.groups[0]) +
                " is too large to analyse (sampling found no valid arrangement)."
            )
        if comp.exact:
            arrangements *= sum(comp.W.values())
    for index, comp in enumerate(components):
        res.regions.append({
            "cells": sum(comp.sizes),
            "clues": len(comp.cons),
            "arrangements": sum(comp.W.values()) if comp.exact else None,
            "exact": comp.exact,
        })
        for cells in comp.groups:
            for cell in cells:
                res.region_of[cell] = index
    res.arrangements = None if res.estimated else arrangements
    all_exact = not res.estimated

    def finish(cell: Cell, num: int, den: int, certain_ok: bool, base_quality: str, reason: str):
        if certain_ok and num == 0:
            res.probs[cell], res.quality[cell] = 0.0, EXACT
        elif certain_ok and num == den:
            res.probs[cell], res.quality[cell] = 1.0, EXACT
        else:
            p = float(Fraction(num, den))
            # an estimate must never masquerade as a certainty
            res.probs[cell] = min(max(p, 0.001), 0.999) if (num == 0 or num == den) else p
            res.quality[cell] = base_quality
        res.reasons[cell] = reason

    def local_certainty(comp: _Component, g: int) -> Optional[bool]:
        if not comp.exact:
            return None
        if not comp.G[g]:
            return False
        s = comp.sizes[g]
        if all(comp.G[g].get(k, 0) == s * w for k, w in comp.W.items()):
            return True
        return None

    if use_count:
        polys = [_poly(comp.W, R) for comp in components]
        prefix = [[1]]
        for p in polys:
            prefix.append(_convolve(prefix[-1], p, R))
        suffix = [[1]]
        for p in reversed(polys):
            suffix.append(_convolve(suffix[-1], p, R))
        suffix.reverse()
        everything = prefix[-1]
        grand = sum(w * comb(n_int, R - j) for j, w in enumerate(everything) if w)
        if grand == 0:
            raise _Inconsistent(
                f"No mine arrangement fits both the clues and the mine count "
                f"({R} mines left to place)."
            )
        for ci, comp in enumerate(components):
            others = _convolve(prefix[ci], suffix[ci + 1], R)
            tail = {}
            for t in comp.W:
                if t <= R:
                    tail[t] = sum(
                        w * comb(n_int, R - t - j)
                        for j, w in enumerate(others) if w and R - t - j >= 0
                    )
            for g, cells in enumerate(comp.groups):
                s = comp.sizes[g]
                num = sum(v * tail.get(t, 0) for t, v in comp.G[g].items())
                den = s * grand
                local = local_certainty(comp, g)
                for cell in cells:
                    if local is not None:
                        num_c = den if local else 0
                        finish(cell, num_c, den, True, EXACT, _enum_reason(local, False))
                    else:
                        certain = all_exact and num in (0, den)
                        reason = (_enum_reason(num == den, True) if certain
                                  else _uncertain_reason(num / den if den else 0, comp.exact and all_exact, True))
                        finish(cell, num, den, all_exact,
                               EXACT if all_exact else ESTIMATED, reason)
        if n_int:
            num = sum(
                w * comb(n_int - 1, R - j - 1)
                for j, w in enumerate(everything) if w and R - j - 1 >= 0
            )
            p = float(Fraction(num, grand))
            if all_exact and num == 0:
                reason = ("Every remaining mine is accounted for next to the revealed "
                          "numbers, so cells away from them are safe.")
            elif all_exact and num == grand:
                reason = ("The mine count forces every cell away from the revealed "
                          "numbers to be a mine.")
            else:
                reason = (f"Not next to any revealed number. On average {p * n_int:.1f} of the "
                          f"{R} unplaced mines fall among the {n_int} such cells, "
                          f"so each has the same {p * 100:.1f}% risk.")
            for cell in interior:
                finish(cell, num, grand, all_exact, EXACT if all_exact else ESTIMATED, reason)
        res.notes.append(
            f"Global mine count used: {R} mine{'s' if R != 1 else ''} still to place among "
            f"{len(constrained) + n_int} undetermined cells."
        )
    else:
        rho = Fraction(prior_density).limit_denominator(1000)
        rho = rho / (1 - rho)
        for comp in components:
            powers = {k: rho ** k for k in comp.W}
            den_base = sum(w * powers[k] for k, w in comp.W.items())
            for g, cells in enumerate(comp.groups):
                s = comp.sizes[g]
                num = sum(v * powers[k] for k, v in comp.G[g].items())
                den = s * den_base
                local = local_certainty(comp, g)
                frac = Fraction(num) / Fraction(den)
                for cell in cells:
                    if local is not None:
                        finish(cell, 1 if local else 0, 1, True, EXACT, _enum_reason(local, False))
                    else:
                        finish(cell, frac.numerator, frac.denominator, False,
                               PRIOR if comp.exact else ESTIMATED,
                               _uncertain_reason(float(frac), comp.exact, False))
        for cell in interior:
            res.probs[cell] = prior_density
            res.quality[cell] = PRIOR
            res.reasons[cell] = (
                f"Not next to any revealed number and the total mine count is unknown, "
                f"so the assumed mine density ({prior_density * 100:.0f}%) is shown."
            )
        res.notes.append(
            f"Total mine count unknown: probabilities assume a mine density of "
            f"{prior_density * 100:.0f}%. Set the total for exact results."
        )

    if res.estimated:
        res.notes.append(
            "At least one region was too large to enumerate exactly; its "
            "probabilities are sampled estimates (marked with ~)."
        )
    elif components:
        res.notes.append(
            f"{len(components)} independent region{'s' if len(components) != 1 else ''}, "
            f"{arrangements:,} valid frontier arrangement{'s' if arrangements != 1 else ''} "
            f"enumerated exactly"
            + (f" ({res.cached_components} reused unchanged from the previous update)."
               if res.cached_components else ".")
        )

    res.safe_cells = sorted(c for c, p in res.probs.items() if p == 0.0)
    res.mine_cells = sorted(c for c, p in res.probs.items() if p == 1.0)


def _enum_reason(is_mine: bool, needs_count: bool) -> str:
    extra = " once the total mine count is taken into account" if needs_count else ""
    if is_mine:
        return f"This cell is a mine in every valid arrangement of the surrounding clues{extra}."
    return f"This cell is a mine in no valid arrangement of the surrounding clues{extra}."


def _uncertain_reason(p: float, exact: bool, with_count: bool) -> str:
    how = "is a mine in" if exact else "is estimated (by sampling) to be a mine in"
    weight = (", weighted by how many ways the rest of the board can hold the remaining mines"
              if with_count else ", weighted by the assumed mine density")
    return f"This cell {how} {p * 100:.1f}% of the valid arrangements{weight}."
