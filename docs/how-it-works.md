# How it works

The detailed version. For the short one, see the [README](../README.md).

## The probability solver

This is the heart of the project, and the part I care most about getting right.

### The goal

For every covered cell: *of all the ways the remaining mines could be placed
that agree with everything visible, in what fraction is this cell a mine?*
Every placement that fits the clues and the total mine count is equally likely,
so that fraction is the exact probability.

### Why "look at the nearest number" is not enough

```
    A  B  C  D
1   .  .  .  .
2   2  .  .  .        3 mines in total
3   .  .  .  .        "." = covered
4   1  .  .  .
```

| Cell | Naive guess from the nearest number | Exact |
|---|---|---|
| A1, B1, B2 | A2 needs 2 of 5 → 40% | **35.3%** |
| A3, B3 | 40% or 33%? | **47.1%** |
| B4 | A4 needs 1 of 3 → 33% | **5.9%** |
| C1, D1, D4, … (no clue) | ? | **11.8%** |

A3 and B3 touch *both* clues, so a mine there satisfies both at once. With only
3 mines on the whole board, arrangements that do that are far more numerous,
which makes B4 almost safe. Counting all 51 valid arrangements gives the exact
numbers above (the tests confirm them by brute force).

### Step 1: turn clues into constraints

Every revealed number becomes an equation: *exactly N mines among these covered
neighbours* (flags are subtracted from N). A2 above becomes
`A1 + B1 + B2 + A3 + B3 = 2`.

### Step 2: propagation

Cheap logical deductions first, repeated until nothing changes:

- a constraint needing 0 mines → all its cells are safe;
- a constraint needing as many mines as it has cells → all are mines;
- **subset rule**: if constraint A's cells are a subset of B's, then B minus A
  holds the difference. This solves patterns like `1-2-1`:

```
.  .  .        →    *  .  *
1  2  1             1  2  1        (the corners are mines, the middle is safe)
```

Cells settled here get 0% or 100%, the only values the app ever shows as certain.

### Step 3: split into independent regions

Constraints that share no cells don't affect each other, so the covered cells
next to numbers (the *frontier*) split into independent regions. Cells inside a
region that belong to exactly the same constraints are interchangeable, so they
are merged into one *group*. "3 cells, 1 mine" is counted as `C(3,1) = 3` ways at
once instead of three separate cases.

### Step 4: count every arrangement in a region (dynamic programming)

Trying every combination explodes quickly, so each region is counted with a
dynamic programme that walks along the frontier:

- Groups are processed one at a time, in an order that keeps few clues
  "half-filled" at once. A frontier is usually a long, thin band, so it starts
  at one end and walks along it.
- After each step, the only thing that matters for the rest of the region is
  **how many mines each open clue still needs**. Partial arrangements that leave
  the same needs are merged.
- Each merged state keeps a small polynomial `{mines used so far: number of ways}`,
  because the total mine count matters later.
- A forward pass counts the ways to reach each state, and a backward pass counts
  the ways to finish from it. Multiplying the two gives, for every group, how many
  arrangements put a mine there, broken down by the total mines used.

All counting uses exact integers.

### Step 5: bring in the global mine count

Cells that touch no number (the *interior*) are only constrained by the total.
If the frontier uses `k` mines, the remaining `R − k` mines can sit among the `I`
interior cells in `C(I, R − k)` ways. So each frontier arrangement is weighted by
that binomial:

```
                Σₖ  (arrangements with a mine in the cell, using k mines) · C(I, R − k)
P(mine) =   ─────────────────────────────────────────────────────────────────────────
                Σₖ  (all arrangements using k mines) · C(I, R − k)
```

Independent regions are combined by convolving their polynomials. Every interior
cell gets the same probability: the expected number of interior mines divided
by `I`.

### Step 6: knowing what it doesn't know

Every probability carries a label:

| Label | Meaning |
|---|---|
| exact | mathematically exact |
| estimated `~` | a region was too large to count within the effort budget, so it was sampled; only that region is affected |
| prior `~` | the total mine count is unknown, so an assumed density (20% by default) is used |
| uncertain `~` | some cells could not be read from the screen (treated safely, see below) |

0% and 100% are only shown for cells that are logically proven. If no
arrangement fits the board at all (a wrong flag, a misread number), the app says
the board is inconsistent instead of guessing.

### Is it right? Is it fast?

- **Correctness:** on 640 random small boards every probability equals a
  brute-force enumeration exactly. In a stress run the solver played 360 full
  games against hidden mine layouts without ever calling a mine safe.
- **Speed:** a typical expert position takes a few milliseconds. On 100
  deliberately nasty positions (scattered openings on boards up to 30×50) the
  median is 37 ms. Only 6 needed an estimate for one region; a simple
  backtracking search needed 77.
- **Live games:** a cache remembers every region, so after a move only the
  regions that changed are counted again.

### Picking the move

A proven-safe cell if one exists; among several, the one most likely to be a 0
and open up more of the board. Otherwise the lowest-probability cell, with the
same opening chance as a tie-breaker. Proven mines are listed for flagging.


## The rest of the system

**Reading the screen.** The board's grid is found from the repeating pattern of
edges, with no theme knowledge. Cells are grouped by look, and which group is
"covered" vs "revealed" is decided by game rules (a revealed blank never touches
an unflagged covered cell, numbers are single-colour digits, raised tiles are
covered) rather than by colour. Digits are matched against shapes from many
fonts plus samples collected from real games. Anything it can't read
confidently becomes `?` instead of a guess. A `?` cell is treated as "covered,
might be a mine", so proven safe cells stay safe whatever it really is. If a
digit shows as `?`, select it and type the number: the app learns that shape.

**Staying in sync.** Capture runs in the background and only passes on a picture
once the board has stopped moving. Recognition skips frames where no cell
changed, and re-reads only what did. A small state machine keeps revealed
numbers stable through hover effects and notices when a new game starts.
Analysis runs on its own thread, so the window never freezes.

**Auto-play.** Reveals one guaranteed-safe cell at a time and waits until the
screen shows the result before the next move. Optionally it also flags mines,
or guesses when stuck. It stops when the game ends, when something looks wrong,
or as soon as you touch the mouse. Some sites don't allow bots, so check before
using it there.

**Settings and explanations.** Capture interval, solver effort, a colour-blind
palette, auto-play options and more, all remembered between runs. The window's
side panels can be resized or hidden.


## Project layout

```
minesweeper_solver/
  core/      board model, probability solver, move recommendation, explanations  (pure Python)
  vision/    screenshot → grid → cell states; digit templates                    (no GUI)
  capture/   screen capture thread, change and "settled" detection
  state/     keeps the board in sync across frames, detects new games
  control/   mouse control and the auto-play move choice
  ui/        PySide6 desktop app
tests/       solver vs brute force, self-play, vision on synthetic themes, live sync, UI
scripts/     screenshots, digit template building, recording and evaluating real games
```

Data flows one way: `capture → vision → state → solver → recommendation → ui`.

## Tests

```bash
.venv\Scripts\python -m pytest
```

91 tests, about a minute. Highlights: the solver checked against brute force,
games played move by move from rendered screenshots, recognition across several
synthetic visual styles and board sizes, and auto-play driven by a fake mouse.

