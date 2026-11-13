"""Per-digit recognition accuracy on labelled frames from harvest_real_games.py.

    python scripts/evaluate_digits.py DIR [DIR ...]

Reads every frame the way the app does (one BoardReader per recorded game, in
order) and compares each cell with the page's own state.
"""
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from minesweeper_solver.core import UNKNOWN  # noqa: E402
from minesweeper_solver.vision import BoardReader  # noqa: E402

LABEL = {-1: "covered", -2: "flag", -3: "mine", 0: "0"}


def frames(dirs):
    games = collections.defaultdict(list)
    for d in dirs:
        for png in sorted(glob.glob(os.path.join(d, "*.png"))):
            games[os.path.basename(png).rsplit("_", 1)[0]].append(png)
    return games


def main(dirs) -> dict:
    stats = collections.defaultdict(collections.Counter)
    wrong_examples = collections.Counter()
    grid_fail = 0
    for game, pngs in sorted(frames(dirs).items()):
        reader = BoardReader()
        for png in pngs:
            truth = json.load(open(png[:-4] + ".json"))["truth"]
            frame = cv2.imread(png)[:, :, ::-1].copy()
            result = reader.read(frame)
            if not result.ok or result.states.shape != (len(truth), len(truth[0])):
                grid_fail += 1
                continue
            for r, row in enumerate(truth):
                for c, want in enumerate(row):
                    if want is None:
                        continue
                    got = int(result.states[r, c])
                    key = LABEL.get(want, str(want))
                    stats[key]["total"] += 1
                    if got == want:
                        stats[key]["right"] += 1
                    elif got == UNKNOWN:
                        stats[key]["unknown"] += 1
                    else:
                        stats[key]["wrong"] += 1
                        wrong_examples[(key, LABEL.get(got, str(got)))] += 1
    print(f"{'cell':>8} {'total':>7} {'right':>7} {'unknown':>8} {'wrong':>6}  unknown%")
    order = ["covered", "flag", "mine", "0"] + [str(d) for d in range(1, 9)]
    for key in order:
        s = stats.get(key)
        if s:
            print(f"{key:>8} {s['total']:7d} {s['right']:7d} {s['unknown']:8d} {s['wrong']:6d}"
                  f"  {100 * s['unknown'] / s['total']:6.1f}%")
    if wrong_examples:
        print("wrong readings (truth -> read):", dict(wrong_examples))
    print("frames without a board:", grid_fail)
    return stats


if __name__ == "__main__":
    main(sys.argv[1:])
