"""Builds minesweeper_solver/vision/digit_samples.npz from labelled real frames.

    python scripts/train_digits.py TRAIN_DIR [TRAIN_DIR ...]

Frames and labels come from harvest_real_games.py. Every revealed number in
every frame is read, its normalised glyph shape is taken, and near-duplicates
are dropped. Keep the frames used for testing (evaluate_digits.py) out of the
training directories.
"""
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from minesweeper_solver.vision import recognize  # noqa: E402
from minesweeper_solver.vision.cells import classify_cells  # noqa: E402
from minesweeper_solver.vision.glyphs import DUPLICATE, SAMPLE_FILE, _prepare  # noqa: E402

MAX_PER_DIGIT = 40


def main(dirs) -> None:
    kept: dict[int, list] = collections.defaultdict(list)
    vectors: dict[int, list] = collections.defaultdict(list)
    seen = collections.Counter()
    for d in dirs:
        for png in sorted(glob.glob(os.path.join(d, "*.png"))):
            truth = json.load(open(png[:-4] + ".json"))["truth"]
            frame = cv2.imread(png)[:, :, ::-1].copy()
            result = recognize(frame)
            if result.grid is None or (result.grid.rows, result.grid.cols) != (len(truth), len(truth[0])):
                continue
            masks = classify_cells(frame, result.grid).features["masks"]
            for r, row in enumerate(truth):
                for c, digit in enumerate(row):
                    if digit is None or not 1 <= digit <= 8 or not masks[r, c].any():
                        continue
                    seen[digit] += 1
                    if len(kept[digit]) >= MAX_PER_DIGIT:
                        continue
                    vec = _prepare(masks[r, c][None])[0]
                    if vectors[digit] and float((np.array(vectors[digit]) @ vec).max()) >= DUPLICATE:
                        continue
                    kept[digit].append(masks[r, c].copy())
                    vectors[digit].append(vec)
    all_masks = [m for d in sorted(kept) for m in kept[d]]
    all_digits = [d for d in sorted(kept) for _ in kept[d]]
    np.savez_compressed(SAMPLE_FILE, masks=np.array(all_masks, np.float32),
                        digits=np.array(all_digits, np.int8))
    for d in range(1, 9):
        print(f"digit {d}: {seen[d]:6d} seen, {len(kept[d]):3d} distinct shapes kept")
    print(f"wrote {len(all_masks)} samples to {SAMPLE_FILE}")


if __name__ == "__main__":
    main(sys.argv[1:])
