"""Digit shape matching against templates.

Glyph masks are normalised to a fixed canvas and compared, by normalised
cross-correlation, with templates of the digits 1-8 from three sources:

* digit_templates.npz  digits rendered from a range of fonts (make_digit_templates.py)
* digit_samples.npz    digits collected from real games (train_digits.py)
* a user file          digits taught in the app by the user (`add_user_samples`)
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

CANVAS = 20
TEMPLATE_FILE = Path(__file__).with_name("digit_templates.npz")
SAMPLE_FILE = Path(__file__).with_name("digit_samples.npz")
DUPLICATE = 0.985          # a new sample this close to an existing one adds nothing


MIN_ASPECT = 0.8   # narrower glyphs (the digit 1) are padded, not stretched, to this


def normalize_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    """Crop `mask` (2-D bool/float) to its bounding box and stretch it to fill
    a CANVAS x CANVAS float image. None if the mask is empty.

    Filling the canvas removes the difference between narrow and wide fonts
    (a squat pixel-art 2 and a tall Arial 2 end up alike). Only very narrow
    glyphs keep some of their slimness, so that a 1 does not become a block.
    """
    ys, xs = np.nonzero(mask > 0)
    if len(ys) == 0:
        return None
    crop = mask[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1].astype(np.float32)
    h, w = crop.shape
    if w < MIN_ASPECT * h:
        pad = int(round(MIN_ASPECT * h)) - w
        crop = np.pad(crop, ((0, 0), (pad // 2, pad - pad // 2)))
    shrink = crop.shape[0] > CANVAS or crop.shape[1] > CANVAS
    return cv2.resize(crop, (CANVAS, CANVAS),
                      interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)


def _prepare(masks: np.ndarray) -> np.ndarray:
    """Blur (tolerates stroke-width differences), then zero-mean / unit-norm
    so that a dot product is the normalised cross-correlation."""
    out = np.empty((len(masks), CANVAS * CANVAS), np.float32)
    for i, mask in enumerate(masks):
        blurred = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.1).ravel()
        blurred -= blurred.mean()
        out[i] = blurred / max(float(np.linalg.norm(blurred)), 1e-6)
    return out


_lock = threading.Lock()
_cache: Optional[tuple[np.ndarray, np.ndarray]] = None
_user_file: Optional[Path] = None


def _read(path: Optional[Path]) -> tuple[np.ndarray, np.ndarray]:
    if path is None or not Path(path).exists():
        return np.zeros((0, CANVAS, CANVAS), np.float32), np.zeros(0, np.int8)
    data = np.load(path)
    return data["masks"].astype(np.float32), data["digits"].astype(np.int8)


def load_templates() -> tuple[np.ndarray, np.ndarray]:
    """(prepared templates [N, CANVAS*CANVAS], digits [N]) from all sources."""
    global _cache
    cached = _cache
    if cached is not None:
        return cached
    with _lock:
        if _cache is None:
            parts = [_read(TEMPLATE_FILE), _read(SAMPLE_FILE), _read(_user_file)]
            masks = np.concatenate([m for m, _ in parts])
            digits = np.concatenate([d for _, d in parts])
            _cache = (_prepare(masks), digits)
        return _cache


def set_user_samples_file(path: Optional[Path]) -> None:
    """Where digits taught in the app are kept (None: none)."""
    global _user_file, _cache
    with _lock:
        _user_file = Path(path) if path is not None else None
        _cache = None


def user_sample_count() -> int:
    return len(_read(_user_file)[1])


def add_user_samples(masks: np.ndarray, digits: np.ndarray) -> int:
    """Remember digits the user identified. Returns how many were new."""
    global _cache
    if _user_file is None:
        raise RuntimeError("no user sample file set")
    masks = np.asarray(masks, np.float32).reshape(-1, CANVAS, CANVAS)
    digits = np.asarray(digits, np.int8).reshape(-1)
    with _lock:
        old_masks, old_digits = _read(_user_file)
        keep_masks, keep_digits = list(old_masks), list(old_digits)
        known = _prepare(old_masks) if len(old_masks) else np.zeros((0, CANVAS * CANVAS), np.float32)
        added = 0
        for mask, digit in zip(masks, digits):
            vec = _prepare(mask[None])[0]
            same = known[np.array(keep_digits) == digit] if len(known) else known
            if len(same) and float((same @ vec).max()) >= DUPLICATE:
                continue
            keep_masks.append(mask)
            keep_digits.append(digit)
            known = np.vstack([known, vec[None]])
            added += 1
        if added:
            _user_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(_user_file, masks=np.array(keep_masks, np.float32),
                                digits=np.array(keep_digits, np.int8))
            _cache = None
        return added


def clear_user_samples() -> None:
    global _cache
    with _lock:
        if _user_file is not None and _user_file.exists():
            _user_file.unlink()
        _cache = None


def digit_scores(glyphs: np.ndarray) -> np.ndarray:
    """Normalised cross-correlation of each glyph [G, CANVAS, CANVAS] with the
    best template of every digit. Returns [G, 9]; column d is digit d
    (column 0 unused). 1.0 is a perfect match."""
    templates, digits = load_templates()
    corr = _prepare(glyphs) @ templates.T                   # [G, N]
    out = np.zeros((len(glyphs), 9), np.float32)
    for d in range(1, 9):
        out[:, d] = corr[:, digits == d].max(axis=1)
    return out


def count_holes(mask: np.ndarray) -> int:
    """Number of sizeable enclosed background regions (0 for 1/2/3/5/7, 1 for
    4/6, 2 for 8). Tiny enclosures are ignored: at small sizes neighbouring
    strokes of a 2 or 3 can touch and pinch off a few pixels."""
    padded = np.pad(mask.astype(np.uint8), 1)
    count, _, stats, _ = cv2.connectedComponentsWithStats(1 - padded, connectivity=4)
    minimum = max(4, 0.045 * mask.size)
    # label 0 is the foreground, label 1 the outer background (it contains the padding)
    return int(sum(stats[label, cv2.CC_STAT_AREA] >= minimum for label in range(2, count)))
