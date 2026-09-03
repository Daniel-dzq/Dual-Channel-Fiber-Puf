"""Valid-mask construction matching Experiment 1 rules."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def build_valid_mask_from_refs(
    refs: list[np.ndarray],
    *,
    abs_floor: float = 5.0,
    percentile: float = 10.0,
) -> tuple[np.ndarray, float, float]:
    """Build mask from reference images (Experiment 1 threshold rule).

    Returns (mask, coverage_fraction, threshold).
    """
    if not refs:
        raise ValueError("Need at least one reference image to build valid_mask")
    stack = np.stack([np.asarray(r, dtype=np.float64) for r in refs], axis=0)
    ref = np.mean(stack, axis=0)
    thr = max(float(abs_floor), float(np.percentile(ref, percentile)))
    mask = ref > thr
    frac = float(mask.mean())
    if int(np.count_nonzero(mask)) < 64:
        raise RuntimeError(f"valid_mask too small ({int(np.count_nonzero(mask))} pixels)")
    return mask.astype(bool), frac, thr


def load_mask_png(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    return arr > 0


def save_mask_png(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def mask_hash(mask: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(mask.astype(np.uint8)).tobytes()).hexdigest()


def combine_pair_mask(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    """Experiment 1 used one global mask; for two device masks use intersection."""
    if mask_a.shape != mask_b.shape:
        raise ValueError(f"mask shape mismatch: {mask_a.shape} vs {mask_b.shape}")
    return mask_a.astype(bool) & mask_b.astype(bool)
