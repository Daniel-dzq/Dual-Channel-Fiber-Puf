"""Zero-mean normalized cross-correlation."""

from __future__ import annotations

import numpy as np


def zero_mean_ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Zero-mean NCC; optional boolean mask selects pixels."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if mask is None:
        x = a.astype(np.float64).ravel()
        y = b.astype(np.float64).ravel()
    else:
        if mask.shape != a.shape:
            raise ValueError(f"mask shape {mask.shape} != image shape {a.shape}")
        sel = mask.astype(bool)
        if int(np.count_nonzero(sel)) < 2:
            return 0.0
        x = a.astype(np.float64)[sel]
        y = b.astype(np.float64)[sel]
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom == 0.0:
        return 0.0
    return float(np.dot(x, y) / denom)
