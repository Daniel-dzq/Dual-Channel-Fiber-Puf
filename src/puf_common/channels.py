"""Color-channel helpers matching Experiment 1 OpenCV BGR convention."""

from __future__ import annotations

import numpy as np


def apply_roi(image: np.ndarray, roi: list[int] | tuple[int, ...] | None) -> np.ndarray:
    if roi is None:
        return image
    x, y, w, h = [int(v) for v in roi]
    return image[y : y + h, x : x + w]


def split_green(frame: np.ndarray) -> np.ndarray:
    """Extract green plane from OpenCV BGR. Reject grayscale."""
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError(
            f"Expected color BGR frame, got shape {getattr(frame, 'shape', None)}."
        )
    return frame[:, :, 1]


def split_red(frame: np.ndarray) -> np.ndarray:
    """Extract red plane from OpenCV BGR. Reject grayscale."""
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError(
            f"Expected color BGR frame, got shape {getattr(frame, 'shape', None)}."
        )
    return frame[:, :, 2]


def dark_subtract(raw: np.ndarray, dark: np.ndarray | float) -> np.ndarray:
    """Experiment 1 dark correction: max(raw - dark, 0) in float64."""
    out = raw.astype(np.float64) - np.asarray(dark, dtype=np.float64)
    return np.maximum(out, 0.0)


def saturation_fraction(image: np.ndarray, threshold: float = 250.0) -> float:
    arr = np.asarray(image)
    if arr.size == 0:
        return float("nan")
    return float(np.count_nonzero(arr >= threshold) / arr.size)
