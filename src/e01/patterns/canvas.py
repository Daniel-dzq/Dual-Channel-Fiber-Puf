"""SLM canvas helpers."""

from __future__ import annotations

import numpy as np

from e01.config import SlmConfig


def blank_canvas(slm: SlmConfig, fill: int | None = None) -> np.ndarray:
    value = slm.gray_off if fill is None else int(fill)
    return np.full((slm.canvas_height, slm.canvas_width), value, dtype=np.uint8)


def place_active_region(slm: SlmConfig, active: np.ndarray, fill: int | None = None) -> np.ndarray:
    if active.ndim != 2:
        raise ValueError(f"active region must be 2D, got shape {active.shape}")
    expected = (slm.active_height, slm.active_width)
    if active.shape != expected:
        raise ValueError(f"active region shape {active.shape} != expected {expected}")
    canvas = blank_canvas(slm, fill=fill)
    canvas[slm.y_start : slm.y_end, slm.x_start : slm.x_end] = active.astype(np.uint8, copy=False)
    return canvas


def expand_binary_nearest(binary: np.ndarray, macro_pixel_size: int) -> np.ndarray:
    if binary.ndim != 2:
        raise ValueError("binary matrix must be 2D")
    if macro_pixel_size <= 0:
        raise ValueError("macro_pixel_size must be positive")
    if not np.isin(binary, [0, 1]).all():
        raise ValueError("binary matrix must contain only 0 and 1")
    return np.repeat(
        np.repeat(binary.astype(np.uint8), macro_pixel_size, axis=0),
        macro_pixel_size,
        axis=1,
    )


def map_binary_to_gray(binary_or_expanded: np.ndarray, gray_off: int, gray_on: int) -> np.ndarray:
    out = np.full(binary_or_expanded.shape, int(gray_off), dtype=np.uint8)
    out[binary_or_expanded.astype(bool)] = int(gray_on)
    return out
