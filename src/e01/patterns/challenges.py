"""Shared binary-challenge helpers."""

from __future__ import annotations

import numpy as np


def normalized_hamming_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.mean(a.astype(np.uint8) != b.astype(np.uint8)))


def duty_cycle(binary: np.ndarray) -> float:
    return float(np.mean(binary.astype(np.float64)))


def generate_balanced_binary(rng: np.random.Generator, grid: int) -> np.ndarray:
    n = grid * grid
    if n % 2 != 0:
        raise ValueError(f"grid size {grid} yields odd number of elements")
    flat = np.zeros(n, dtype=np.uint8)
    flat[: n // 2] = 1
    rng.shuffle(flat)
    return flat.reshape(grid, grid)
