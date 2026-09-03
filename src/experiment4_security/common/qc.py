"""QC helpers: objective acquisition faults only."""

from __future__ import annotations

from typing import Any


def catastrophic_flags(
    *,
    saturation_fraction: float,
    mean_intensity: float,
    n_frames: int,
    max_sat: float = 0.05,
    min_mean: float = 1.0,
    min_frames: int = 30,
) -> list[str]:
    flags: list[str] = []
    if saturation_fraction > max_sat:
        flags.append("severe_saturation")
    if mean_intensity < min_mean:
        flags.append("nearly_absent_signal")
    if n_frames < min_frames:
        flags.append("insufficient_frames")
    return flags


def should_exclude(flags: list[str]) -> bool:
    """Exclude only objective catastrophic failures, never poor auth scores."""
    return any(
        f in flags
        for f in ("severe_saturation", "nearly_absent_signal", "insufficient_frames", "unreadable")
    )
