"""Naming helpers and path layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def mp_tag(macro_pixel_size: int) -> str:
    return f"mp{int(macro_pixel_size):03d}"


def challenge_grid_size(active_size: int, macro_pixel_size: int) -> int:
    if macro_pixel_size <= 0:
        raise ValueError("macro_pixel_size must be positive")
    if active_size % macro_pixel_size != 0:
        raise ValueError(
            f"active_size {active_size} not divisible by macro_pixel_size {macro_pixel_size}"
        )
    return active_size // macro_pixel_size


@dataclass(frozen=True)
class PathLayout:
    root: Path

    @property
    def patterns(self) -> Path:
        return self.root / "patterns"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    @property
    def videos(self) -> Path:
        return self.root / "videos"

    @property
    def metadata(self) -> Path:
        return self.root / "metadata"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def calibration(self) -> Path:
        return self.root / "calibration"

    @property
    def patterns_calibration(self) -> Path:
        return self.root / "patterns" / "calibration"

    @property
    def analysis(self) -> Path:
        return self.root / "analysis"

    @property
    def screening(self) -> Path:
        return self.root / "analysis" / "screening"
