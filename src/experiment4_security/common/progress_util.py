"""Progress helpers: one bar per stage, no spam under tee."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

from tqdm.auto import tqdm

from puf_common.progress import LineProgress, mute_console_logging

T = TypeVar("T")


def stage_tqdm(
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
    unit: str = "it",
) -> Iterator[T]:
    """Single tqdm bar with throttled refresh (avoids log flooding)."""
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        dynamic_ncols=True,
        mininterval=0.5,
        maxinterval=5.0,
        leave=True,
    )


def line_progress(total: int, desc: str) -> LineProgress:
    return LineProgress(total=total, desc=desc, status_interval_s=2.0)


__all__ = ["stage_tqdm", "line_progress", "mute_console_logging", "LineProgress"]
