"""Shared tqdm progress helpers (Experiments 1–3).

Keeps tqdm bars, but avoids log flooding when stdout/stderr are piped (e.g. tee):
mid-item postfix refreshes are disabled unless stderr is an interactive TTY.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from tqdm.auto import tqdm

from puf_common.progress import mute_console_logging

T = TypeVar("T")

__all__ = [
    "stage_tqdm",
    "make_video_progress",
    "mute_console_logging",
    "is_interactive_stderr",
]


def is_interactive_stderr() -> bool:
    """True only when in-place \\r updates are safe."""
    try:
        return bool(sys.stderr.isatty())
    except Exception:
        return False


def stage_tqdm(
    iterable: Iterable[T],
    *,
    desc: str,
    unit: str = "it",
    total: int | None = None,
    **kwargs: Any,
) -> tqdm:
    """Standard stage bar: tqdm on stderr, tee-safe refresh policy."""
    interactive = is_interactive_stderr()
    opts: dict[str, Any] = {
        "desc": desc,
        "unit": unit,
        "total": total,
        "file": sys.stderr,
        "dynamic_ncols": interactive,
        "mininterval": 0.3 if interactive else 60.0,
        "maxinterval": 10.0 if interactive else 120.0,
        "leave": True,
        "smoothing": 0.05,
    }
    opts.update(kwargs)
    return tqdm(iterable, **opts)


def make_video_progress(
    pbar: tqdm,
    label: str,
) -> Callable[[str], None] | None:
    """Per-video decode/correct callback for tqdm postfix.

    Interactive TTY: throttled postfix updates.
    Piped/tee: set label once without refresh; return None so mid-video
    callbacks do not spam the log.
    """
    if not is_interactive_stderr():
        pbar.set_postfix_str(label, refresh=False)
        return None

    last_t = 0.0
    min_dt = 0.4

    def _on_progress(msg: str) -> None:
        nonlocal last_t
        now = time.monotonic()
        if (now - last_t) < min_dt and not str(msg).startswith("aggregate"):
            return
        last_t = now
        pbar.set_postfix_str(f"{label} | {msg}", refresh=True)

    pbar.set_postfix_str(label, refresh=False)
    return _on_progress
