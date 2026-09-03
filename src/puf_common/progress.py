"""Terminal progress helpers.

Uses in-place \\r updates only on a real interactive TTY.
When stdout is piped (e.g. ``... | tee log``), prints one clean newline
per completed item so the log and Cursor terminal stay readable.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Iterator, TextIO


def open_progress_stream() -> TextIO:
    """Prefer the real terminal when available."""
    try:
        stream = open("/dev/tty", "w", encoding="utf-8", buffering=1)
        if stream.isatty():
            return stream
        stream.close()
    except OSError:
        pass
    return sys.stderr


@contextmanager
def mute_console_logging() -> Iterator[None]:
    """Keep FileHandlers; silence StreamHandlers so logs do not break the bar."""
    root = logging.getLogger()
    muted: list[tuple[logging.Handler, int]] = []
    for handler in root.handlers:
        stream = getattr(handler, "stream", None)
        if stream is sys.stdout or stream is sys.stderr:
            muted.append((handler, handler.level))
            handler.setLevel(logging.CRITICAL + 1)
    try:
        yield
    finally:
        for handler, level in muted:
            handler.setLevel(level)


class LineProgress:
    """Progress display that stays readable under ``tee`` and in Cursor."""

    def __init__(
        self,
        total: int,
        desc: str,
        *,
        status_interval_s: float = 2.0,
    ) -> None:
        self.total = max(int(total), 1)
        self.desc = desc
        self.n = 0
        self._file = open_progress_stream()
        self._owns_file = self._file not in (sys.stdout, sys.stderr)
        # Piped stdout (tee) → newline mode. Interactive TTY → in-place bar.
        self._inplace = bool(sys.stdout.isatty() and self._file.isatty())
        self._status_interval_s = status_interval_s
        self._last_status_t = 0.0
        self._last_status = ""
        self._closed = False
        self._emit(force=True)

    def status(self, status: str = "") -> None:
        """Update current-item detail without advancing the counter."""
        now = time.monotonic()
        # In newline/tee mode, avoid spamming; only refresh periodically.
        min_interval = 0.15 if self._inplace else self._status_interval_s
        if (
            status == self._last_status
            or (now - self._last_status_t) < min_interval
        ):
            return
        self._last_status_t = now
        self._emit(status=status, force=False)

    def update(self, status: str = "") -> None:
        """Advance counter by one (one finished item)."""
        self.n = min(self.n + 1, self.total)
        self._last_status_t = time.monotonic()
        self._emit(status=status, force=True)

    def _format(self, status: str) -> str:
        frac = self.n / self.total
        pct = int(round(100 * frac))
        if self._inplace:
            bar_w = 22
            filled = int(round(bar_w * frac))
            bar = "#" * filled + "-" * (bar_w - filled)
            return f"{self.desc}: {pct:3d}%|{bar}| {self.n}/{self.total} {status}".rstrip()
        return f"{self.desc}: {self.n}/{self.total} ({pct:3d}%) {status}".rstrip()

    def _emit(self, status: str = "", *, force: bool) -> None:
        self._last_status = status
        text = self._format(status)
        if self._inplace:
            # Clear to end of line so shorter statuses do not leave junk.
            self._file.write("\r\033[K" + text)
            self._file.flush()
        else:
            # Newline mode for tee / Cursor: only print on force (item done)
            # or throttled status refreshes.
            if force or status:
                print(text, flush=True)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._inplace:
            self._file.write("\n")
            self._file.flush()
        if self._owns_file:
            self._file.close()

    def __enter__(self) -> "LineProgress":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
