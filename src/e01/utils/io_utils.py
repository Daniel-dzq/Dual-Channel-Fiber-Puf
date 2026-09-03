"""Filesystem helpers."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def assert_writable(path: Path, force: bool = False) -> None:
    if path.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing path: {path}. Pass --force to overwrite."
        )
