"""Experiment 3 figure generation (disabled).

Automatic plotting was removed from the analysis pipeline.
Add figure routines here when the target panels are specified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def generate_all_figures(*_args: Any, **_kwargs: Any) -> list[Path]:
    """No-op placeholder; figures are not produced by default."""
    return []
