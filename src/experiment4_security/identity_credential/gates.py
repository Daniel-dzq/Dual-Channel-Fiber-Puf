"""Safety gates: prepare-only and incomplete-data must not emit formal science."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from experiment4_security.identity_credential.schemas import (
    FORBIDDEN_PREPARE_METRIC_TOKENS,
    MODE_FORMAL,
    STATUS_DATA_READY,
)


class GateError(RuntimeError):
    """Raised when a forbidden formal artifact or metric would be emitted."""


_PROTECTED = (
    "src/experiment4_security/lifecycle",
    "outputs/experiment4/lifecycle",
    "outputs/experiment4/security/runs/20260728_213919_f01_green_pilot",
)

_REMOVED_LEGACY = (
    "outputs/experiment4/ml_attack",
    "outputs/experiment4/identity_credential",
)


def assert_not_writing_protected(path: Path, cfg_project_root: Path) -> None:
    path = Path(path).resolve()
    root = Path(cfg_project_root).resolve()
    for suf in _PROTECTED:
        guard = (root / suf).resolve()
        try:
            path.relative_to(guard)
            raise GateError(f"Refusing write under protected path: {guard}")
        except ValueError:
            continue
    for suf in _REMOVED_LEGACY:
        guard = (root / suf).resolve()
        try:
            path.relative_to(guard)
            raise GateError(
                f"Legacy output path removed; use outputs/experiment4/security/runs/: {guard}"
            )
        except ValueError:
            continue


def scan_forbidden_tokens(text: str) -> list[str]:
    hits: list[str] = []
    low = text.lower()
    for tok in FORBIDDEN_PREPARE_METRIC_TOKENS:
        if tok.lower() in low:
            hits.append(tok)
    if re.search(r"\bauc[_ ]?red\b\s*[:=]\s*[0-9]", low):
        hits.append("numeric_auc_red")
    if re.search(r"\beer[_ ]?red\b\s*[:=]\s*[0-9]", low):
        hits.append("numeric_eer_red")
    return hits


def assert_prepare_text_safe(text: str, *, context: str = "") -> None:
    hits = scan_forbidden_tokens(text)
    if hits:
        raise GateError(f"Forbidden formal tokens in prepare output ({context}): {hits}")


def require_data_ready_for_formal(data_status: str, mode: str) -> None:
    if mode == MODE_FORMAL and data_status != STATUS_DATA_READY:
        raise GateError(
            f"Formal full run refused: data_status={data_status!r} (need {STATUS_DATA_READY!r}). "
            "Place complete F01–F10 synchronized data, then run validate."
        )


def formal_metrics_allowed(mode: str, data_status: str) -> bool:
    return mode == MODE_FORMAL and data_status == STATUS_DATA_READY


def refuse_fake_red_arrays() -> GateError:
    return GateError(
        "Refusing random/synthetic arrays as formal red results. "
        "Synthetic fixtures are tests-only; formal run requires real synchronized red videos."
    )


def check_paths_writable(paths: Iterable[Path], project_root: Path) -> None:
    for p in paths:
        assert_not_writing_protected(p, project_root)
