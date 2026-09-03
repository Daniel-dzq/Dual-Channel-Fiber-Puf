"""Threshold freeze helpers for formal multi-device run (gated)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from experiment4_security.identity_credential.gates import GateError, formal_metrics_allowed
from experiment4_security.lifecycle.thresholds import select_tau


def freeze_tau_from_scores(
    genuine: np.ndarray,
    impostor: np.ndarray,
    *,
    mode: str,
    data_status: str,
    label: str,
) -> dict[str, Any]:
    if not formal_metrics_allowed(mode, data_status):
        raise GateError(f"Refusing to freeze {label}: not in formal DATA_READY mode")
    tau = select_tau(np.asarray(genuine), np.asarray(impostor))
    return {"label": label, "tau": float(tau)}


def write_thresholds(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
