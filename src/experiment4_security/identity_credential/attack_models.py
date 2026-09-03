"""Attack model registry for Track E (challenge-only / red-only / challenge+red).

Formal fitting happens only under DATA_READY; prepare mode exposes names only.
"""

from __future__ import annotations

from typing import Any

from experiment4_security.identity_credential.gates import GateError, formal_metrics_allowed


MODEL_SPECS: tuple[dict[str, str], ...] = (
    {"name": "challenge_only_ridge", "inputs": "challenge_features", "target": "green_detail_cm"},
    {"name": "red_only_ridge", "inputs": "red_13d", "target": "green_detail_cm"},
    {"name": "challenge_plus_red_ridge", "inputs": "challenge_features+red_13d", "target": "green_detail_cm"},
)


def list_attack_models() -> list[dict[str, str]]:
    return [dict(m) for m in MODEL_SPECS]


def fit_attack_models_formal(*, mode: str, data_status: str) -> dict[str, Any]:
    if not formal_metrics_allowed(mode, data_status):
        raise GateError("Attack model fitting refused outside formal DATA_READY run")
    raise GateError("Attack model fitting deferred until synchronized dataset is present")
