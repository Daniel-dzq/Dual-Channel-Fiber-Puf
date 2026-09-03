"""Red identity enrollment (standardizer fit) — formal run only when DATA_READY."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiment4_security.identity_credential.gates import GateError, formal_metrics_allowed
from experiment4_security.identity_credential.red_feature_adapter import (
    apply_standardizer,
    fit_standardizer,
)


def fit_red_standardizer_on_development(
    vectors_by_key: dict[tuple[str, str, str], np.ndarray],
    development_devices: tuple[str, ...],
) -> dict[str, Any]:
    """Fit μ,σ on development-device red vectors (all states/phases present).

    vectors_by_key keys: (device_id, state_id, phase)
    """
    vecs = [v for (d, _s, _p), v in vectors_by_key.items() if d in development_devices]
    if not vecs:
        raise GateError("No development red vectors available for standardizer fit")
    mu, sd = fit_standardizer(vecs)
    return {
        "mu": mu,
        "sd": sd,
        "n_fit": len(vecs),
        "development_devices": list(development_devices),
        "feature_dim": int(mu.size),
    }


def apply_all(
    vectors_by_key: dict[tuple[str, str, str], np.ndarray],
    mu: np.ndarray,
    sd: np.ndarray,
) -> dict[tuple[str, str, str], np.ndarray]:
    return {k: apply_standardizer(v, mu, sd) for k, v in vectors_by_key.items()}


def enrollment_allowed(mode: str, data_status: str) -> bool:
    return formal_metrics_allowed(mode, data_status)
