"""Bootstrap helpers (device-unit when multi-device formal run is allowed)."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiment4_security.identity_credential.gates import formal_metrics_allowed


def device_bootstrap_indices(
    n_devices: int,
    n_resamples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_devices, size=(n_resamples, n_devices))


def bootstrap_status(mode: str, data_status: str) -> dict[str, Any]:
    if not formal_metrics_allowed(mode, data_status):
        return {
            "status": "WITHHELD",
            "note": "Device-unit bootstrap reserved for formal DATA_READY multi-device run",
        }
    return {"status": "READY"}
