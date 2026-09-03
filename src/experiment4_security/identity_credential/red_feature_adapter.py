"""Read-only façade over lifecycle 9-D Fiber ID feature extraction.

Do NOT reimplement RED_LOWDIM_NAMES or extract_red_vector.
Do NOT modify lifecycle source.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from experiment4_security.common.frozen_protocol import RED_LOWDIM_NAMES
from experiment4_security.lifecycle.red_identity import (
    apply_standardizer,
    extract_red_pack,
    extract_red_vector,
    fit_standardizer,
    red_score,
)

# Re-export canonical API
__all__ = [
    "RED_LOWDIM_NAMES",
    "extract_red_vector",
    "extract_red_pack",
    "fit_standardizer",
    "apply_standardizer",
    "red_score",
    "assert_feature_contract",
]


def assert_feature_contract() -> dict[str, Any]:
    if len(RED_LOWDIM_NAMES) != 9:
        raise AssertionError(f"RED_LOWDIM_NAMES must be length 9, got {len(RED_LOWDIM_NAMES)}")
    return {
        "dim": 9,
        "names": list(RED_LOWDIM_NAMES),
        "source": "formal.lifecycle.red_identity",
        "score": "S_R = -standardized_euclidean",
    }


def standardize_matrix(vectors: list[np.ndarray], mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return np.stack([apply_standardizer(v, mu, sd) for v in vectors], axis=0)
