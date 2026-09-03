"""Red persistent identity: frozen 9-D Fiber-ID + standardized Euclidean."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiment4_security.common.frozen_protocol import RED_LOWDIM_NAMES
from experiment4_security.common.video_io import ProcessedRecording
from puf_common.fiber_id import extract_fiber_id_vector


def extract_red_vector(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Frozen 9-D Fiber-ID vector. No envelope, no common-component subtraction."""
    vec = extract_fiber_id_vector(img, mask)
    if vec.size != len(RED_LOWDIM_NAMES):
        raise RuntimeError(
            f"Fiber-ID vector length {vec.size} != {len(RED_LOWDIM_NAMES)}"
        )
    return vec


def extract_red_pack(rec: ProcessedRecording, mask: np.ndarray) -> dict[str, Any]:
    rep = extract_red_vector(rec.representative, mask)
    return {
        "device_id": rec.device_id,
        "state_id": rec.state_id,
        "representative_vector": rep,
    }


def fit_standardizer(vectors: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stack = np.stack(vectors, axis=0)
    mu = np.mean(stack, axis=0)
    sd = np.std(stack, axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return mu, sd


def apply_standardizer(vec: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (vec - mu) / sd


def red_score(a_std: np.ndarray, b_std: np.ndarray) -> float:
    """S_R = -standardized Euclidean distance."""
    d = a_std - b_std
    return -float(np.sqrt(np.sum(d * d)))
