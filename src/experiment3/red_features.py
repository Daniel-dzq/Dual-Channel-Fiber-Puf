"""Primary red identity features: low-dimensional spatial/spectral statistics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from experiment3.config import Experiment3Config
from experiment3.video_processing import ProcessedRecording
from puf_common.fiber_id import FIBER_ID_FEATURE_NAMES, extract_fiber_id_features
from puf_common.features import mean_template

LOWDIM_NAMES = list(FIBER_ID_FEATURE_NAMES)


def extract_lowdim_vector(
    img: np.ndarray,
    mask: np.ndarray,
    cfg: Experiment3Config,
) -> tuple[np.ndarray, dict[str, float]]:
    """Extract the frozen 9-D Fiber-ID vector (no common-component subtraction).

    ``cfg`` is accepted for call-site compatibility and is not used to select
    PSD bin count or to restore retired intensity features.
    """
    _ = cfg
    return extract_fiber_id_features(img, mask)


def extract_red_features_for_recording(
    rec: ProcessedRecording,
    mask: np.ndarray,
    cfg: Experiment3Config,
) -> dict[str, Any]:
    """Red uses intensity (or detail) templates without challenge common subtraction."""
    # Primary identity uses the corrected intensity representative (frozen remount practice).
    rep_vec, scalars = extract_lowdim_vector(rec.representative, mask, cfg)
    block_vecs = []
    for bi, block in enumerate(rec.blocks):
        bvec, _ = extract_lowdim_vector(block, mask, cfg)
        block_vecs.append(bvec)
    return {
        "device_id": rec.device_id,
        "state_id": rec.state_id,
        "channel": "red",
        "representative_vector": rep_vec,
        "block_vectors": block_vecs,
        "scalars": scalars,
        "n_blocks": len(block_vecs),
    }


def fit_standardizer(
    vectors: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    stack = np.stack(vectors, axis=0)
    mu = np.mean(stack, axis=0)
    sd = np.std(stack, axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return mu, sd


def apply_standardizer(vec: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (vec - mu) / sd


def standardized_euclidean(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.sum(d * d)))


def red_similarity_score(a_std: np.ndarray, b_std: np.ndarray) -> float:
    """Official red score: S_R = - standardized Euclidean distance."""
    return -standardized_euclidean(a_std, b_std)


def features_to_dataframe(packs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for p in packs:
        row = {
            "device_id": p["device_id"],
            "state_id": p["state_id"],
            "channel": "red",
            "n_blocks": p["n_blocks"],
        }
        for i, name in enumerate(LOWDIM_NAMES):
            row[f"lowdim_{name}"] = float(p["representative_vector"][i])
        for k, v in p["scalars"].items():
            row[f"scalar_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def mean_block_vector(pack: dict[str, Any]) -> np.ndarray:
    if not pack["block_vectors"]:
        return pack["representative_vector"]
    return mean_template(pack["block_vectors"])
