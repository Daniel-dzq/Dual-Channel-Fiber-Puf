"""Partial-leakage attack model zoo (PL0-PL5).

PL0/PL1 are challenge-side / template-lookup baselines needing no PCA.
PL2-PL5 reuse the frozen Track-C regressors in ``models.py`` unchanged,
fit only on the leaked subset passed in by the caller.

Hard leakage guard (enforced by construction):
every fit_* function below only ever consumes a ``leaked_templates`` dict
supplied by the caller. It never receives hidden-challenge templates, Round
B data, or target-state data. ``models.py``'s regressors additionally
intersect ``train_features.keys() & train_templates.keys()``, so even a
full 128-challenge feature dict cannot leak into the fit set as long as
``train_templates`` only contains leaked ids (guaranteed by callers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from experiment4_security.ml_attack.baselines import fit_mean_baseline
from experiment4_security.ml_attack.config import ModelsConfig
from experiment4_security.ml_attack.models import (
    fit_enrollment_representation,
    fit_kernel_ridge_fixed,
    fit_mlp_fixed,
    fit_rff_ridge_fixed,
    fit_ridge_fixed,
)
from experiment4_security.ml_attack.representations import RepresentationBundle

PL_ATTACK_METHOD_MAP = {
    "PL0_mean_leaked_response": "mean_leaked_response",
    "PL1_nearest_leaked_challenge": "nearest_leaked_challenge",
    "PL2_ridge_clone": "ridge_clone",
    "PL3_kernel_ridge_clone": "kernel_ridge_clone",
    "PL4_random_fourier_ridge_clone": "random_fourier_ridge_clone",
    "PL5_small_mlp_clone": "small_mlp_clone",
}


def effective_pca_dimension(n_leaked: int, pca_dimension_max: int) -> int:
    """min(pca_dimension_max, n_leaked - 1); numerical rank is enforced by
    sklearn's PCA(n_components=...) itself never exceeding min(n_samples,
    n_features), which is already <= n_leaked here."""
    return max(1, min(int(pca_dimension_max), int(n_leaked) - 1))


@dataclass
class PL0MeanPredictor:
    mean_vector: np.ndarray
    model_name: str = "PL0_mean_leaked_response"

    def predict_batch(self, hidden_ids: list[str]) -> dict[str, np.ndarray]:
        return {cid: self.mean_vector for cid in hidden_ids}


def fit_pl0_mean(leaked_templates: dict[str, np.ndarray]) -> PL0MeanPredictor:
    base = fit_mean_baseline(leaked_templates)
    return PL0MeanPredictor(mean_vector=base.mean_vector)


def load_hamming_distance_matrix(hamming_csv: str, challenge_ids: list[str]) -> pd.DataFrame:
    """Symmetric normalized-Hamming distance table indexed by challenge_id.

    The CSV stores each unordered pair once (challenge_id_a < challenge_id_b);
    this returns a full symmetric ``n x n`` DataFrame with zero diagonal so
    nearest-neighbor lookups need no special-casing.
    """
    raw = pd.read_csv(hamming_csv)
    mat = pd.DataFrame(np.nan, index=challenge_ids, columns=challenge_ids, dtype=np.float64)
    for cid in challenge_ids:
        mat.loc[cid, cid] = np.inf  # never self-match
    for row in raw.itertuples(index=False):
        a, b, d = str(row.challenge_id_a), str(row.challenge_id_b), float(row.normalized_hamming)
        if a in mat.index and b in mat.columns:
            mat.loc[a, b] = d
            mat.loc[b, a] = d
    return mat


@dataclass
class PL1NearestPredictor:
    leaked_templates: dict[str, np.ndarray]
    distance_matrix: pd.DataFrame  # full challenge_id x challenge_id, restricted lookups to leaked columns
    model_name: str = "PL1_nearest_leaked_challenge"

    def predict_batch(self, hidden_ids: list[str]) -> tuple[dict[str, np.ndarray], dict[str, str]]:
        leaked_ids = list(self.leaked_templates.keys())
        preds: dict[str, np.ndarray] = {}
        nearest: dict[str, str] = {}
        sub = self.distance_matrix.loc[hidden_ids, leaked_ids]
        for h in hidden_ids:
            nearest_id = sub.loc[h].idxmin()
            nearest[h] = str(nearest_id)
            preds[h] = self.leaked_templates[nearest_id]
        return preds, nearest


def fit_pl1_nearest(
    leaked_templates: dict[str, np.ndarray], distance_matrix: pd.DataFrame
) -> PL1NearestPredictor:
    leaked_ids = list(leaked_templates.keys())
    return PL1NearestPredictor(
        leaked_templates=leaked_templates,
        distance_matrix=distance_matrix.loc[:, leaked_ids],
    )


@dataclass
class PLRegressionPredictor:
    inner: Any  # RegressionPredictor or MLPPredictor from models.py
    model_name: str
    bundle: RepresentationBundle
    effective_pca_dim: int

    def predict_batch(self, features_by_cid: dict[str, np.ndarray], hidden_ids: list[str]) -> dict[str, np.ndarray]:
        return self.inner.predict_batch(features_by_cid, hidden_ids)


def fit_pl_regressors(
    leaked_features: dict[str, np.ndarray],
    leaked_templates: dict[str, np.ndarray],
    cfg: ModelsConfig,
    *,
    pca_dimension_max: int,
    which: tuple[str, ...] = (
        "PL2_ridge_clone",
        "PL3_kernel_ridge_clone",
        "PL4_random_fourier_ridge_clone",
        "PL5_small_mlp_clone",
    ),
) -> dict[str, PLRegressionPredictor]:
    n_leaked = len(leaked_templates)
    pca_dim = effective_pca_dimension(n_leaked, pca_dimension_max)
    bundle = fit_enrollment_representation(leaked_templates, pca_dimension=pca_dim)
    effective_dim = int(bundle.pca_dimension)

    out: dict[str, PLRegressionPredictor] = {}
    if "PL2_ridge_clone" in which:
        inner = fit_ridge_fixed(leaked_features, leaked_templates, bundle, cfg)
        out["PL2_ridge_clone"] = PLRegressionPredictor(inner, "PL2_ridge_clone", bundle, effective_dim)
    if "PL3_kernel_ridge_clone" in which:
        inner = fit_kernel_ridge_fixed(leaked_features, leaked_templates, bundle, cfg)
        out["PL3_kernel_ridge_clone"] = PLRegressionPredictor(inner, "PL3_kernel_ridge_clone", bundle, effective_dim)
    if "PL4_random_fourier_ridge_clone" in which:
        inner = fit_rff_ridge_fixed(leaked_features, leaked_templates, bundle, cfg)
        out["PL4_random_fourier_ridge_clone"] = PLRegressionPredictor(
            inner, "PL4_random_fourier_ridge_clone", bundle, effective_dim
        )
    if "PL5_small_mlp_clone" in which:
        inner = fit_mlp_fixed(leaked_features, leaked_templates, bundle, cfg)
        out["PL5_small_mlp_clone"] = PLRegressionPredictor(inner, "PL5_small_mlp_clone", bundle, effective_dim)
    return out
