"""Fixed-hyperparameter response-reconstruction regressors (Track C/D).

Round B is NEVER used for model selection, early stopping, PCA dimension,
or regularization. All hyperparameters come from YAML / ModelsConfig as
pre-fixed values.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from experiment4_security.ml_attack.config import ModelsConfig
from experiment4_security.ml_attack.representations import (
    RepresentationBundle,
    fit_representation,
    inverse_transform,
    to_pca_coefficients,
)

logger = logging.getLogger(__name__)


def _inverse_transform_batch(coeffs_batch: np.ndarray, bundle: RepresentationBundle) -> np.ndarray:
    xn = bundle.pca.inverse_transform(coeffs_batch)
    return xn * bundle.sigma.astype(np.float64) + bundle.mu.astype(np.float64)


@dataclass
class RegressionPredictor:
    regressor: Any
    bundle: RepresentationBundle
    model_name: str
    n_parameters: int
    hyperparameters: dict[str, Any]
    feature_transform: Any = None
    challenge_ids: list[str] | None = None

    def predict(self, challenge_binary_vector: np.ndarray) -> np.ndarray:
        x = challenge_binary_vector.reshape(1, -1).astype(np.float64)
        if self.feature_transform is not None:
            x = self.feature_transform.transform(x)
        coeffs = self.regressor.predict(x).ravel()
        return inverse_transform(coeffs, self.bundle)

    def predict_batch(self, features_by_cid: dict[str, np.ndarray], ids: list[str]) -> dict[str, np.ndarray]:
        X = np.stack([features_by_cid[c] for c in ids], axis=0).astype(np.float64)
        if self.feature_transform is not None:
            X = self.feature_transform.transform(X)
        coeffs = np.atleast_2d(self.regressor.predict(X))
        preds = _inverse_transform_batch(coeffs, self.bundle)
        return {cid: preds[i].astype(np.float32) for i, cid in enumerate(ids)}


@dataclass
class MLPPredictor:
    mlp: MLPRegressor
    bundle: RepresentationBundle
    model_name: str = "C5_small_mlp_clone"
    n_parameters: int = 0
    hyperparameters: dict[str, Any] | None = None

    def predict(self, challenge_binary_vector: np.ndarray) -> np.ndarray:
        coeffs = self.mlp.predict(challenge_binary_vector.reshape(1, -1).astype(np.float64)).ravel()
        return inverse_transform(coeffs, self.bundle)

    def predict_batch(self, features_by_cid: dict[str, np.ndarray], ids: list[str]) -> dict[str, np.ndarray]:
        X = np.stack([features_by_cid[c] for c in ids], axis=0).astype(np.float64)
        coeffs = np.atleast_2d(self.mlp.predict(X))
        preds = _inverse_transform_batch(coeffs, self.bundle)
        return {cid: preds[i].astype(np.float32) for i, cid in enumerate(ids)}


def mlp_param_count(mlp: MLPRegressor) -> int:
    return int(sum(c.size for c in mlp.coefs_) + sum(i.size for i in mlp.intercepts_))


def fit_enrollment_representation(
    enrollment_templates: dict[str, np.ndarray],
    *,
    pca_dimension: int,
) -> RepresentationBundle:
    """PCA + normalization fit on source Round-A enrollment templates only."""
    return fit_representation(
        enrollment_templates,
        space="fullres_detail",  # templates already in detail_cm
        pca_dimension=pca_dimension,
        forbidden_sample_ids=None,
    )


def _build_xy(
    challenge_features: dict[str, np.ndarray],
    templates: dict[str, np.ndarray],
    bundle: RepresentationBundle,
    ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack([challenge_features[cid] for cid in ids], axis=0).astype(np.float64)
    Y = np.stack([to_pca_coefficients(templates[cid], bundle) for cid in ids], axis=0)
    return X, Y


def fit_ridge_fixed(
    train_features: dict[str, np.ndarray],
    train_templates: dict[str, np.ndarray],
    bundle: RepresentationBundle,
    cfg: ModelsConfig,
) -> RegressionPredictor:
    ids = sorted(train_features.keys() & train_templates.keys())
    X, Y = _build_xy(train_features, train_templates, bundle, ids)
    alpha = float(cfg.ridge_alpha)
    reg = Ridge(alpha=alpha)
    reg.fit(X, Y)
    return RegressionPredictor(
        regressor=reg,
        bundle=bundle,
        model_name="C2_ridge_clone",
        n_parameters=int(reg.coef_.size + np.asarray(reg.intercept_).size),
        hyperparameters={"alpha": alpha},
        challenge_ids=ids,
    )


def fit_kernel_ridge_fixed(
    train_features: dict[str, np.ndarray],
    train_templates: dict[str, np.ndarray],
    bundle: RepresentationBundle,
    cfg: ModelsConfig,
) -> RegressionPredictor:
    ids = sorted(train_features.keys() & train_templates.keys())
    X, Y = _build_xy(train_features, train_templates, bundle, ids)
    alpha = float(cfg.kernel_ridge_alpha)
    gamma = float(cfg.kernel_ridge_gamma)
    reg = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
    reg.fit(X, Y)
    return RegressionPredictor(
        regressor=reg,
        bundle=bundle,
        model_name="C3_kernel_ridge_clone",
        n_parameters=int(np.asarray(reg.dual_coef_).size),
        hyperparameters={"alpha": alpha, "gamma": gamma},
        challenge_ids=ids,
    )


def fit_rff_ridge_fixed(
    train_features: dict[str, np.ndarray],
    train_templates: dict[str, np.ndarray],
    bundle: RepresentationBundle,
    cfg: ModelsConfig,
) -> RegressionPredictor:
    ids = sorted(train_features.keys() & train_templates.keys())
    X, Y = _build_xy(train_features, train_templates, bundle, ids)
    gamma = float(cfg.rff_gamma)
    alpha = float(cfg.rff_alpha)
    sampler = RBFSampler(gamma=gamma, n_components=cfg.rff_n_components, random_state=cfg.mlp_random_seed)
    Xt = sampler.fit_transform(X)
    reg = Ridge(alpha=alpha)
    reg.fit(Xt, Y)
    return RegressionPredictor(
        regressor=reg,
        bundle=bundle,
        model_name="C4_random_fourier_ridge_clone",
        n_parameters=int(reg.coef_.size + np.asarray(reg.intercept_).size),
        hyperparameters={"alpha": alpha, "gamma": gamma, "n_components": cfg.rff_n_components},
        feature_transform=sampler,
        challenge_ids=ids,
    )


def fit_mlp_fixed(
    train_features: dict[str, np.ndarray],
    train_templates: dict[str, np.ndarray],
    bundle: RepresentationBundle,
    cfg: ModelsConfig,
) -> MLPPredictor:
    """Fixed-epoch MLP. Uses train loss only; Round B never consulted."""
    ids = sorted(train_features.keys() & train_templates.keys())
    X, Y = _build_xy(train_features, train_templates, bundle, ids)
    mlp = MLPRegressor(
        hidden_layer_sizes=cfg.mlp_hidden_sizes,
        activation="relu",
        solver="adam",
        learning_rate_init=cfg.mlp_learning_rate,
        max_iter=1,
        warm_start=True,
        random_state=cfg.mlp_random_seed,
        n_iter_no_change=10**9,
    )
    t0 = time.time()
    for epoch in range(1, cfg.mlp_max_iter + 1):
        mlp.partial_fit(X, Y)
        if epoch == 1 or epoch % 5 == 0 or epoch == cfg.mlp_max_iter:
            logger.info(
                "[Track C] MLP epoch %d/%d train_loss=%.6f elapsed=%.1fs",
                epoch,
                cfg.mlp_max_iter,
                float(getattr(mlp, "loss_", float("nan"))),
                time.time() - t0,
            )
    return MLPPredictor(
        mlp=mlp,
        bundle=bundle,
        n_parameters=mlp_param_count(mlp),
        hyperparameters={
            "hidden_layer_sizes": list(cfg.mlp_hidden_sizes),
            "max_iter": cfg.mlp_max_iter,
            "learning_rate": cfg.mlp_learning_rate,
            "early_stopping_on_round_b": False,
        },
    )
