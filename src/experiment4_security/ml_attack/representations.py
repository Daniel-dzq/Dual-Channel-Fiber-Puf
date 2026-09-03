"""Leakage-safe response representation: normalization + PCA (+ optional
train-only-frozen common-template subtraction).

Representation spaces
----------------------
- ``fullres_detail`` (primary): the full-resolution envelope-removed
  ``detail`` vector produced by `video_preprocessing.process_clip`, restricted
  to `valid_mask` pixels. No cross-challenge common template is involved, so
  there is no representation-leakage risk from this space by construction.
- ``fullres_detail_cm_train_frozen`` (optional secondary): ``detail`` minus a
  common template computed from TRAIN challenges only, then frozen. Every
  other sample (validation, test, target/held-out state) has the SAME frozen
  common template subtracted -- it is never recomputed on them.

Within either space, PCA and per-feature normalization are always fit on the
caller-supplied "fit" sample set only. Callers (Track A/B/C) are responsible
for making sure that fit set is exactly the intended train pool; this module
adds a defense-in-depth `forbidden_sample_ids` guard.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA

SPACE_DETAIL = "fullres_detail"
SPACE_DETAIL_CM_TRAIN_FROZEN = "fullres_detail_cm_train_frozen"


def _hash_ids(ids: list[str]) -> str:
    return hashlib.sha256(("|".join(sorted(ids))).encode("utf-8")).hexdigest()


def fit_common_template(train_vectors: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    ids = sorted(train_vectors.keys())
    stack = np.stack([train_vectors[i] for i in ids], axis=0).astype(np.float64)
    common = stack.mean(axis=0)
    return common.astype(np.float32), ids


def to_representation_space(
    vector_detail: np.ndarray, *, space: str, common_template: np.ndarray | None
) -> np.ndarray:
    if space == SPACE_DETAIL:
        return vector_detail
    if space == SPACE_DETAIL_CM_TRAIN_FROZEN:
        if common_template is None:
            raise ValueError("common_template required for fullres_detail_cm_train_frozen space")
        return vector_detail.astype(np.float64) - common_template.astype(np.float64)
    raise ValueError(f"Unknown representation space: {space}")


@dataclass
class RepresentationBundle:
    space: str
    pca_dimension: int
    mu: np.ndarray
    sigma: np.ndarray
    pca: PCA
    common_template: np.ndarray | None
    common_fit_sample_ids: list[str]
    normalization_fit_sample_ids: list[str]
    pca_fit_sample_ids: list[str]
    fit_data_hash: str
    explained_variance_ratio: np.ndarray = field(repr=False)


def fit_representation(
    train_vectors: dict[str, np.ndarray],
    *,
    space: str,
    pca_dimension: int,
    forbidden_sample_ids: set[str] | None = None,
) -> RepresentationBundle:
    fit_ids = sorted(train_vectors.keys())
    if forbidden_sample_ids:
        overlap = set(fit_ids) & set(forbidden_sample_ids)
        if overlap:
            raise RuntimeError(
                f"Representation fit set overlaps forbidden (val/test/held-out) "
                f"sample_ids: {sorted(overlap)[:10]}"
            )
    if len(fit_ids) < 2:
        raise ValueError(f"Need >=2 fit samples, got {len(fit_ids)}")

    common_template: np.ndarray | None = None
    common_fit_ids: list[str] = []
    if space == SPACE_DETAIL_CM_TRAIN_FROZEN:
        common_template, common_fit_ids = fit_common_template(train_vectors)

    space_vectors = {
        cid: to_representation_space(v, space=space, common_template=common_template)
        for cid, v in train_vectors.items()
    }
    X = np.stack([space_vectors[i] for i in fit_ids], axis=0).astype(np.float64)

    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma_safe = np.where(sigma < 1e-8, 1.0, sigma)
    Xn = (X - mu) / sigma_safe

    n_comp = int(min(pca_dimension, Xn.shape[0], Xn.shape[1]))
    pca = PCA(n_components=n_comp, svd_solver="full", random_state=0)
    pca.fit(Xn)

    fit_data_hash = hashlib.sha256(
        f"space={space};pca_dim={pca_dimension};n_comp={n_comp};ids={'|'.join(fit_ids)}".encode("utf-8")
    ).hexdigest()

    return RepresentationBundle(
        space=space,
        pca_dimension=n_comp,
        mu=mu.astype(np.float32),
        sigma=sigma_safe.astype(np.float32),
        pca=pca,
        common_template=common_template,
        common_fit_sample_ids=common_fit_ids,
        normalization_fit_sample_ids=fit_ids,
        pca_fit_sample_ids=fit_ids,
        fit_data_hash=fit_data_hash,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )


def to_pca_coefficients(vector_detail: np.ndarray, bundle: RepresentationBundle) -> np.ndarray:
    x = to_representation_space(vector_detail, space=bundle.space, common_template=bundle.common_template)
    xn = (x.astype(np.float64) - bundle.mu) / bundle.sigma
    return bundle.pca.transform(xn.reshape(1, -1)).ravel()


def to_eval_vector(vector_detail: np.ndarray, bundle: RepresentationBundle) -> np.ndarray:
    """Ground-truth measured vector expressed in `bundle.space`, for a fair
    apples-to-apples comparison against `inverse_transform` reconstructions."""
    return to_representation_space(vector_detail, space=bundle.space, common_template=bundle.common_template)


def inverse_transform(coeffs: np.ndarray, bundle: RepresentationBundle) -> np.ndarray:
    xn = bundle.pca.inverse_transform(coeffs.reshape(1, -1)).ravel()
    x = xn * bundle.sigma + bundle.mu
    return x.astype(np.float32)


def representation_audit_dict(
    bundle: RepresentationBundle,
    *,
    validation_ids: set[str],
    test_ids: set[str],
    held_out_state_ids: set[str],
) -> dict:
    fit_set = set(bundle.pca_fit_sample_ids) | set(bundle.common_fit_sample_ids)
    return {
        "representation_space": bundle.space,
        "pca_dimension_effective": bundle.pca_dimension,
        "pca_fit_sample_ids": bundle.pca_fit_sample_ids,
        "normalization_fit_sample_ids": bundle.normalization_fit_sample_ids,
        "common_template_fit_sample_ids": bundle.common_fit_sample_ids,
        "fit_data_hash": bundle.fit_data_hash,
        "explained_variance_ratio": [float(v) for v in bundle.explained_variance_ratio],
        "includes_validation_samples": bool(fit_set & validation_ids),
        "includes_test_samples": bool(fit_set & test_ids),
        "includes_held_out_state_samples": bool(fit_set & held_out_state_ids),
        "leakage_detected": bool(fit_set & (validation_ids | test_ids | held_out_state_ids)),
    }
