"""Red Fiber-ID descriptor: nine active features extracted from the temporally aggregated red
intensity image (no envelope normalization, no common-component subtraction).

Frozen PSD parameters: 32 radial bins over [0, 0.5] cycles/pixel; bands low [0.00, 0.15),
intermediate [0.15, 0.35), high [0.35, 0.50).

Stored feature tables in the public dataset (e.g. ``threshold_development/red_features.csv``)
also carry four ``acf_*`` coordinates that are identically zero; they are storage-level
compatibility columns and are not part of the nine-dimensional identity descriptor.
"""

from __future__ import annotations

import numpy as np

from puf_common.acf_features import extract_acf_features_fiber_id
from puf_common.intensity_features import extract_intensity_features
from puf_common.psd_features import extract_psd_features
from puf_common.texture_features import extract_texture_features

FIBER_ID_ACF_IMPLEMENTATION = "from_origin_first_half_max"
FIBER_ID_ACF_MAX_LAG = 64

# Storage-level zero columns present in stored feature tables (not part of the descriptor).
FIBER_ID_INACTIVE_ACF_NAMES: tuple[str, ...] = (
    "acf_fwhm_x",
    "acf_fwhm_y",
    "acf_fwhm_radial",
    "acf_anisotropy",
)

FIBER_ID_FEATURE_NAMES: tuple[str, ...] = (
    "psd_spectral_centroid",
    "psd_spectral_bandwidth",
    "psd_spectral_entropy",
    "psd_e_low_ratio",
    "psd_e_mid_ratio",
    "psd_e_high_ratio",
    "speckle_contrast_norm",
    "tex_grad_mean",
    "tex_laplacian_energy",
)
FIBER_ID_PSD_N_RADIAL_BINS = 32
FIBER_ID_PSD_BANDS: dict[str, list[float]] = {
    "low": [0.00, 0.15],
    "mid": [0.15, 0.35],
    "high": [0.35, 0.50],
}


def extract_fiber_id_scalars(
    img: np.ndarray,
    mask: np.ndarray,
    *,
    names: tuple[str, ...] | list[str] | None = None,
) -> dict[str, float]:
    """Compute Fiber-ID scalars. ACF is computed only if an ACF name is requested."""
    requested = tuple(FIBER_ID_FEATURE_NAMES if names is None else names)
    need_acf = any(n.startswith("acf_") for n in requested)
    parts: list[dict] = []
    if need_acf:
        parts.append(extract_acf_features_fiber_id(img, mask, max_lag=FIBER_ID_ACF_MAX_LAG))
    parts.append(
        extract_psd_features(
            img,
            mask,
            n_radial_bins=FIBER_ID_PSD_N_RADIAL_BINS,
            bands=FIBER_ID_PSD_BANDS,
        )
    )
    parts.append(extract_intensity_features(img, mask))
    parts.append(extract_texture_features(img, mask))
    scalars: dict[str, float] = {}
    for d in parts:
        for k, v in d.items():
            if k.startswith("_") or str(k).endswith("_qc") or isinstance(v, str):
                continue
            if str(k).startswith("psd_radial_bin"):
                continue
            try:
                scalars[k] = float(v)
            except (TypeError, ValueError):
                continue
    return scalars


def vector_from_scalars(
    scalars: dict[str, float],
    names: tuple[str, ...] | list[str],
) -> np.ndarray:
    vec = np.array([scalars.get(name, 0.0) for name in names], dtype=np.float64)
    return np.nan_to_num(vec, nan=0.0)


def extract_fiber_id_features(
    img: np.ndarray,
    mask: np.ndarray,
    *,
    names: tuple[str, ...] | list[str] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return (vector in ``names`` order, scalar dict); ``names`` defaults to the nine active features."""
    names = tuple(FIBER_ID_FEATURE_NAMES if names is None else names)
    scalars = extract_fiber_id_scalars(img, mask, names=names)
    return vector_from_scalars(scalars, names), scalars


def extract_fiber_id_vector(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vec, _ = extract_fiber_id_features(img, mask)
    return vec


def extract_fiber_id_vector_16(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Historical 16-D vector (inactive ACF + 9-D + intensity extras)."""
    vec, _ = extract_fiber_id_features(img, mask, names=FIBER_ID_FEATURE_NAMES_16)
    return vec
