"""Intensity / speckle statistics within a frozen valid mask."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def extract_intensity_features(
    img: np.ndarray,
    mask: np.ndarray,
    *,
    eps: float = 1e-8,
    sat_level: float | None = None,
    low_frac: float = 0.05,
) -> dict[str, Any]:
    vals = img[mask].astype(np.float64)
    if vals.size == 0:
        return {k: float("nan") for k in [
            "int_mean", "int_std", "int_cv", "speckle_contrast",
            "int_skew", "int_kurtosis", "int_entropy",
            "int_p05", "int_p25", "int_p50", "int_p75", "int_p95",
            "int_saturation_frac", "int_low_signal_frac",
            "int_mean_norm", "int_std_norm", "speckle_contrast_norm",
        ]} | {"int_qc": "empty_mask"}

    mu = float(np.mean(vals))
    sd = float(np.std(vals))
    cv = sd / (abs(mu) + eps)
    contrast = sd / (abs(mu) + eps)
    sk = float(stats.skew(vals))
    ku = float(stats.kurtosis(vals, fisher=True))
    # histogram entropy
    hist, _ = np.histogram(vals, bins=64, density=True)
    hist = hist[hist > 0]
    bin_w = (vals.max() - vals.min()) / 64.0 + eps
    ent = float(-np.sum(hist * np.log(hist + eps) * bin_w))
    pcts = np.percentile(vals, [5, 25, 50, 75, 95])
    if sat_level is None:
        # assume 8-bit-ish if max near 255, else top percentile
        sat_level = 250.0 if float(np.max(vals)) <= 260 else float(np.percentile(vals, 99.5))
    sat_frac = float(np.mean(vals >= sat_level))
    low_thr = float(np.percentile(vals, low_frac * 100)) if False else float(np.quantile(vals, 0.05))
    # low-signal: below 5% of dynamic range from min
    dyn = float(np.max(vals) - np.min(vals)) + eps
    low_frac_val = float(np.mean(vals <= (np.min(vals) + 0.05 * dyn)))

    # normalized versions relative to local mean (power-robust)
    vals_n = vals / (abs(mu) + eps)
    mu_n = float(np.mean(vals_n))
    sd_n = float(np.std(vals_n))
    contrast_n = sd_n / (abs(mu_n) + eps)

    return {
        "int_mean": mu,
        "int_std": sd,
        "int_cv": cv,
        "speckle_contrast": contrast,
        "int_skew": sk,
        "int_kurtosis": ku,
        "int_entropy": ent,
        "int_p05": float(pcts[0]),
        "int_p25": float(pcts[1]),
        "int_p50": float(pcts[2]),
        "int_p75": float(pcts[3]),
        "int_p95": float(pcts[4]),
        "int_saturation_frac": sat_frac,
        "int_low_signal_frac": low_frac_val,
        "int_mean_norm": mu_n,
        "int_std_norm": sd_n,
        "speckle_contrast_norm": contrast_n,
        "int_qc": "ok",
    }
