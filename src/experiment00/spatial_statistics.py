"""Branch A: physical spatial statistics on dark-corrected raw green (not detail_cm)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from experiment00.config import Experiment00Config
from experiment00.video_processing import ProcessedVideo
from puf_common.acf_features import extract_acf_features
from puf_common.envelope import estimate_speckle_width, speckle_contrast
from puf_common.intensity_features import extract_intensity_features
from puf_common.psd_features import extract_psd_features


def spatial_metrics_for_image(
    img: np.ndarray,
    mask: np.ndarray | None,
    cfg: Experiment00Config,
) -> dict:
    m = mask.astype(bool) if mask is not None else None
    sel = img[m] if m is not None else img.ravel()
    mean_i = float(np.mean(sel)) if sel.size else float("nan")
    std_i = float(np.std(sel)) if sel.size else float("nan")
    snr = float(mean_i / std_i) if std_i and np.isfinite(std_i) and std_i > 0 else float("nan")
    sat = float(np.mean(sel >= 250)) if sel.size else float("nan")
    c_sp = float(speckle_contrast(img, mask=m, signed=False))
    inten = extract_intensity_features(img, m)
    try:
        acf = extract_acf_features(img, m, max_lag=64)
        acf_w = float(acf.get("acf_fwhm_radial", np.nan))
    except Exception:  # noqa: BLE001
        acf = {}
        acf_w = float(estimate_speckle_width(img, mask=m))
    cap = float(cfg.quality_control.acf_width_cap_px)
    censored = bool(np.isfinite(acf_w) and acf_w >= cap)
    if censored:
        acf_w = cap
    try:
        psd = extract_psd_features(img, m, n_radial_bins=24)
    except Exception:  # noqa: BLE001
        psd = {}
    a_valid = float(np.count_nonzero(m)) if m is not None else float(img.size)
    n_eff = float(a_valid / (np.pi * (max(acf_w, 1e-6) / 2.0) ** 2)) if np.isfinite(acf_w) else float("nan")
    return {
        "mean_intensity": mean_i,
        "std_intensity": std_i,
        "snr": snr,
        "saturation_fraction": sat,
        "speckle_contrast": c_sp,
        "int_entropy": float(inten.get("int_entropy", np.nan)),
        "acf_width_px": acf_w,
        "acf_width_censored": censored,
        "psd_spectral_entropy": float(psd.get("psd_spectral_entropy", np.nan)),
        "n_eff": n_eff,
        "valid_pixels": a_valid,
    }


def compute_spatial_metrics_table(
    green: list[ProcessedVideo],
    mask: np.ndarray | None,
    cfg: Experiment00Config,
) -> pd.DataFrame:
    rows = []
    for rec in tqdm(green, desc="Spatial metrics", unit="vid"):
        mets = spatial_metrics_for_image(rec.representative_raw, mask, cfg)
        rows.append(
            {
                "length_cm": rec.length_cm,
                "fiber_id": rec.fiber_id,
                "round": rec.round,
                "challenge": rec.challenge,
                "challenge_id": getattr(rec, "challenge_id", None) or rec.challenge,
                "sequence_id": getattr(rec, "sequence_id", None),
                "acquisition_position": getattr(rec, "acquisition_position", None),
                "order_source": getattr(rec, "order_source", None),
                "filename": rec.path.name,
                **mets,
            }
        )
    return pd.DataFrame(rows)


def summarize_spatial_by_length(per_video: pd.DataFrame) -> pd.DataFrame:
    if per_video.empty:
        return per_video
    metrics = [
        "mean_intensity",
        "snr",
        "speckle_contrast",
        "acf_width_px",
        "int_entropy",
        "n_eff",
        "psd_spectral_entropy",
    ]
    rows = []
    for L, sub in per_video.groupby("length_cm"):
        row = {"length_cm": int(L), "n_videos": int(len(sub))}
        for m in metrics:
            if m in sub.columns:
                row[f"{m}_median"] = float(sub[m].median())
                row[f"{m}_q25"] = float(sub[m].quantile(0.25))
                row[f"{m}_q75"] = float(sub[m].quantile(0.75))
        rows.append(row)
    return pd.DataFrame(rows)
