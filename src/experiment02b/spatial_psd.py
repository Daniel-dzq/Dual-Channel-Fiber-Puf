"""Spatial power-spectrum endpoints (protocol-locked)."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_normalized_psd(patch_windowed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (psd_shifted normalized sum=1 with DC zeroed, fx, fy) in cycles/pixel."""
    x = np.asarray(patch_windowed, dtype=np.float64)
    h, w = x.shape
    F = np.fft.fftshift(np.fft.fft2(x))
    psd = np.abs(F) ** 2
    cy, cx = h // 2, w // 2
    psd[cy, cx] = 0.0  # exclude DC
    total = float(np.sum(psd))
    if total < 1e-30:
        total = 1.0
    psd = psd / total
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    return psd, fx, fy


def radial_psd_curve(
    psd: np.ndarray, fx: np.ndarray, fy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    XX, YY = np.meshgrid(fx, fy)
    rr = np.sqrt(XX**2 + YY**2)
    # fine bins by unique rounded frequency rings
    rmax = 0.5
    n_bins = max(32, int(min(psd.shape) // 2))
    edges = np.linspace(0.0, rmax, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    vals = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_bins):
        m = (rr >= edges[i]) & (rr < edges[i + 1])
        vals[i] = float(np.sum(psd[m]))  # energy in ring
    return centers, vals


def _cum_freq(centers: np.ndarray, energy: np.ndarray, frac: float) -> float:
    e = np.asarray(energy, dtype=np.float64)
    c = np.asarray(centers, dtype=np.float64)
    s = float(np.sum(e))
    if s <= 0:
        return float("nan")
    cdf = np.cumsum(e) / s
    for i, v in enumerate(cdf):
        if v >= frac:
            if i == 0:
                return float(c[i])
            # interpolate
            v0 = float(cdf[i - 1])
            v1 = float(v)
            t = 0.0 if abs(v1 - v0) < 1e-18 else (frac - v0) / (v1 - v0)
            return float(c[i - 1] + t * (c[i] - c[i - 1]))
    return float(c[-1])


def extract_psd_metrics(
    patch_windowed: np.ndarray,
    *,
    high_freq_threshold: float = 0.1,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    psd, fx, fy = compute_normalized_psd(patch_windowed)
    XX, YY = np.meshgrid(fx, fy)
    rr = np.sqrt(XX**2 + YY**2)
    # pixel-wise centroid (exclude DC already zero)
    wsum = float(np.sum(psd))
    if wsum <= 1e-30:
        centroid = float("nan")
        rms_bw = float("nan")
    else:
        centroid = float(np.sum(rr * psd) / wsum)
        rms_bw = float(np.sqrt(np.sum(((rr - centroid) ** 2) * psd) / wsum))

    centers, radial_e = radial_psd_curve(psd, fx, fy)
    # entropy on valid positive bins of normalized 2D psd
    p = psd[psd > 0].ravel()
    n_valid = int(p.size)
    if n_valid == 0:
        ent_norm = float("nan")
        spr = float("nan")
        spr_norm = float("nan")
    else:
        ent = float(-np.sum(p * np.log(p + 1e-30)))
        ent_norm = float(ent / np.log(n_valid)) if n_valid > 1 else float("nan")
        spr = float(1.0 / np.sum(p**2))
        spr_norm = float(spr / n_valid)

    f50 = _cum_freq(centers, radial_e, 0.5)
    f90 = _cum_freq(centers, radial_e, 0.9)
    hf = float(np.sum(psd[rr > float(high_freq_threshold)]))

    metrics = {
        "psd_centroid_cyc_per_px": centroid,
        "psd_rms_bandwidth_cyc_per_px": rms_bw,
        "psd_entropy_normalized": ent_norm,
        "psd_f50_cyc_per_px": f50,
        "psd_f90_cyc_per_px": f90,
        "high_frequency_fraction_above_0p1": hf,
        "spatial_spectral_participation_ratio": spr,
        "spatial_spectral_participation_ratio_normalized": spr_norm,
        "psd_sum_check": float(np.sum(psd)),
    }
    profiles = {
        "psd_freq_cyc_per_px": centers,
        "psd_radial_energy": radial_e,
        "psd_2d": psd.astype(np.float32),
        "fx": fx.astype(np.float32),
        "fy": fy.astype(np.float32),
    }
    return metrics, profiles
