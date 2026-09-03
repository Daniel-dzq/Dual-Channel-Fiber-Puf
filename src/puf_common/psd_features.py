"""Spatial power-spectrum features with frozen frequency bands."""

from __future__ import annotations

from typing import Any

import numpy as np


def _tukey_window_2d(shape: tuple[int, int], alpha: float = 0.25) -> np.ndarray:
    h, w = shape
    wy = np.hanning(h) if alpha >= 1.0 else _tukey_1d(h, alpha)
    wx = np.hanning(w) if alpha >= 1.0 else _tukey_1d(w, alpha)
    return np.outer(wy, wx)


def _tukey_1d(n: int, alpha: float) -> np.ndarray:
    if alpha <= 0:
        return np.ones(n)
    x = np.linspace(0, 1, n)
    w = np.ones(n)
    edge = int(alpha * n / 2)
    if edge <= 0:
        return w
    for i in range(edge):
        w[i] = 0.5 * (1 - np.cos(np.pi * i / edge))
        w[-(i + 1)] = w[i]
    return w


def compute_psd_2d(
    img: np.ndarray,
    mask: np.ndarray,
    *,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (psd_shifted, fx, fy) with energy-normalized PSD."""
    h, w = img.shape
    mu = float(np.mean(img[mask])) if np.any(mask) else float(np.mean(img))
    x = (img - mu).astype(np.float64)
    x = np.where(mask, x, 0.0)
    if window == "tukey":
        win = _tukey_window_2d((h, w), alpha=0.25)
    else:
        win = np.outer(np.hanning(h), np.hanning(w))
    xw = x * win
    F = np.fft.fftshift(np.fft.fft2(xw))
    psd = np.abs(F) ** 2
    total = float(np.sum(psd))
    if total < 1e-30:
        total = 1.0
    psd = psd / total
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    return psd, fx, fy


def radial_average_psd(
    psd: np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
    n_bins: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Radial average; frequencies in cycles/sample (Nyquist=0.5)."""
    XX, YY = np.meshgrid(fx, fy)
    rr = np.sqrt(XX**2 + YY**2)
    rmax = 0.5
    edges = np.linspace(0.0, rmax, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    vals = np.full(n_bins, np.nan, dtype=np.float64)
    for i in range(n_bins):
        m = (rr >= edges[i]) & (rr < edges[i + 1])
        if np.any(m):
            vals[i] = float(np.mean(psd[m]))
    return centers, vals


def band_energy(centers: np.ndarray, radial: np.ndarray, lo: float, hi: float) -> float:
    m = (centers >= lo) & (centers < hi) & np.isfinite(radial)
    if not np.any(m):
        return float("nan")
    # approximate integral with bin widths
    return float(np.nansum(radial[m]))


def extract_psd_features(
    img: np.ndarray,
    mask: np.ndarray,
    *,
    window: str = "hann",
    n_radial_bins: int = 32,
    bands: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    if bands is None:
        bands = {"low": [0.0, 0.15], "mid": [0.15, 0.35], "high": [0.35, 0.50]}
    psd, fx, fy = compute_psd_2d(img, mask, window=window)
    centers, radial = radial_average_psd(psd, fx, fy, n_bins=n_radial_bins)
    e_low = band_energy(centers, radial, *bands["low"])
    e_mid = band_energy(centers, radial, *bands["mid"])
    e_high = band_energy(centers, radial, *bands["high"])
    e_tot = np.nansum([e_low, e_mid, e_high])
    if e_tot < 1e-30 or not np.isfinite(e_tot):
        e_tot = 1.0
    # spectral moments on radial PSD
    w = np.where(np.isfinite(radial), radial, 0.0)
    s = float(np.sum(w))
    if s < 1e-30:
        centroid = float("nan")
        bandwidth = float("nan")
        dominant = float("nan")
    else:
        centroid = float(np.sum(centers * w) / s)
        bandwidth = float(np.sqrt(np.sum(((centers - centroid) ** 2) * w) / s))
        dominant = float(centers[int(np.nanargmax(radial))])
    # spectral entropy
    p = w / (s + 1e-30)
    p = p[p > 0]
    sent = float(-np.sum(p * np.log(p + 1e-30))) if p.size else float("nan")
    # log-frequency slope (exclude DC bin)
    m = (centers > 0.02) & np.isfinite(radial) & (radial > 0)
    if np.sum(m) >= 3:
        lx = np.log(centers[m])
        ly = np.log(radial[m])
        slope = float(np.polyfit(lx, ly, 1)[0])
    else:
        slope = float("nan")
    out: dict[str, Any] = {
        "psd_spectral_centroid": centroid,
        "psd_spectral_bandwidth": bandwidth,
        "psd_spectral_entropy": sent,
        "psd_e_low_ratio": e_low / e_tot,
        "psd_e_mid_ratio": e_mid / e_tot,
        "psd_e_high_ratio": e_high / e_tot,
        "psd_dominant_radial_freq": dominant,
        "psd_log_slope": slope,
        "psd_qc": "ok" if np.isfinite(centroid) else "invalid",
    }
    for i, v in enumerate(radial):
        out[f"psd_radial_bin_{i:02d}"] = float(v) if np.isfinite(v) else float("nan")
    out["_psd_radial_centers"] = centers
    out["_psd_radial"] = radial
    return out
