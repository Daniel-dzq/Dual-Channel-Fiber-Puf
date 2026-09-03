"""2D autocorrelation features with QC flags."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage


def _masked_mean_std(img: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    vals = img[mask]
    if vals.size == 0:
        return 0.0, 1.0
    mu = float(np.mean(vals))
    sd = float(np.std(vals))
    if sd < 1e-12:
        sd = 1.0
    return mu, sd


def normalized_acf_2d(img: np.ndarray, mask: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute central crop of normalized 2D ACF via FFT."""
    h, w = img.shape
    mu, sd = _masked_mean_std(img, mask)
    x = np.zeros_like(img, dtype=np.float64)
    x[mask] = (img[mask] - mu) / sd
    # zero outside mask
    fx = np.fft.rfft2(x, s=(2 * h, 2 * w))
    acf_full = np.fft.irfft2(np.abs(fx) ** 2, s=(2 * h, 2 * w)).real
    acf_full = np.fft.fftshift(acf_full)
    cy, cx = h, w
    # peak at center of doubled array after shift: (h, w)
    peaki, peakj = h, w
    # normalize by zero-lag
    zero = acf_full[peaki, peakj]
    if abs(zero) < 1e-18:
        zero = 1.0
    acf_full = acf_full / zero
    half = max_lag
    yi0, yi1 = peaki - half, peaki + half + 1
    xi0, xi1 = peakj - half, peakj + half + 1
    yi0 = max(0, yi0)
    xi0 = max(0, xi0)
    yi1 = min(acf_full.shape[0], yi1)
    xi1 = min(acf_full.shape[1], xi1)
    crop = acf_full[yi0:yi1, xi0:xi1]
    return crop


def _fwhm_1d(profile: np.ndarray, dx: float = 1.0) -> tuple[float, str]:
    """Half-max width of a 1D ACF profile centered at index mid."""
    n = len(profile)
    mid = n // 2
    peak = float(profile[mid])
    if not np.isfinite(peak) or peak <= 0:
        return float("nan"), "peak_invalid"
    half = 0.5 * peak
    # search right
    right = None
    for i in range(mid + 1, n):
        if profile[i] <= half:
            # linear interpolate
            y0, y1 = float(profile[i - 1]), float(profile[i])
            if abs(y1 - y0) < 1e-18:
                right = (i - mid) * dx
            else:
                t = (half - y0) / (y1 - y0)
                right = ((i - 1) + t - mid) * dx
            break
    left = None
    for i in range(mid - 1, -1, -1):
        if profile[i] <= half:
            y0, y1 = float(profile[i + 1]), float(profile[i])
            if abs(y1 - y0) < 1e-18:
                left = (mid - i) * dx
            else:
                t = (half - y0) / (y1 - y0)
                left = (mid - (i + t)) * dx
            break
    if left is None and right is None:
        return float("nan"), "half_max_not_found"
    if left is None:
        return 2.0 * float(right), "one_sided_right"
    if right is None:
        return 2.0 * float(left), "one_sided_left"
    return float(left + right), "ok"


def radial_acf_profile(acf: np.ndarray) -> np.ndarray:
    cy, cx = np.array(acf.shape) // 2
    yy, xx = np.indices(acf.shape)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = int(min(cy, cx))
    profile = np.zeros(rmax + 1, dtype=np.float64)
    counts = np.zeros(rmax + 1, dtype=np.float64)
    rbin = np.clip(np.round(rr).astype(int), 0, rmax)
    for r in range(rmax + 1):
        m = rbin == r
        if np.any(m):
            profile[r] = float(np.mean(acf[m]))
            counts[r] = float(np.sum(m))
    return profile


def second_moment_radius(acf: np.ndarray, mask_radius: int | None = None) -> float:
    cy, cx = np.array(acf.shape) // 2
    yy, xx = np.indices(acf.shape, dtype=np.float64)
    dy = yy - cy
    dx = xx - cx
    r2 = dy * dy + dx * dx
    if mask_radius is None:
        mask_radius = int(min(cy, cx))
    m = r2 <= mask_radius * mask_radius
    w = np.clip(acf, 0, None)
    w = w * m
    s = float(np.sum(w))
    if s < 1e-18:
        return float("nan")
    return float(np.sqrt(np.sum(w * r2) / s))


def first_half_maximum_radius_from_origin(
    profile: np.ndarray,
    *,
    level: float = 0.5,
) -> tuple[float, str]:
    """First outward half-maximum radius on a one-sided profile with r=0 at index 0.

    Linear interpolation between adjacent samples. Later oscillations ignored.
    """
    p = np.asarray(profile, dtype=np.float64)
    if p.size < 2:
        return float("nan"), "too_short"
    for i in range(1, len(p)):
        if not np.isfinite(p[i - 1]) or not np.isfinite(p[i]):
            continue
        if p[i] <= level:
            y0, y1 = float(p[i - 1]), float(p[i])
            if abs(y1 - y0) < 1e-18:
                return float(i), "ok"
            t = (level - y0) / (y1 - y0)
            return float((i - 1) + t), "ok"
    return float("nan"), "half_max_not_found"


def radial_acf_fwhm_from_origin(acf: np.ndarray) -> tuple[float, str]:
    """Radial ACF FWHM = 2 * r_{1/2} from the first A_r=0.5 crossing at r=0.

    Normalizes A_r(0) to 1 before the crossing search. Do not pass the
    one-sided radial profile to ``_fwhm_1d`` (that helper assumes a centered
    two-sided line and is retained only for horizontal/vertical FWHM and
    legacy callers).
    """
    rad = radial_acf_profile(acf)
    if rad.size < 2:
        return float("nan"), "too_short"
    peak = float(rad[0])
    if not np.isfinite(peak) or peak <= 0:
        return float("nan"), "peak_invalid"
    rad_n = rad / peak
    r50, qc = first_half_maximum_radius_from_origin(rad_n, level=0.5)
    if not np.isfinite(r50):
        return float("nan"), qc
    return float(2.0 * r50), qc


def extract_acf_features(
    img: np.ndarray,
    mask: np.ndarray,
    *,
    max_lag: int = 64,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Legacy ACF scalars. ``acf_fwhm_radial`` uses centered ``_fwhm_1d``.

    Fiber-ID must call ``extract_acf_features_fiber_id`` instead. This default
    is frozen for Experiment 0 / other non-Fiber-ID consumers.
    """
    acf = normalized_acf_2d(img, mask, max_lag=max_lag)
    cy, cx = np.array(acf.shape) // 2
    horiz = acf[cy, :]
    vert = acf[:, cx]
    wx, qx = _fwhm_1d(horiz)
    wy, qy = _fwhm_1d(vert)
    rad = radial_acf_profile(acf)
    wr, qr = _fwhm_1d(rad)
    r2m = second_moment_radius(acf)
    if not (np.isfinite(wx) and np.isfinite(wy)):
        aniso = float("nan")
        aniso_qc = "undefined"
    else:
        aniso = abs(wx - wy) / (wx + wy + eps)
        aniso_qc = "ok"
    qc_parts = [qx, qy, qr, aniso_qc]
    qc = "ok" if all(p == "ok" for p in qc_parts) else "|".join(qc_parts)
    return {
        "acf_fwhm_x": wx,
        "acf_fwhm_y": wy,
        "acf_fwhm_radial": wr,
        "acf_second_moment_radius": r2m,
        "acf_anisotropy": aniso,
        "acf_qc": qc,
        "acf_peak": float(acf[cy, cx]),
    }


def extract_acf_features_fiber_id(
    img: np.ndarray,
    mask: np.ndarray,
    *,
    max_lag: int = 64,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Fiber-ID ACF scalars: x/y FWHM unchanged; radial FWHM from r=0."""
    acf = normalized_acf_2d(img, mask, max_lag=max_lag)
    cy, cx = np.array(acf.shape) // 2
    horiz = acf[cy, :]
    vert = acf[:, cx]
    wx, qx = _fwhm_1d(horiz)
    wy, qy = _fwhm_1d(vert)
    wr, qr = radial_acf_fwhm_from_origin(acf)
    r2m = second_moment_radius(acf)
    if not (np.isfinite(wx) and np.isfinite(wy)):
        aniso = float("nan")
        aniso_qc = "undefined"
    else:
        aniso = abs(wx - wy) / (wx + wy + eps)
        aniso_qc = "ok"
    qc_parts = [qx, qy, qr, aniso_qc]
    qc = "ok" if all(p == "ok" for p in qc_parts) else "|".join(qc_parts)
    return {
        "acf_fwhm_x": wx,
        "acf_fwhm_y": wy,
        "acf_fwhm_radial": wr,
        "acf_second_moment_radius": r2m,
        "acf_anisotropy": aniso,
        "acf_qc": qc,
        "acf_peak": float(acf[cy, cx]),
        "acf_radial_fwhm_mode": "from_origin_first_half_max",
    }
