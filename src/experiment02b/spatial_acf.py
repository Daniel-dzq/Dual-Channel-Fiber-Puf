"""2D spatial autocorrelation endpoints (protocol-locked definitions)."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiment02b.preprocessing import hann2d


def normalized_acf_fft(patch_windowed: np.ndarray) -> np.ndarray:
    """ACF from already mean-subtracted + Hann-windowed patch; ACF(0,0)=1."""
    x = np.asarray(patch_windowed, dtype=np.float64)
    F = np.fft.fft2(x)
    acf = np.fft.ifft2(np.abs(F) ** 2).real
    acf = np.fft.fftshift(acf)
    cy, cx = np.array(acf.shape) // 2
    zero = float(acf[cy, cx])
    if abs(zero) < 1e-18:
        zero = 1.0
    return acf / zero


def radial_acf_profile(acf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (radius_px, mean_acf)."""
    cy, cx = np.array(acf.shape) // 2
    yy, xx = np.indices(acf.shape)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = int(min(cy, cx))
    radii = np.arange(rmax + 1, dtype=np.float64)
    profile = np.zeros_like(radii)
    for r in range(rmax + 1):
        m = (np.round(rr).astype(int) == r)
        if np.any(m):
            profile[r] = float(np.mean(acf[m]))
        else:
            profile[r] = np.nan
    return radii, profile


def first_crossing_radius(radii: np.ndarray, profile: np.ndarray, level: float) -> float:
    """First radius where profile drops to `level`, linear interpolation."""
    r = np.asarray(radii, dtype=np.float64)
    p = np.asarray(profile, dtype=np.float64)
    if r.size < 2:
        return float("nan")
    # start from r=0
    for i in range(1, len(p)):
        if not np.isfinite(p[i - 1]) or not np.isfinite(p[i]):
            continue
        if p[i] <= level <= p[i - 1] or p[i] <= level:
            y0, y1 = float(p[i - 1]), float(p[i])
            x0, x1 = float(r[i - 1]), float(r[i])
            if abs(y1 - y0) < 1e-18:
                return x1
            t = (level - y0) / (y1 - y0)
            return float(x0 + t * (x1 - x0))
    return float("nan")


def effective_correlation_area(acf: np.ndarray, *, positive_only: bool = True) -> float:
    """Integral of clipped positive ACF (pixel^2); pre-specified, not tuned post-hoc."""
    a = np.asarray(acf, dtype=np.float64)
    w = np.clip(a, 0.0, None) if positive_only else a
    return float(np.sum(w))


def extract_acf_metrics(patch_windowed: np.ndarray) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    acf = normalized_acf_fft(patch_windowed)
    radii, profile = radial_acf_profile(acf)
    r50 = first_crossing_radius(radii, profile, 0.5)
    r1e = first_crossing_radius(radii, profile, float(np.e) ** -1)
    area = effective_correlation_area(acf)
    analysis_area = float(patch_windowed.shape[0] * patch_windowed.shape[1])
    n_eff = float(analysis_area / area) if area > 1e-12 else float("nan")
    # optional anisotropy (secondary)
    cy, cx = np.array(acf.shape) // 2
    prof_x = acf[cy, :]
    prof_y = acf[:, cx]
    rx = first_crossing_radius(np.arange(len(prof_x)) - cx, prof_x, 0.5)
    ry = first_crossing_radius(np.arange(len(prof_y)) - cy, prof_y, 0.5)
    # use absolute distance from center along axes
    def _axis_r50(line: np.ndarray, center: int) -> float:
        # right side
        for i in range(center + 1, len(line)):
            if line[i] <= 0.5:
                y0, y1 = float(line[i - 1]), float(line[i])
                if abs(y1 - y0) < 1e-18:
                    return float(i - center)
                t = (0.5 - y0) / (y1 - y0)
                return float((i - 1) + t - center)
        return float("nan")

    acf_width_x = _axis_r50(prof_x, cx) * 2.0
    acf_width_y = _axis_r50(prof_y, cy) * 2.0
    aniso = (
        float(max(acf_width_x, acf_width_y) / min(acf_width_x, acf_width_y))
        if np.isfinite(acf_width_x) and np.isfinite(acf_width_y) and min(acf_width_x, acf_width_y) > 1e-12
        else float("nan")
    )
    metrics = {
        "acf_r50_px": float(r50),
        "acf_fwhm_px": float(2.0 * r50) if np.isfinite(r50) else float("nan"),
        "acf_r1e_px": float(r1e),
        "acf_diameter_1e_px": float(2.0 * r1e) if np.isfinite(r1e) else float("nan"),
        "effective_correlation_area": float(area),
        "effective_speckle_degrees_of_freedom": float(n_eff),
        "acf_width_x": float(acf_width_x),
        "acf_width_y": float(acf_width_y),
        "acf_anisotropy_ratio": float(aniso),
    }
    profiles = {"acf_radius_px": radii, "acf_radial": profile, "acf_2d": acf.astype(np.float32)}
    return metrics, profiles
