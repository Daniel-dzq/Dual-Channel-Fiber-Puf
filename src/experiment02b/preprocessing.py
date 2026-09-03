"""Background estimate, centroid crop, and detail representation (no common-mode)."""

from __future__ import annotations

from typing import Any

import numpy as np

from puf_common.features import to_detail
from puf_common.envelope import gaussian_blur


def estimate_edge_background(image: np.ndarray, edge_fraction: float = 0.08) -> float:
    """Robust median of a border band (fallback when no dark video)."""
    img = np.asarray(image, dtype=np.float64)
    h, w = img.shape[:2]
    band = max(1, int(round(edge_fraction * min(h, w))))
    border = np.concatenate(
        [
            img[:band, :].ravel(),
            img[-band:, :].ravel(),
            img[:, :band].ravel(),
            img[:, -band:].ravel(),
        ]
    )
    return float(np.median(border))


def subtract_background(
    image: np.ndarray,
    *,
    dark: np.ndarray | None = None,
    edge_fraction: float = 0.08,
) -> tuple[np.ndarray, dict[str, Any]]:
    img = np.asarray(image, dtype=np.float64)
    if dark is not None:
        out = img - np.asarray(dark, dtype=np.float64)
        mode = "true_dark"
        bg_level = float(np.median(dark))
    else:
        bg_level = estimate_edge_background(img, edge_fraction=edge_fraction)
        out = img - bg_level
        mode = "edge_median_estimate"
    out = np.clip(out, 0.0, None)
    return out.astype(np.float32), {"background_mode": mode, "background_level": bg_level}


def intensity_centroid(
    image: np.ndarray,
    *,
    smooth_sigma: float = 24.0,
) -> tuple[float, float, dict[str, Any]]:
    img = np.asarray(image, dtype=np.float64)
    if smooth_sigma > 0:
        sm = gaussian_blur(img, sigma=float(smooth_sigma))
    else:
        sm = img
    pos = np.clip(sm, 0.0, None)
    total = float(pos.sum())
    h, w = pos.shape
    if total <= 1e-12:
        return 0.5 * (w - 1), 0.5 * (h - 1), {"centroid_ok": False, "reason": "zero_intensity"}
    yy, xx = np.indices(pos.shape, dtype=np.float64)
    cx = float((pos * xx).sum() / total)
    cy = float((pos * yy).sum() / total)
    return cx, cy, {"centroid_ok": True, "reason": ""}


def crop_fixed_square(
    image: np.ndarray,
    cx: float,
    cy: float,
    size: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Crop size×size centered at (cx,cy). No resize. Fail if OOB (no pad)."""
    img = np.asarray(image)
    h, w = img.shape[:2]
    half = size // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1, y1 = x0 + size, y0 + size
    info = {
        "crop_x0": x0,
        "crop_y0": y0,
        "crop_size": size,
        "image_w": w,
        "image_h": h,
    }
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        info["ok"] = False
        info["reason"] = "crop_out_of_bounds"
        return None, info
    info["ok"] = True
    info["reason"] = ""
    return img[y0:y1, x0:x1].copy(), info


def center_inner_square(image: np.ndarray, inner: int) -> np.ndarray:
    """Extract centered inner×inner from a square crop. No resize."""
    img = np.asarray(image)
    h, w = img.shape[:2]
    if h != w:
        raise ValueError(f"expected square crop, got {img.shape}")
    if inner > h:
        raise ValueError(f"inner {inner} > crop {h}")
    off = (h - inner) // 2
    return img[off : off + inner, off : off + inner].copy()


def make_detail(
    image: np.ndarray,
    *,
    sigma: float = 42.0,
    epsilon: float = 1.0,
) -> np.ndarray:
    """Local-ratio detail via puf_common (algebraically = (I-E)/(E+eps))."""
    return to_detail(np.asarray(image, dtype=np.float32), sigma=float(sigma), eps=float(epsilon)).astype(
        np.float32
    )


def hann2d(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    return np.outer(np.hanning(h), np.hanning(w)).astype(np.float64)


def prepare_analysis_patch(
    detail_crop: np.ndarray,
    *,
    inner: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (windowed zero-mean patch, boolean mask full True)."""
    patch = center_inner_square(detail_crop, inner).astype(np.float64)
    patch = patch - float(patch.mean())
    win = hann2d(patch.shape)
    return patch * win, np.ones(patch.shape, dtype=bool)


def beam_envelope_qc(image: np.ndarray) -> dict[str, float]:
    """Simple intensity-envelope descriptors on background-corrected median (not mechanism endpoints)."""
    img = np.asarray(image, dtype=np.float64)
    pos = np.clip(img, 0.0, None)
    total = float(pos.sum())
    h, w = pos.shape
    yy, xx = np.indices(pos.shape, dtype=np.float64)
    if total <= 1e-12:
        return {
            "centroid_x": float("nan"),
            "centroid_y": float("nan"),
            "equivalent_beam_radius": float("nan"),
            "beam_area": float("nan"),
            "major_axis_length": float("nan"),
            "minor_axis_length": float("nan"),
            "ellipticity": float("nan"),
            "orientation": float("nan"),
            "signal_to_background_ratio": float("nan"),
            "saturation_fraction": float(np.mean(img >= 250)),
            "dark_fraction": float(np.mean(img <= 1.0)),
        }
    cx = float((pos * xx).sum() / total)
    cy = float((pos * yy).sum() / total)
    # second moments
    dx = xx - cx
    dy = yy - cy
    mxx = float((pos * dx * dx).sum() / total)
    myy = float((pos * dy * dy).sum() / total)
    mxy = float((pos * dx * dy).sum() / total)
    # eigenvalues of covariance
    tr = mxx + myy
    det = mxx * myy - mxy * mxy
    disc = max(0.0, tr * tr - 4.0 * det)
    l1 = 0.5 * (tr + np.sqrt(disc))
    l2 = 0.5 * (tr - np.sqrt(disc))
    r_eq = float(np.sqrt(max(l1 + l2, 0.0)))
    major = 2.0 * float(np.sqrt(max(l1, 0.0)))
    minor = 2.0 * float(np.sqrt(max(l2, 0.0)))
    ellip = float(1.0 - (minor / major)) if major > 1e-12 else float("nan")
    orient = 0.5 * float(np.arctan2(2 * mxy, mxx - myy))
    # SBR: mean of top 10% vs edge background
    thr = float(np.quantile(pos, 0.9))
    sig = float(pos[pos >= thr].mean()) if np.any(pos >= thr) else float(pos.mean())
    bg = estimate_edge_background(img, 0.08)
    sbr = float(sig / (bg + 1e-6))
    return {
        "centroid_x": cx,
        "centroid_y": cy,
        "equivalent_beam_radius": r_eq,
        "beam_area": float(np.pi * r_eq * r_eq),
        "major_axis_length": major,
        "minor_axis_length": minor,
        "ellipticity": ellip,
        "orientation": orient,
        "signal_to_background_ratio": sbr,
        "saturation_fraction": float(np.mean(img >= 250)),
        "dark_fraction": float(np.mean(img <= 1.0)),
    }


def roi_coverage(detail_inner: np.ndarray, *, abs_thresh: float = 1e-6) -> float:
    """Fraction of analysis pixels with non-trivial |detail| after windowing prep."""
    x = np.asarray(detail_inner)
    return float(np.mean(np.abs(x) > abs_thresh))
