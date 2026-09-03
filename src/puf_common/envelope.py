"""Envelope removal and speckle-width estimation (Experiment 1 definitions)."""

from __future__ import annotations

import cv2
import numpy as np


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with sigma in pixels (OpenCV auto kernel)."""
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    return cv2.GaussianBlur(
        image.astype(np.float64),
        (0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT101,
    )


def local_ratio_detail(image: np.ndarray, sigma: float, eps_env: float) -> np.ndarray:
    """Local envelope normalization: I / (blur(I)+eps) - 1."""
    low = gaussian_blur(image, sigma)
    return image.astype(np.float64) / (low + float(eps_env)) - 1.0


def highpass_detail(image: np.ndarray, sigma: float) -> np.ndarray:
    """High-pass residual: I - blur(I)."""
    low = gaussian_blur(image, sigma)
    return image.astype(np.float64) - low


def estimate_speckle_width(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Estimate speckle autocorrelation 1/e half-width in pixels (Experiment 1)."""
    img = image.astype(np.float64)
    remove_sigma = max(16.0, min(img.shape) / 16.0)
    img = img - gaussian_blur(img, remove_sigma)

    if mask is not None:
        sel = mask.astype(bool)
        if int(np.count_nonzero(sel)) < 64:
            return 8.0
        ys, xs = np.where(sel)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        patch = img[y0:y1, x0:x1].copy()
        m = sel[y0:y1, x0:x1]
        mu = float(patch[m].mean())
        patch = patch - mu
        patch[~m] = 0.0
    else:
        patch = img - float(img.mean())

    max_side = 256
    h, w = patch.shape
    scale = max(h, w) / max_side
    if scale > 1.0:
        new_w = max(32, int(round(w / scale)))
        new_h = max(32, int(round(h / scale)))
        patch = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_AREA)
        scale_back = scale
    else:
        scale_back = 1.0

    f = np.fft.rfft2(patch)
    ac = np.fft.irfft2(np.abs(f) ** 2, s=patch.shape)
    ac = np.fft.fftshift(ac)
    cy, cx = ac.shape[0] // 2, ac.shape[1] // 2
    center = float(ac[cy, cx])
    if center <= 0:
        return 8.0
    line = ac[cy, cx:] / center
    target = 1.0 / np.e
    idx = np.where(line <= target)[0]
    if idx.size == 0:
        width = float(line.size - 1)
    else:
        width = float(idx[0])
    return float(np.clip(width * scale_back, 2.0, 40.0))


def speckle_contrast(image: np.ndarray, mask: np.ndarray | None = None, signed: bool = False) -> float:
    """Speckle contrast on mask pixels (Experiment 1 convention)."""
    img = image.astype(np.float64)
    if mask is None:
        sel = img.ravel()
    else:
        sel = img[mask.astype(bool)]
    if sel.size < 2:
        return float("nan")
    if signed:
        mu = float(np.mean(np.abs(sel)))
    else:
        mu = float(np.mean(sel))
    sd = float(np.std(sel))
    if mu <= 1e-12:
        return float("nan")
    return sd / mu
