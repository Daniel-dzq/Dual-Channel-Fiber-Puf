"""Lightweight multiscale texture features (no heavy handcrafted banks)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage


def extract_texture_features(img: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    vals = img[mask]
    if vals.size == 0:
        return {
            "tex_grad_mean": float("nan"),
            "tex_grad_std": float("nan"),
            "tex_laplacian_energy": float("nan"),
            "tex_local_contrast_s3": float("nan"),
            "tex_local_contrast_s7": float("nan"),
            "tex_ms_energy_s1": float("nan"),
            "tex_ms_energy_s2": float("nan"),
            "tex_ms_energy_s3": float("nan"),
            "tex_qc": "empty_mask",
        }

    gx = ndimage.sobel(img, axis=1, mode="nearest")
    gy = ndimage.sobel(img, axis=0, mode="nearest")
    gmag = np.hypot(gx, gy)
    lap = ndimage.laplace(img, mode="nearest")

    def _ms(sigma: float) -> float:
        blur = ndimage.gaussian_filter(img, sigma=sigma)
        d = img - blur
        return float(np.mean(d[mask] ** 2))

    def _local_contrast(size: int) -> float:
        # local std / local mean
        mean = ndimage.uniform_filter(img, size=size, mode="nearest")
        mean_sq = ndimage.uniform_filter(img**2, size=size, mode="nearest")
        var = np.clip(mean_sq - mean**2, 0, None)
        std = np.sqrt(var)
        c = std / (np.abs(mean) + 1e-8)
        return float(np.mean(c[mask]))

    return {
        "tex_grad_mean": float(np.mean(gmag[mask])),
        "tex_grad_std": float(np.std(gmag[mask])),
        "tex_laplacian_energy": float(np.mean(lap[mask] ** 2)),
        "tex_local_contrast_s3": _local_contrast(3),
        "tex_local_contrast_s7": _local_contrast(7),
        "tex_ms_energy_s1": _ms(1.0),
        "tex_ms_energy_s2": _ms(2.0),
        "tex_ms_energy_s3": _ms(4.0),
        "tex_qc": "ok",
    }
