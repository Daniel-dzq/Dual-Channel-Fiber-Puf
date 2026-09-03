"""Feature construction: local-ratio detail and shared-common residual."""

from __future__ import annotations

import numpy as np

from puf_common.envelope import local_ratio_detail


def to_detail(image: np.ndarray, sigma: float, eps: float) -> np.ndarray:
    return local_ratio_detail(image, sigma=sigma, eps_env=eps)


def subtract_common(image: np.ndarray, common: np.ndarray) -> np.ndarray:
    return image.astype(np.float64) - common.astype(np.float64)


def mean_template(images: list[np.ndarray]) -> np.ndarray:
    """Experiment 1 representative template = mean over windows/blocks."""
    return np.mean(np.stack([np.asarray(x, dtype=np.float64) for x in images], axis=0), axis=0)


def median_template(images: list[np.ndarray]) -> np.ndarray:
    """Pixel-wise median aggregation (Experiment 2 block aggregation option)."""
    return np.median(np.stack([np.asarray(x, dtype=np.float64) for x in images], axis=0), axis=0)


def build_common_from_templates(templates: list[np.ndarray]) -> np.ndarray:
    """Shared common = mean of challenge templates (Experiment 1 detail_cm)."""
    return mean_template(templates)
