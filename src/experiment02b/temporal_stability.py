"""Within-video and across-repeat stability via zero-mean NCC (puf_common)."""

from __future__ import annotations

from typing import Any

import numpy as np

from puf_common.ncc import zero_mean_ncc


def frame_to_template_ncc(
    frames_detail: list[np.ndarray] | np.ndarray,
    template: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    arr = [np.asarray(f, dtype=np.float64) for f in frames_detail]
    scores = [zero_mean_ncc(f, template, mask=mask) for f in arr]
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0:
        return {
            "frame_to_template_ncc_median": float("nan"),
            "frame_to_template_ncc_q05": float("nan"),
            "frame_to_template_ncc_q95": float("nan"),
            "frame_to_template_ncc_iqr": float("nan"),
            "frame_to_template_ncc_min": float("nan"),
        }
    q05, q25, q50, q75, q95 = np.percentile(s, [5, 25, 50, 75, 95])
    return {
        "frame_to_template_ncc_median": float(q50),
        "frame_to_template_ncc_q05": float(q05),
        "frame_to_template_ncc_q95": float(q95),
        "frame_to_template_ncc_iqr": float(q75 - q25),
        "frame_to_template_ncc_min": float(np.min(s)),
    }


def block_to_block_ncc(
    block_templates: list[np.ndarray],
    *,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    scores = []
    for i in range(len(block_templates)):
        for j in range(i + 1, len(block_templates)):
            scores.append(zero_mean_ncc(block_templates[i], block_templates[j], mask=mask))
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0:
        return {
            "block_to_block_ncc_median": float("nan"),
            "block_to_block_ncc_min": float("nan"),
        }
    return {
        "block_to_block_ncc_median": float(np.median(s)),
        "block_to_block_ncc_min": float(np.min(s)),
    }


def repeat_to_repeat_ncc(
    repeat_templates: dict[int, np.ndarray],
    *,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    ids = sorted(repeat_templates.keys())
    scores = []
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            scores.append(zero_mean_ncc(repeat_templates[a], repeat_templates[b], mask=mask))
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0:
        return {
            "repeat_to_repeat_ncc_median": float("nan"),
            "repeat_to_repeat_ncc_min": float("nan"),
            "n_repeat_pairs": 0,
        }
    return {
        "repeat_to_repeat_ncc_median": float(np.median(s)),
        "repeat_to_repeat_ncc_min": float(np.min(s)),
        "n_repeat_pairs": int(s.size),
    }


def estimate_shift_phase_correlation(
    a: np.ndarray, b: np.ndarray
) -> dict[str, float]:
    """Optional QC shift estimate (not used to warp primary analysis)."""
    import cv2

    a32 = np.asarray(a, dtype=np.float32)
    b32 = np.asarray(b, dtype=np.float32)
    (sx, sy), _ = cv2.phaseCorrelate(a32, b32)
    return {
        "estimated_shift_x": float(sx),
        "estimated_shift_y": float(sy),
        "estimated_shift_magnitude": float(np.hypot(sx, sy)),
    }


def repeat_cv(values: list[float] | np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float("nan")
    mu = float(np.mean(v))
    if abs(mu) < 1e-12:
        return float("nan")
    return float(np.std(v, ddof=1) / abs(mu))
