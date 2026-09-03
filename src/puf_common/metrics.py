"""Discriminability helpers (Experiment 1 definitions)."""

from __future__ import annotations

import numpy as np


def d_prime(genuine: np.ndarray, impostor: np.ndarray) -> float:
    g = np.asarray(genuine, dtype=np.float64)
    i = np.asarray(impostor, dtype=np.float64)
    if g.size == 0 or i.size == 0:
        return float("nan")
    var = 0.5 * (float(np.var(g)) + float(np.var(i)))
    if var <= 0.0:
        return float("inf") if float(np.mean(g)) > float(np.mean(i)) else 0.0
    return float((np.mean(g) - np.mean(i)) / np.sqrt(var))


def equal_error_rate(genuine: np.ndarray, impostor: np.ndarray, n_thresholds: int = 501) -> float:
    """EER for scores where higher means more similar (NCC)."""
    g = np.asarray(genuine, dtype=np.float64)
    i = np.asarray(impostor, dtype=np.float64)
    if g.size == 0 or i.size == 0:
        return float("nan")
    lo = float(min(g.min(), i.min()))
    hi = float(max(g.max(), i.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0
    thresholds = np.linspace(lo, hi, n_thresholds)
    best_eer = 1.0
    best_gap = 1e9
    best_thr = float(lo)
    for t in thresholds:
        frr = float(np.mean(g < t))
        far = float(np.mean(i >= t))
        gap = abs(frr - far)
        eer = 0.5 * (frr + far)
        if gap < best_gap or (gap == best_gap and eer < best_eer):
            best_gap = gap
            best_eer = eer
            best_thr = float(t)
    return float(best_eer)


def equal_error_rate_with_threshold(
    genuine: np.ndarray, impostor: np.ndarray, n_thresholds: int = 501
) -> tuple[float, float]:
    """Return (EER, threshold) using the Experiment 1 EER sweep."""
    g = np.asarray(genuine, dtype=np.float64)
    i = np.asarray(impostor, dtype=np.float64)
    if g.size == 0 or i.size == 0:
        return float("nan"), float("nan")
    lo = float(min(g.min(), i.min()))
    hi = float(max(g.max(), i.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0, lo
    thresholds = np.linspace(lo, hi, n_thresholds)
    best_eer = 1.0
    best_gap = 1e9
    best_thr = float(lo)
    for t in thresholds:
        frr = float(np.mean(g < t))
        far = float(np.mean(i >= t))
        gap = abs(frr - far)
        eer = 0.5 * (frr + far)
        if gap < best_gap or (gap == best_gap and eer < best_eer):
            best_gap = gap
            best_eer = eer
            best_thr = float(t)
    return float(best_eer), best_thr


def auc_roc(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """AUC treating higher NCC as genuine (Mann-Whitney form)."""
    g = np.asarray(genuine, dtype=np.float64)
    i = np.asarray(impostor, dtype=np.float64)
    if g.size == 0 or i.size == 0:
        return float("nan")
    total = 0.0
    for gv in g:
        total += float(np.sum(i < gv) + 0.5 * np.sum(i == gv))
    return float(total / (g.size * i.size))


def robust_gap(positive: np.ndarray, negative: np.ndarray) -> float:
    """Experiment 1 robust gap: Q05(positive) - Q95(negative)."""
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    return float(np.percentile(pos, 5) - np.percentile(neg, 95))


def margin(positive: np.ndarray, negative: np.ndarray) -> float:
    """Median(positive) - median(negative)."""
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    return float(np.median(pos) - np.median(neg))
