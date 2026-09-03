"""Statistics helpers for device-level inference."""

from __future__ import annotations

import numpy as np


def paired_ttest(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Paired t-test on device-level pairs. Returns (statistic, pvalue)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size != b.size or a.size < 2:
        return float("nan"), float("nan")
    d = a - b
    n = d.size
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    if sd <= 0:
        return float("inf") if mean != 0 else 0.0, 0.0 if mean != 0 else 1.0
    t = mean / (sd / np.sqrt(n))
    # Two-sided p via regularized incomplete beta (no scipy required)
    from math import lgamma

    nu = n - 1
    x = nu / (nu + t * t)
    # incomplete beta approximation for Student-t CDF
    try:
        from scipy.stats import t as student_t

        p = float(2 * student_t.sf(abs(t), nu))
    except Exception:
        p = float("nan")
    return float(t), p


def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    try:
        from scipy.stats import wilcoxon

        res = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return float("nan"), float("nan")


def cluster_bootstrap_mean(
    values_by_cluster: dict[str, list[float]],
    *,
    n_iterations: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap mean CI by resampling clusters (devices)."""
    rng = np.random.default_rng(seed)
    keys = list(values_by_cluster.keys())
    if not keys:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    obs = [float(np.mean(values_by_cluster[k])) for k in keys if values_by_cluster[k]]
    if not obs:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    means = []
    n = len(obs)
    for _ in range(int(n_iterations)):
        idx = rng.integers(0, n, size=n)
        means.append(float(np.mean([obs[i] for i in idx])))
    arr = np.asarray(means, dtype=np.float64)
    return {
        "mean": float(np.mean(obs)),
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
    }
