"""Fiber-level 2×2 paired contrasts, cluster bootstrap, optional mixed model."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from puf_common.statistics import cluster_bootstrap_mean

logger = logging.getLogger(__name__)

CONDITIONS = [
    (532, "axial"),
    (532, "lateral"),
    (650, "axial"),
    (650, "lateral"),
]



def aggregate_fiber_condition(per_video: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    """Median (primary) + mean/std over M0–M2 for each fiber × wavelength × geometry."""
    keys = ["fiber_id", "wavelength_nm", "excitation_geometry"]
    rows = []
    for (fiber, wl, geom), g in per_video.groupby(keys, sort=True):
        row: dict[str, Any] = {
            "fiber_id": int(fiber),
            "wavelength_nm": int(wl),
            "excitation_geometry": str(geom),
            "n_repeats": int(len(g)),
        }
        for c in metric_cols:
            if c not in g.columns:
                continue
            v = pd.to_numeric(g[c], errors="coerce")
            row[f"{c}_median"] = float(v.median())
            row[f"{c}_mean"] = float(v.mean())
            row[f"{c}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _y(table: pd.DataFrame, fiber: int, wl: int, geom: str, col: str) -> float:
    m = (
        (table["fiber_id"] == fiber)
        & (table["wavelength_nm"] == wl)
        & (table["excitation_geometry"] == geom)
    )
    sub = table.loc[m, col]
    if sub.empty:
        return float("nan")
    return float(sub.iloc[0])


def fiber_contrasts(fiber_table: pd.DataFrame, metric_median_col: str) -> pd.DataFrame:
    """Per-fiber Δ_geometry, Δ_wavelength, interaction and simple effects."""
    rows = []
    for fiber in sorted(fiber_table["fiber_id"].unique()):
        y_532_a = _y(fiber_table, fiber, 532, "axial", metric_median_col)
        y_532_l = _y(fiber_table, fiber, 532, "lateral", metric_median_col)
        y_650_a = _y(fiber_table, fiber, 650, "axial", metric_median_col)
        y_650_l = _y(fiber_table, fiber, 650, "lateral", metric_median_col)
        d_geom = 0.5 * ((y_532_l - y_532_a) + (y_650_l - y_650_a))
        d_wl = 0.5 * ((y_532_a - y_650_a) + (y_532_l - y_650_l))
        d_int = (y_532_l - y_532_a) - (y_650_l - y_650_a)
        rows.append(
            {
                "fiber_id": int(fiber),
                "metric": metric_median_col,
                "y_532_axial": y_532_a,
                "y_532_lateral": y_532_l,
                "y_650_axial": y_650_a,
                "y_650_lateral": y_650_l,
                "Delta_geometry": d_geom,
                "Delta_wavelength": d_wl,
                "Delta_interaction": d_int,
                "geometry_effect_at_532": y_532_l - y_532_a,
                "geometry_effect_at_650": y_650_l - y_650_a,
                "wavelength_effect_under_axial": y_532_a - y_650_a,
                "wavelength_effect_under_lateral": y_532_l - y_650_l,
            }
        )
    return pd.DataFrame(rows)


def summarize_contrasts(contrasts: pd.DataFrame, *, effect_col: str) -> dict[str, Any]:
    d = pd.to_numeric(contrasts[effect_col], errors="coerce").dropna()
    if d.empty:
        return {
            "n_fibers": 0,
            "median_effect": float("nan"),
            "mean_effect": float("nan"),
            "n_positive": 0,
            "n_negative": 0,
        }
    return {
        "n_fibers": int(d.size),
        "median_effect": float(d.median()),
        "mean_effect": float(d.mean()),
        "n_positive": int((d > 0).sum()),
        "n_negative": int((d < 0).sum()),
    }


def standardized_paired_effect(deltas: np.ndarray) -> float:
    d = np.asarray(deltas, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return float("nan")
    sd = float(np.std(d, ddof=1))
    if sd < 1e-12:
        return float("nan")
    return float(np.mean(d) / sd)


def cluster_bootstrap_effect(
    contrasts: pd.DataFrame,
    effect_col: str,
    *,
    n_iterations: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap CI by resampling fibers (clusters)."""
    by = {
        str(int(r.fiber_id)): [float(getattr(r, effect_col))]
        for r in contrasts.itertuples(index=False)
        if np.isfinite(getattr(r, effect_col))
    }
    return cluster_bootstrap_mean(by, n_iterations=n_iterations, seed=seed)


def complexity_oriented_acf(acf_fwhm: float) -> float:
    if not np.isfinite(acf_fwhm) or acf_fwhm <= 0:
        return float("nan")
    return float(-np.log(acf_fwhm))


def try_mixed_model(per_video: pd.DataFrame, metric: str) -> dict[str, Any]:
    """Optional LMM: metric_z ~ C(wavelength)*C(geometry) + C(repeat) + (1|fiber)."""
    try:
        import statsmodels.formula.api as smf  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {"status": "SKIPPED", "reason": f"statsmodels unavailable: {exc}"}

    df = per_video.copy()
    if metric not in df.columns:
        return {"status": "SKIPPED", "reason": f"missing metric {metric}"}
    y = pd.to_numeric(df[metric], errors="coerce")
    mu, sd = float(y.mean()), float(y.std(ddof=1))
    if not np.isfinite(sd) or sd < 1e-12:
        return {"status": "FAILED", "reason": "degenerate metric variance"}
    df["metric_z"] = (y - mu) / sd
    df["wavelength"] = df["wavelength_nm"].astype(str)
    df["coupling"] = df["excitation_geometry"].astype(str)
    df["repeat_id"] = df["repeat_id"].astype(str)
    df["fiber_id"] = df["fiber_id"].astype(str)
    try:
        model = smf.mixedlm(
            "metric_z ~ C(wavelength) * C(coupling) + C(repeat_id)",
            df.dropna(subset=["metric_z"]),
            groups=df.dropna(subset=["metric_z"])["fiber_id"],
        )
        res = model.fit(reml=True, method="lbfgs")
        return {
            "status": "OK",
            "metric": metric,
            "summary": str(res.summary()),
            "params": {k: float(v) for k, v in res.params.items()},
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Mixed model failed for %s: %s", metric, exc)
        return {"status": "FAILED", "reason": str(exc), "metric": metric}


def build_factorial_tables(
    per_video: pd.DataFrame,
    *,
    primary_metrics: list[str],
    bootstrap_iterations: int = 10000,
    bootstrap_seed: int = 42,
    mixed_model_enabled: bool = True,
) -> dict[str, Any]:
    # Prefer median columns on fiber table
    fiber = aggregate_fiber_condition(per_video, primary_metrics)
    contrast_frames = []
    summary_rows = []
    boot_rows = []
    for m in primary_metrics:
        med_col = f"{m}_median"
        if med_col not in fiber.columns:
            continue
        # For ACF, also compute complexity-oriented contrasts
        c = fiber_contrasts(fiber, med_col)
        contrast_frames.append(c)
        for effect in ("Delta_geometry", "Delta_wavelength", "Delta_interaction"):
            sm = summarize_contrasts(c, effect_col=effect)
            std_e = standardized_paired_effect(c[effect].to_numpy())
            boot = cluster_bootstrap_effect(
                c, effect, n_iterations=bootstrap_iterations, seed=bootstrap_seed
            )
            summary_rows.append(
                {
                    "metric": m,
                    "effect": effect,
                    **sm,
                    "standardized_paired_effect": std_e,
                    "bootstrap_mean": boot["mean"],
                    "bootstrap_ci_low": boot["ci_low"],
                    "bootstrap_ci_high": boot["ci_high"],
                }
            )
            boot_rows.append(
                {
                    "metric": m,
                    "effect": effect,
                    **boot,
                }
            )
        # geometry vs wavelength ratio on standardized effects
        g = standardized_paired_effect(c["Delta_geometry"].to_numpy())
        w = standardized_paired_effect(c["Delta_wavelength"].to_numpy())
        summary_rows.append(
            {
                "metric": m,
                "effect": "geometry_to_wavelength_effect_ratio",
                "standardized_geometry_effect": g,
                "standardized_wavelength_effect": w,
                "geometry_to_wavelength_effect_ratio": (g / w)
                if np.isfinite(g) and np.isfinite(w) and abs(w) > 1e-12
                else float("nan"),
                "n_fibers_consistent_geometry_neg_acf_or_pos_psd": int(
                    (c["Delta_geometry"] < 0).sum()
                    if "acf" in m
                    else (c["Delta_geometry"] > 0).sum()
                ),
            }
        )

    mixed = []
    if mixed_model_enabled:
        for m in primary_metrics:
            mixed.append(try_mixed_model(per_video, m))

    return {
        "per_fiber_condition": fiber,
        "contrasts": pd.concat(contrast_frames, ignore_index=True) if contrast_frames else pd.DataFrame(),
        "contrast_summary": pd.DataFrame(summary_rows),
        "bootstrap_intervals": pd.DataFrame(boot_rows),
        "mixed_model_results": mixed,
    }
