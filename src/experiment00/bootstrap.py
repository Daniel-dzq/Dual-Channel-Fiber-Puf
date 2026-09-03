"""Hierarchical device-then-challenge bootstrap for length metrics."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def _maximin_from_pairs(sub: pd.DataFrame) -> float:
    g = sub.loc[sub.score_type == "S_intra", "score"].to_numpy(float)
    ich = sub.loc[sub.score_type == "S_inter_challenge", "score"].to_numpy(float)
    idev = sub.loc[sub.score_type == "S_inter_device", "score"].to_numpy(float)
    if g.size == 0:
        return float("nan")
    gaps = []
    if ich.size:
        gaps.append(float(np.quantile(g, 0.05) - np.quantile(ich, 0.95)))
    if idev.size:
        gaps.append(float(np.quantile(g, 0.05) - np.quantile(idev, 0.95)))
    return float(np.nanmin(gaps)) if gaps else float("nan")


def hierarchical_bootstrap_length_metrics(
    pair_scores: pd.DataFrame,
    *,
    n_iterations: int = 5000,
    seed: int = 20260721,
    confidence_level: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (summary CI table, selection counts, pairwise probs)."""
    rng = np.random.default_rng(seed)
    lengths = sorted(int(x) for x in pair_scores["length_cm"].unique())
    alpha = 1.0 - confidence_level
    boot_rows = []
    winners: list[int] = []

    for it in tqdm(range(n_iterations), desc="Hierarchical bootstrap", unit="iter"):
        vals = {}

        for L in lengths:
            sub = pair_scores.loc[pair_scores.length_cm == L]
            fibers = sorted(sub["fiber_id"].dropna().unique().tolist())
            if not fibers:
                vals[L] = float("nan")
                continue
            # Layer 1: resample fibers
            samp_f = rng.choice(fibers, size=len(fibers), replace=True)
            pieces = []
            for f in samp_f:
                fsub = sub.loc[sub.fiber_id == f]
                chans = sorted(
                    set(fsub["challenge"].dropna().tolist())
                    | set(fsub.get("challenge_b", pd.Series(dtype=str)).dropna().tolist())
                )
                # Prefer challenges appearing in S_intra
                intra_ch = sorted(fsub.loc[fsub.score_type == "S_intra", "challenge"].unique())
                pool = intra_ch if intra_ch else sorted(fsub["challenge"].dropna().unique())
                if not pool:
                    continue
                samp_c = set(rng.choice(pool, size=len(pool), replace=True).tolist())
                # Keep rows involving resampled challenges for this fiber
                m = fsub["fiber_id"].eq(f) & (
                    fsub["challenge"].isin(samp_c)
                    | fsub.get("challenge_b", pd.Series(index=fsub.index)).isin(samp_c)
                )
                pieces.append(fsub.loc[m])
            if not pieces:
                vals[L] = float("nan")
                continue
            boot_sub = pd.concat(pieces, ignore_index=True)
            vals[L] = _maximin_from_pairs(boot_sub)
        boot_rows.append({"iteration": it, **{f"L{L}": vals[L] for L in lengths}})
        finite = {L: v for L, v in vals.items() if np.isfinite(v)}
        if finite:
            # Prefer shorter length on ties
            best_v = max(finite.values())
            cands = [L for L, v in finite.items() if abs(v - best_v) < 1e-15]
            winners.append(int(min(cands)))

    boot_df = pd.DataFrame(boot_rows)
    summary = []
    for L in lengths:
        col = f"L{L}"
        arr = boot_df[col].to_numpy(float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            summary.append(
                {
                    "length_cm": L,
                    "metric": "robust_gap_min",
                    "mean": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                }
            )
            continue
        summary.append(
            {
                "length_cm": L,
                "metric": "robust_gap_min",
                "mean": float(np.mean(arr)),
                "ci_low": float(np.quantile(arr, alpha / 2)),
                "ci_high": float(np.quantile(arr, 1 - alpha / 2)),
            }
        )
    summary_df = pd.DataFrame(summary)

    counts = Counter(winners)
    sel_df = pd.DataFrame(
        [
            {
                "length_cm": L,
                "n_selected": int(counts.get(L, 0)),
                "selection_probability": float(counts.get(L, 0) / max(len(winners), 1)),
            }
            for L in lengths
        ]
    )

    # Pairwise: P(Li > Lj)
    pair_rows = []
    for i, Li in enumerate(lengths):
        for Lj in lengths[i + 1 :]:
            a = boot_df[f"L{Li}"].to_numpy(float)
            b = boot_df[f"L{Lj}"].to_numpy(float)
            m = np.isfinite(a) & np.isfinite(b)
            if not m.any():
                p = float("nan")
            else:
                p = float(np.mean(a[m] > b[m]))
            pair_rows.append({"length_a_cm": Li, "length_b_cm": Lj, "p_a_gt_b": p})
            pair_rows.append({"length_a_cm": Lj, "length_b_cm": Li, "p_a_gt_b": 1.0 - p if p == p else p})
    pairwise_df = pd.DataFrame(pair_rows)
    return summary_df, sel_df, pairwise_df
