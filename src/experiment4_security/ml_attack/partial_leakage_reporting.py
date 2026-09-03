"""Aggregation, device-cluster bootstrap, model-level summary, figure-data
export, and rule-based narrative selection for the partial-leakage
supplementary experiment (Tracks C-PL / D-PL).

Statistical unit discipline (Section 8 of the spec): hidden challenges,
split repetitions, and mechanical states are NOT independent biological/
device samples. The statistical unit is DEVICE. Aggregation order:

  1. Collapse the 5 repetitions to their median, per
     (device, source_state, target_state, leak_size, attack_method).
  2. Any confidence interval over the resulting 48 device x state points
     must use device-cluster bootstrap (resample devices with replacement,
     keep every resampled device's full set of states, 10000 iters, seed 42).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

REP_COLLAPSE_KEYS = ["device_id", "source_state", "target_state", "leak_size", "attack_method"]

NARRATIVE_A = (
    "Full-database clone risk primarily relies on already-leaked templates; "
    "limited CRP leakage did not show effective generalization to hidden "
    "registered challenges. (Complete database-clone risk depends on leaked "
    "templates; limited CRP leakage shows no effective generalization to "
    "hidden registered challenges.)"
)
NARRATIVE_B = (
    "Partial registered-database leakage supports hidden-challenge modeling "
    "within the source state, but mechanical reconfiguration revokes that "
    "model (partial-leakage model valid only in source state)."
)
NARRATIVE_C = (
    "The partial-leakage model shows cross-state transfer and must be "
    "flagged as a credential-reconfiguration risk; this cannot be hidden."
)
NARRATIVE_D = (
    "Results show model- and leak-size-dependent behavior; no single unified "
    "security conclusion is given (inconclusive)."
)


def collapse_repetitions(transfer_summary: pd.DataFrame) -> pd.DataFrame:
    """Median across the 5 split repetitions, per device/state(s)/leak/method."""
    if transfer_summary.empty:
        return transfer_summary
    agg_cols = [
        c
        for c in (
            "median_S_A_partial",
            "q05_S_A_partial",
            "q95_S_A_partial",
            "median_S_G_hidden",
            "RG_A_partial",
            "AUC_A_partial",
            "EER_A_partial",
            "Top1_hidden",
            "Top5_hidden",
            "MRR_hidden",
            "chance_top1",
            "chance_top5",
            "normalized_retrieval_lift",
            "median_NMSE",
            "median_residual_ncc",
            "gain_over_mean",
            "n_leaked",
            "n_hidden",
            "effective_pca_dim",
        )
        if c in transfer_summary.columns
    ]
    keep_first = [c for c in ("is_diagonal",) if c in transfer_summary.columns]
    grouped = transfer_summary.groupby(REP_COLLAPSE_KEYS, as_index=False)
    out = grouped[list(agg_cols)].median(numeric_only=True)
    if keep_first:
        first = grouped[keep_first].first()
        out = out.merge(first, on=REP_COLLAPSE_KEYS)
    return out


def device_cluster_bootstrap_ci(
    values_by_device: dict[str, np.ndarray],
    *,
    statistic: Callable[[np.ndarray], float] = np.median,
    n_iter: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    devices = [d for d, v in values_by_device.items() if v.size > 0]
    if not devices:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(devices)
    boot = np.empty(n_iter, dtype=np.float64)
    arrays = [values_by_device[d] for d in devices]
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        pooled = np.concatenate([arrays[j] for j in idx])
        boot[i] = statistic(pooled)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def model_level_summary(
    collapsed: pd.DataFrame, *, bootstrap_iterations: int = 10000, bootstrap_seed: int = 42
) -> pd.DataFrame:
    """One row per (attack_method, leak_size): diag vs off-diag aggregates +
    device-cluster bootstrap CI on the diagonal median S_A_partial."""
    rows: list[dict[str, Any]] = []
    if collapsed.empty:
        return pd.DataFrame(rows)
    for (method, leak_size), grp in collapsed.groupby(["attack_method", "leak_size"]):
        diag = grp[grp["is_diagonal"]]
        off = grp[~grp["is_diagonal"]]
        diag_median = float(diag["median_S_A_partial"].median()) if not diag.empty else float("nan")
        off_median = float(off["median_S_A_partial"].median()) if not off.empty else float("nan")
        by_device = {
            d: sub["median_S_A_partial"].to_numpy(dtype=np.float64) for d, sub in diag.groupby("device_id")
        }
        ci = device_cluster_bootstrap_ci(
            by_device, statistic=np.median, n_iter=bootstrap_iterations, seed=bootstrap_seed
        )
        rows.append(
            {
                "attack_method": method,
                "leak_size": int(leak_size),
                "n_device_state_diag_points": int(len(diag)),
                "n_device_state_off_diag_points": int(len(off)),
                "diagonal_median_S_A": diag_median,
                "off_diagonal_median_S_A": off_median,
                "delta_partial_same_to_cross": (
                    diag_median - off_median if np.isfinite(diag_median) and np.isfinite(off_median) else float("nan")
                ),
                "diagonal_Top1": float(diag["Top1_hidden"].median()) if not diag.empty else float("nan"),
                "off_diagonal_Top1": float(off["Top1_hidden"].median()) if not off.empty else float("nan"),
                "diagonal_MRR": float(diag["MRR_hidden"].median()) if not diag.empty else float("nan"),
                "off_diagonal_MRR": float(off["MRR_hidden"].median()) if not off.empty else float("nan"),
                "diagonal_chance_top1": float(diag["chance_top1"].median()) if not diag.empty else float("nan"),
                "device_cluster_bootstrap_CI_lo": ci[0],
                "device_cluster_bootstrap_CI_hi": ci[1],
                "device_cluster_bootstrap_CI": f"[{ci[0]:.4f}, {ci[1]:.4f}]",
                "device_cluster_bootstrap_iterations": bootstrap_iterations,
                "device_cluster_bootstrap_seed": bootstrap_seed,
            }
        )
    return pd.DataFrame(rows).sort_values(["attack_method", "leak_size"]).reset_index(drop=True)


REGRESSION_METHODS = ("ridge_clone", "kernel_ridge_clone", "random_fourier_ridge_clone", "small_mlp_clone")
BASELINE_METHOD = "mean_leaked_response"

# Pre-registered thresholds (fixed before results were seen; mirror the
# Track-D full-database thresholds in track_d_clone_transfer.py so the two
# supplementary/primary experiments are read on a consistent scale).
GAIN_THRESHOLD_SAME_STATE = 0.15
CROSS_STATE_LOW_THRESHOLD = 0.4
CROSS_STATE_HIGH_THRESHOLD = 0.5


def select_narrative(model_summary: pd.DataFrame) -> dict[str, Any]:
    """Rule-based (not hand-picked) selection among Sections 14 A/B/C/D.

    Evaluated at leak_size=96 (largest, most favorable to the attacker)
    across the 4 non-baseline regression methods PL2-PL5.
    """
    if model_summary.empty:
        return {"narrative_key": "D", "narrative": NARRATIVE_D, "reason": "no partial-leakage results available"}

    at_max_leak = model_summary[model_summary["leak_size"] == model_summary["leak_size"].max()]
    baseline_row = at_max_leak[at_max_leak["attack_method"] == BASELINE_METHOD]
    baseline_diag = float(baseline_row["diagonal_median_S_A"].iloc[0]) if not baseline_row.empty else float("nan")

    per_method: list[dict[str, Any]] = []
    for method in REGRESSION_METHODS:
        row = at_max_leak[at_max_leak["attack_method"] == method]
        if row.empty:
            continue
        diag = float(row["diagonal_median_S_A"].iloc[0])
        off = float(row["off_diagonal_median_S_A"].iloc[0])
        gain = diag - baseline_diag if np.isfinite(baseline_diag) else float("nan")
        same_state_generalizes = np.isfinite(gain) and gain >= GAIN_THRESHOLD_SAME_STATE
        cross_state_collapses = np.isfinite(off) and off < CROSS_STATE_LOW_THRESHOLD
        cross_state_high = np.isfinite(off) and off >= CROSS_STATE_HIGH_THRESHOLD
        per_method.append(
            {
                "attack_method": method,
                "diagonal_median_S_A": diag,
                "off_diagonal_median_S_A": off,
                "gain_over_baseline": gain,
                "same_state_generalizes": bool(same_state_generalizes),
                "cross_state_collapses": bool(cross_state_collapses),
                "cross_state_high": bool(cross_state_high),
            }
        )

    if not per_method:
        return {"narrative_key": "D", "narrative": NARRATIVE_D, "reason": "no regression-method rows at max leak size"}

    n = len(per_method)
    n_generalize = sum(1 for m in per_method if m["same_state_generalizes"])
    n_cross_high = sum(1 for m in per_method if m["cross_state_high"])
    n_cross_collapse = sum(1 for m in per_method if m["cross_state_collapses"])

    if n_cross_high > 0:
        key, text = "C", NARRATIVE_C
    elif n_generalize == 0:
        key, text = "A", NARRATIVE_A
    elif n_generalize == n and n_cross_collapse == n:
        key, text = "B", NARRATIVE_B
    else:
        key, text = "D", NARRATIVE_D

    return {
        "narrative_key": key,
        "narrative": text,
        "evaluated_at_leak_size": int(at_max_leak["leak_size"].iloc[0]),
        "baseline_diagonal_median_S_A": baseline_diag,
        "per_method": per_method,
        "n_methods_generalize": n_generalize,
        "n_methods_cross_state_high": n_cross_high,
        "n_methods_cross_state_collapse": n_cross_collapse,
        "thresholds": {
            "gain_threshold_same_state": GAIN_THRESHOLD_SAME_STATE,
            "cross_state_low_threshold": CROSS_STATE_LOW_THRESHOLD,
            "cross_state_high_threshold": CROSS_STATE_HIGH_THRESHOLD,
        },
    }


# ---------------------------------------------------------------------------
# Figure data (Section 12) — CSVs only, no plotting here.
# ---------------------------------------------------------------------------

def panel_learning_curve(collapsed: pd.DataFrame) -> pd.DataFrame:
    """x=n_leaked (leak_size), y=median_S_A_partial per method, plus device x
    state points and the hidden-genuine reference band, for the caller to plot."""
    if collapsed.empty:
        return collapsed
    diag = collapsed[collapsed["is_diagonal"]].copy()
    return diag[
        [
            "device_id",
            "source_state",
            "leak_size",
            "attack_method",
            "median_S_A_partial",
            "median_S_G_hidden",
            "Top1_hidden",
            "chance_top1",
            "MRR_hidden",
        ]
    ].sort_values(["attack_method", "leak_size", "device_id", "source_state"]).reset_index(drop=True)


def panel_same_cross_dumbbell(model_summary: pd.DataFrame) -> pd.DataFrame:
    if model_summary.empty:
        return model_summary
    return model_summary[
        [
            "attack_method",
            "leak_size",
            "diagonal_median_S_A",
            "off_diagonal_median_S_A",
            "delta_partial_same_to_cross",
            "device_cluster_bootstrap_CI_lo",
            "device_cluster_bootstrap_CI_hi",
        ]
    ].sort_values(["attack_method", "leak_size"]).reset_index(drop=True)


def render_analysis_markdown(
    *,
    generated_at: str,
    devices: list[str],
    states: list[str],
    leak_sizes: list[int],
    repetitions: int,
    seeds: tuple[int, ...],
    n_banks: int,
    model_summary: pd.DataFrame,
    narrative: dict[str, Any],
    representative_condition: dict[str, Any],
) -> str:
    """PARTIAL_LEAKAGE_ANALYSIS.md content (Section 14, all 12 items). Every
    number below is read from ``model_summary`` / ``narrative`` (i.e. from the
    actual CSV/JSON results) — nothing here is hand-picked."""
    lines: list[str] = []
    lines.append("# Partial-Leakage Registered-Bank Generalization — Analysis")
    lines.append("")
    lines.append(f"Generated: {generated_at}")
    lines.append("")

    lines.append("## 1. Threat model")
    lines.append("")
    lines.append(
        "Server registers a fixed, public challenge bank C001-C128 per device x "
        "mechanical state. The attacker compromises a partial database record: "
        "for a subset of challenges (`leaked`), the attacker obtains the "
        "*challenge* pattern (`bitmap_32`, public regardless) AND the "
        "*server-stored, enrollment-processed* green template "
        "`T_full[d,s,c] = detail(A[d,s,c]) - common_full_A[d,s]`. This is a "
        "**conservative partial database-compromise model**: leaked records "
        "are the server-stored, enrollment-processed templates, so the "
        "server's full-bank common template is indirectly baked into what "
        "was stolen. This is NOT a raw-CRP-only observation model, NOT an "
        "unrestricted strong-PUF challenge space, and NOT a cryptographic "
        "security proof."
    )
    lines.append("")

    lines.append("## 2. Why hidden challenges are attacker-hidden but server-registered")
    lines.append("")
    lines.append(
        "The remaining challenges (`hidden`) are part of the SAME registered "
        "128-challenge bank -- the server has already enrolled them -- but no "
        "response record for them was part of the leak. The attacker may "
        "still see the hidden challenges' public bitmap (it is needed to even "
        "ask 'what would this registered challenge produce?'), but never a "
        "measured or stored response, and never at fit/PCA/normalization/"
        "model-selection time. This is 'attacker-held-out registered "
        "challenges', not 'new challenges' and not an 'unseen-challenge "
        "security' claim."
    )
    lines.append("")

    lines.append("## 3. Split design")
    lines.append("")
    lines.append(
        f"128 challenges partitioned into {n_banks} analysis banks of 8. For "
        f"each of {repetitions} repetition seeds ({', '.join(str(s) for s in seeds)}), "
        "every bank is independently shuffled with a seed derived from "
        "(repetition seed, bank_id) -- never from measured responses. Leak "
        f"sizes {leak_sizes} take the first 1/2/4/6 shuffled challenges per "
        "bank, so every leak size is bank-balanced (all banks represented) "
        "and the four leak sizes are strictly nested. leaked and hidden are "
        "always disjoint and always union to the full 128-challenge bank."
    )
    lines.append("")

    lines.append("## 4. Leakage-prevention measures")
    lines.append("")
    lines.append(
        "- Response normalization and PCA are fit ONLY on leaked templates "
        "(`fit_representation` receives a dict containing only leaked ids).\n"
        "- Regression fit functions intersect `features.keys() & templates.keys()`, "
        "so even a full 128-challenge feature dict cannot leak hidden ids into "
        "the fit set once `templates` is leaked-only.\n"
        "- PL1's nearest-neighbor search is restricted to `distance_matrix.loc[:, leaked_ids]`.\n"
        "- Track D-PL reuses the EXACT frozen predictor from Track C-PL (same "
        "object, same weights) — no target-state refit, no PCA/mean update, "
        "no reselection.\n"
        "- `exact_template_replay` is excluded from the PL zoo — hidden "
        "templates are undefined-to-the-attacker by construction.\n"
        "- Hidden ids never reach PCA, normalization or model fitting; PL1 "
        "searches leaked ids only; effective_pca_dim <= n_leaked - 1; Top-1 "
        "chance scales with n_hidden; the base-run Track A-D / red files are "
        "not rewritten."
    )
    lines.append("")

    lines.append("## 5. Learning curves per model (median S_A_partial vs leak size)")
    lines.append("")
    if not model_summary.empty:
        pivot = model_summary.pivot_table(index="attack_method", columns="leak_size", values="diagonal_median_S_A")
        lines.append(pivot.to_string())
    else:
        lines.append("(no results)")
    lines.append("")

    lines.append("## 6. Does performance increase with leak size?")
    lines.append("")
    if not model_summary.empty:
        for method, grp in model_summary.groupby("attack_method"):
            grp_sorted = grp.sort_values("leak_size")
            vals = grp_sorted["diagonal_median_S_A"].to_numpy()
            monotonic = bool(np.all(np.diff(vals) >= -1e-9)) if len(vals) > 1 else True
            lines.append(f"- `{method}`: diagonal_median_S_A by leak_size = {list(np.round(vals, 4))}; monotonic_nondecreasing={monotonic}")
    else:
        lines.append("(no results)")
    lines.append("")

    lines.append("## 7. Comparison against PL0/PL1 baselines")
    lines.append("")
    if narrative.get("per_method"):
        lines.append(
            f"At leak_size={narrative.get('evaluated_at_leak_size')}, baseline "
            f"PL0 diagonal_median_S_A = {narrative.get('baseline_diagonal_median_S_A'):.4f}."
        )
        for m in narrative["per_method"]:
            lines.append(
                f"- `{m['attack_method']}`: diagonal={m['diagonal_median_S_A']:.4f}, "
                f"gain_over_PL0={m['gain_over_baseline']:.4f}, "
                f"same_state_generalizes={m['same_state_generalizes']}"
            )
    else:
        lines.append("(insufficient data for baseline comparison)")
    lines.append("")

    lines.append("## 8. Same-state performance")
    lines.append("")
    if not model_summary.empty:
        lines.append(model_summary[["attack_method", "leak_size", "diagonal_median_S_A", "diagonal_Top1", "diagonal_MRR"]].to_string(index=False))
    else:
        lines.append("(no results)")
    lines.append("")

    lines.append("## 9. Cross-state performance")
    lines.append("")
    if not model_summary.empty:
        lines.append(model_summary[["attack_method", "leak_size", "off_diagonal_median_S_A", "off_diagonal_Top1", "off_diagonal_MRR", "delta_partial_same_to_cross"]].to_string(index=False))
    else:
        lines.append("(no results)")
    lines.append("")

    lines.append("## 10. Common-dominance")
    lines.append("")
    lines.append(
        "See `common_dominance_flag` in track_c_pl_partial_leakage_summary.csv "
        "and track_d_pl_partial_leakage_transfer_summary.csv (flagged "
        "`HIGH_NCC_DOMINATED_BY_COMMON_RESPONSE` when predictions are >0.95 "
        "NCC to the leaked-only PCA mean and the residual NCC is <0.2)."
    )
    lines.append("")

    lines.append("## 11. Relationship to full-database Track C / D")
    lines.append("")
    lines.append(
        "Track C/D (base formal run) train on ALL 128 registered Round-A "
        "templates and test the SAME 128 challenges' independent Round-B "
        "queries -- i.e. full registered-database leakage / software clone. "
        "Track C-PL/D-PL train on a LEAKED SUBSET only and test HIDDEN, "
        "never-fit registered challenges -- i.e. generalization within the "
        "registered bank under partial leakage. A high Track C score does "
        "not imply a high Track C-PL score, and vice versa."
    )
    lines.append("")

    lines.append("## 12. Allowed and prohibited conclusions")
    lines.append("")
    lines.append(f"**Selected narrative ({narrative.get('narrative_key')}):** {narrative.get('narrative')}")
    lines.append("")
    lines.append(
        "Allowed: hidden registered-challenge generalization observed / not "
        "observed; partial-leakage model valid only in source state; "
        "partial-leakage model transfers across reconfiguration; result "
        "inconclusive.\n\n"
        "Prohibited: absolute ASR; ML-proof; unseen-challenge-secure; "
        "reconfiguration-proof; ML-resistant; red-green interaction claims; "
        "commercial-infographic framing; cherry-picked representative "
        f"conditions (fixed pre-registered: leak_size={representative_condition.get('leak_size')}, "
        f"model={representative_condition.get('attack_method')})."
    )
    lines.append("")
    return "\n".join(lines)


def panel_heatmap(collapsed: pd.DataFrame, *, leak_size: int, attack_method: str) -> pd.DataFrame:
    """8x8 source_state x target_state heatmap for the pre-registered
    representative condition, cell = median S_A_partial across device x
    repetition (repetitions already collapsed by caller) x hidden challenge."""
    sub = collapsed[(collapsed["leak_size"] == leak_size) & (collapsed["attack_method"] == attack_method)]
    if sub.empty:
        return pd.DataFrame(columns=["source_state", "target_state", "median_S_A_partial", "n_device_points"])
    grp = sub.groupby(["source_state", "target_state"], as_index=False).agg(
        median_S_A_partial=("median_S_A_partial", "median"), n_device_points=("median_S_A_partial", "size")
    )
    return grp.sort_values(["source_state", "target_state"]).reset_index(drop=True)
