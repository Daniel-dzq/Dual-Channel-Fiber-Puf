"""Length operating-window selection (never assumes 9 cm is optimal)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SelectionDecision:
    selected_length_cm: int | None
    decision_type: str  # unique_optimum | broad_plateau | pareto_knee | no_detectable_effect
    criterion: str
    bootstrap_selection_probability: float | None
    competing_lengths: list[int] = field(default_factory=list)
    plateau_lengths: list[int] = field(default_factory=list)
    red_guardrail_status: str = "not_used"
    qc_status: str = "ok"
    evidence_summary: str = ""
    limitations: str = ""
    frozen_for_future_experiments: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_length(
    length_metrics: pd.DataFrame,
    selection_bootstrap: pd.DataFrame,
    *,
    criterion: str = "maximin_robust_gap",
    never_force_unique_optimum: bool = True,
    prefer_shorter_length_inside_plateau: bool = True,
    selection_prob_unique_min: float = 0.60,
    red_metrics: pd.DataFrame | None = None,
    use_red_as_guardrail: bool = True,
    failed_lengths: list[int] | None = None,
) -> SelectionDecision:
    failed = set(failed_lengths or [])
    df = length_metrics.copy()
    if "length_cm" not in df.columns or df.empty:
        return SelectionDecision(
            selected_length_cm=None,
            decision_type="no_detectable_effect",
            criterion=criterion,
            bootstrap_selection_probability=None,
            evidence_summary="No length metrics available.",
            limitations="Empty metrics table.",
        )
    df = df.loc[~df["length_cm"].isin(failed)].copy()
    if df.empty:
        return SelectionDecision(
            None,
            "no_detectable_effect",
            criterion,
            None,
            qc_status="all_lengths_failed_qc",
            evidence_summary="All lengths failed hard QC.",
        )

    metric_col = "robust_gap_min"
    if metric_col not in df.columns:
        raise KeyError("length_metrics must contain robust_gap_min")

    # Point estimate ranking
    df = df.sort_values(["robust_gap_min", "length_cm"], ascending=[False, True])
    best_row = df.iloc[0]
    best_L = int(best_row["length_cm"])
    best_v = float(best_row["robust_gap_min"])

    # Bootstrap selection probabilities
    prob_map = {}
    if selection_bootstrap is not None and not selection_bootstrap.empty:
        for _, r in selection_bootstrap.iterrows():
            prob_map[int(r["length_cm"])] = float(r["selection_probability"])
    best_p = float(prob_map.get(best_L, np.nan))

    # Plateau: lengths whose values are within small absolute tolerance of best
    # and/or whose bootstrap CIs would overlap — use relative gap of metrics
    vals = {int(r.length_cm): float(r.robust_gap_min) for r in df.itertuples()}
    finite_vals = [v for v in vals.values() if np.isfinite(v)]
    spread = float(np.nanmax(finite_vals) - np.nanmin(finite_vals)) if finite_vals else 0.0

    plateau = []
    if np.isfinite(best_v):
        tol = max(0.02, 0.15 * abs(best_v)) if abs(best_v) > 1e-6 else 0.02
        plateau = sorted([L for L, v in vals.items() if np.isfinite(v) and abs(v - best_v) <= tol])

    red_status = "not_used"
    if use_red_as_guardrail and red_metrics is not None and not red_metrics.empty:
        # Hard fail if median S_R_intra for a length is NaN or extremely low
        for L in list(plateau) or [best_L]:
            sub = red_metrics.loc[red_metrics.length_cm == L, "S_R_intra"]
            if sub.empty or not np.isfinite(sub.median()) or float(sub.median()) < 0.2:
                red_status = f"guardrail_flag_length_{L}"
                if L in plateau:
                    plateau = [x for x in plateau if x != L]
        if red_status == "not_used":
            red_status = "passed"

    notes: list[str] = []
    # Case D: no detectable effect
    if spread < 0.03 or (len(plateau) >= max(3, len(vals) - 1) and never_force_unique_optimum):
        chosen = int(min(plateau)) if plateau and prefer_shorter_length_inside_plateau else best_L
        return SelectionDecision(
            selected_length_cm=chosen,
            decision_type="no_detectable_effect",
            criterion=criterion,
            bootstrap_selection_probability=prob_map.get(chosen),
            competing_lengths=sorted(vals.keys()),
            plateau_lengths=plateau,
            red_guardrail_status=red_status,
            evidence_summary=(
                f"Length dependence is weak (spread={spread:.4f}). "
                f"Engineering selection of {chosen} cm "
                f"{'(shortest in comparable set)' if prefer_shorter_length_inside_plateau else ''}."
            ),
            limitations="Only five fibers per length; do not overstate significance.",
            frozen_for_future_experiments=True,
            notes=notes,
        )

    # Case A: unique optimum
    if (
        np.isfinite(best_p)
        and best_p >= selection_prob_unique_min
        and len(plateau) <= 1
        and not never_force_unique_optimum
    ) or (
        np.isfinite(best_p)
        and best_p >= selection_prob_unique_min
        and len(plateau) == 1
    ):
        return SelectionDecision(
            selected_length_cm=best_L,
            decision_type="unique_optimum",
            criterion=criterion,
            bootstrap_selection_probability=best_p,
            competing_lengths=[L for L in vals if L != best_L],
            plateau_lengths=[best_L],
            red_guardrail_status=red_status,
            evidence_summary=(
                f"Under maximin robust-gap, {best_L} cm ranks first "
                f"(robust_gap_min={best_v:.4f}) with bootstrap selection probability {best_p:.3f}."
            ),
            limitations="Cluster bootstrap with five devices; CIs remain wide.",
            frozen_for_future_experiments=True,
            notes=notes,
        )

    # Case B: broad plateau
    if len(plateau) >= 2 and never_force_unique_optimum:
        chosen = int(min(plateau)) if prefer_shorter_length_inside_plateau else best_L
        return SelectionDecision(
            selected_length_cm=chosen,
            decision_type="broad_plateau",
            criterion=criterion,
            bootstrap_selection_probability=prob_map.get(chosen),
            competing_lengths=plateau,
            plateau_lengths=plateau,
            red_guardrail_status=red_status,
            evidence_summary=(
                f"{min(plateau)}–{max(plateau)} cm form a statistically comparable operating plateau. "
                f"Selected {chosen} cm from the plateau for compactness / lower loss "
                f"without claiming significant superiority over other plateau lengths."
            ),
            limitations="Plateau membership uses a practical tolerance on robust_gap_min.",
            frozen_for_future_experiments=True,
            notes=notes,
        )

    # Case C: Pareto knee (if complexity rises while robustness falls)
    if "n_eff_median" in df.columns:
        ordered = df.sort_values("length_cm")
        n_eff = ordered["n_eff_median"].to_numpy(float)
        rob = ordered["robust_gap_min"].to_numpy(float)
        Ls = ordered["length_cm"].to_numpy(int)
        if np.all(np.diff(n_eff[np.isfinite(n_eff)]) >= -1e-9) and np.any(np.diff(rob[np.isfinite(rob)]) < 0):
            # knee: max of normalized n_eff gain minus robustness loss
            # pick length maximizing robust_gap among increasing complexity
            knee = int(Ls[int(np.nanargmax(rob))])
            return SelectionDecision(
                selected_length_cm=knee,
                decision_type="pareto_knee",
                criterion=criterion,
                bootstrap_selection_probability=prob_map.get(knee),
                competing_lengths=list(map(int, Ls)),
                plateau_lengths=[],
                red_guardrail_status=red_status,
                evidence_summary=(
                    f"Complexity increases with length while robustness declines; "
                    f"{knee} cm identified as the Pareto knee under maximin robust-gap."
                ),
                limitations="Knee identification is descriptive with n=5 devices/length.",
                frozen_for_future_experiments=True,
            )

    # Fallback: treat as plateau/engineering choice
    chosen = int(min(plateau)) if plateau and prefer_shorter_length_inside_plateau else best_L
    return SelectionDecision(
        selected_length_cm=chosen,
        decision_type="broad_plateau" if len(plateau) >= 2 else "unique_optimum",
        criterion=criterion,
        bootstrap_selection_probability=prob_map.get(chosen),
        competing_lengths=plateau or [best_L],
        plateau_lengths=plateau or [best_L],
        red_guardrail_status=red_status,
        evidence_summary=f"Selected {chosen} cm by maximin robust-gap ranking (best={best_L}, value={best_v:.4f}).",
        limitations="Fallback path; inspect bootstrap probabilities carefully.",
        frozen_for_future_experiments=True,
        notes=notes,
    )


# --- synthetic scenario helpers for tests ---

def decision_from_synthetic_gaps(
    gaps: dict[int, float],
    probs: dict[int, float] | None = None,
    **kwargs,
) -> SelectionDecision:
    rows = [{"length_cm": L, "robust_gap_min": g} for L, g in gaps.items()]
    lm = pd.DataFrame(rows)
    if probs is None:
        # deterministic winner
        best = max(gaps, key=gaps.get)
        probs = {L: (1.0 if L == best else 0.0) for L in gaps}
    sb = pd.DataFrame(
        [{"length_cm": L, "n_selected": 0, "selection_probability": probs.get(L, 0.0)} for L in gaps]
    )
    return select_length(lm, sb, **kwargs)
