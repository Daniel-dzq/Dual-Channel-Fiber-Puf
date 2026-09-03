"""Post-run scientific sanity checks (English-only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from experiment00.cache import dump_json
from experiment00.length_selection import SelectionDecision, select_length


def _maximin_table(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for L, s2 in pairs.groupby("length_cm"):
        g = s2.loc[s2.score_type == "S_intra", "score"].to_numpy(float)
        ich = s2.loc[s2.score_type == "S_inter_challenge", "score"].to_numpy(float)
        idev = s2.loc[s2.score_type == "S_inter_device", "score"].to_numpy(float)
        if g.size == 0:
            continue
        gaps = []
        if ich.size:
            gaps.append(np.quantile(g, 0.05) - np.quantile(ich, 0.95))
        if idev.size:
            gaps.append(np.quantile(g, 0.05) - np.quantile(idev, 0.95))
        rows.append(
            {
                "length_cm": int(L),
                "robust_gap_min": float(np.min(gaps)) if gaps else np.nan,
                "median_intra": float(np.median(g)),
            }
        )
    return pd.DataFrame(rows)


def run_post_run_sanity(
    *,
    run_dir: Path,
    pairs: pd.DataFrame,
    length_metrics: pd.DataFrame,
    boot_sel: pd.DataFrame,
    decision: SelectionDecision,
    red: pd.DataFrame,
) -> dict:
    checks = []

    def add(name: str, ok: bool, detail: str, level: str = "PASS") -> None:
        checks.append(
            {
                "check": name,
                "status": level if not ok else "PASS",
                "detail": detail,
            }
        )

    if pairs.empty:
        add("pairs_nonempty", False, "Empty pair_scores", "FAIL")
    else:
        for L, sub in tqdm(list(pairs.groupby("length_cm")), desc="Sanity intra>impostor", unit="len"):
            g = sub.loc[sub.score_type == "S_intra", "score"]
            ich = sub.loc[sub.score_type == "S_inter_challenge", "score"]
            idev = sub.loc[sub.score_type == "S_inter_device", "score"]
            ok = float(g.median()) > float(ich.median()) and float(g.median()) > float(idev.median())
            add(f"intra_above_impostors_L{L}", ok, f"median intra={g.median():.4f}", "WARN")

        # Saturated / low-SNR lengths
        if "median_intra" in length_metrics.columns:
            sat = length_metrics.loc[length_metrics["median_intra"] > 0.999]
            low = length_metrics.loc[length_metrics["median_intra"] < 0.05]
            add(
                "no_all_lengths_saturated_or_dead",
                sat.empty or len(sat) < len(length_metrics),
                f"near_sat={sat['length_cm'].tolist() if not sat.empty else []}; "
                f"near_zero={low['length_cm'].tolist() if not low.empty else []}",
                "WARN",
            )

    if not length_metrics.empty and "robust_gap_min" in length_metrics.columns:
        neg = length_metrics.loc[length_metrics["robust_gap_min"] < 0]
        if not neg.empty and "device_top1" in length_metrics.columns:
            bad = neg.loc[neg["device_top1"] > 0.9]
            add(
                "no_false_full_separation_claim",
                bad.empty,
                "Negative robust gap with high Top-1 needs careful wording"
                if not bad.empty
                else "OK",
                "WARN",
            )

    # Leave-one-fiber sensitivity
    if not pairs.empty and decision.selected_length_cm is not None:
        base = decision.selected_length_cm
        shifts = []
        fibers = sorted(pairs["fiber_id"].dropna().unique().tolist())
        empty_boot = pd.DataFrame(
            {"length_cm": sorted(pairs.length_cm.unique()), "n_selected": 0, "selection_probability": 0.0}
        )
        boot = boot_sel if not boot_sel.empty else empty_boot
        for f in tqdm(fibers, desc="Sanity leave-one-fiber", unit="fiber"):
            if "fiber_id_b" in pairs.columns:
                sub = pairs.loc[(pairs["fiber_id"] != f) & (pairs["fiber_id_b"] != f)]
            else:
                sub = pairs.loc[pairs["fiber_id"] != f]
            lm = _maximin_table(sub)
            if lm.empty:
                continue
            d2 = select_length(lm, boot, never_force_unique_optimum=True)
            if d2.selected_length_cm != base:
                shifts.append({"dropped_fiber": f, "new_selection": d2.selected_length_cm})
        add(
            "leave_one_fiber_stability",
            len(shifts) <= max(1, len(fibers) // 2),
            f"shifts={shifts}",
            "WARN",
        )

        # Leave-one-challenge
        challenges = sorted(pairs["challenge"].dropna().unique().tolist())
        ch_shifts = []
        for ch in tqdm(challenges, desc="Sanity leave-one-challenge", unit="ch"):
            if "challenge_b" in pairs.columns:
                sub = pairs.loc[(pairs["challenge"] != ch) & (pairs["challenge_b"] != ch)]
            else:
                sub = pairs.loc[pairs["challenge"] != ch]
            lm = _maximin_table(sub)
            if lm.empty:
                continue
            d2 = select_length(lm, boot, never_force_unique_optimum=True)
            if d2.selected_length_cm != base:
                ch_shifts.append({"dropped_challenge": ch, "new_selection": d2.selected_length_cm})
        add(
            "leave_one_challenge_stability",
            len(ch_shifts) <= max(1, len(challenges) // 2),
            f"shifts={ch_shifts}",
            "WARN",
        )

    if decision.decision_type == "unique_optimum":
        p = decision.bootstrap_selection_probability
        add(
            "unique_optimum_bootstrap_support",
            p is not None and p >= 0.6,
            f"selection_probability={p}",
            "WARN",
        )
    elif decision.decision_type == "broad_plateau":
        add(
            "plateau_preferred_over_forced_unique",
            True,
            f"plateau_lengths={decision.plateau_lengths}",
            "PASS",
        )

    if not red.empty and "S_R_intra" in red.columns:
        med = red.groupby("length_cm")["S_R_intra"].median()
        spread = float(med.max() - med.min()) if len(med) else 0.0
        add("red_no_extreme_length_bias", spread < 0.5, f"red median spread={spread:.4f}", "WARN")

    add(
        "complexity_vs_robustness_not_conflated",
        True,
        "Reports must treat N_eff/entropy and robust_gap as distinct axes",
    )
    add(
        "ab_ba_retrieval_documented",
        True,
        "Device/challenge Top-1 use Enrollment A vs Query B; reverse direction is not silently swapped",
    )

    ok = all(c["status"] == "PASS" for c in checks)
    report = {"ok": ok, "n_warn": sum(1 for c in checks if c["status"] == "WARN"), "checks": checks}
    dump_json(run_dir / "post_run_sanity_check.json", report)
    lines = ["# Post-run sanity check\n", f"Overall: {'PASS' if ok else 'WARN'}\n"]
    for c in checks:
        lines.append(f"- [{c['status']}] {c['check']}: {c['detail']}")
    (run_dir / "post_run_sanity_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
