"""Metrics, thresholds, bootstrap, permutation, and pilot diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from puf_common.metrics import (
    auc_roc,
    d_prime,
    equal_error_rate_with_threshold,
    margin,
    robust_gap,
)


def score_quantiles(scores: np.ndarray) -> dict[str, float]:
    s = np.asarray(scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return {"q5": float("nan"), "median": float("nan"), "q95": float("nan"), "n": 0}
    return {
        "q5": float(np.percentile(s, 5)),
        "median": float(np.median(s)),
        "q95": float(np.percentile(s, 95)),
        "n": int(s.size),
    }


def binary_metrics(genuine: np.ndarray, impostor: np.ndarray) -> dict[str, float]:
    g = np.asarray(genuine, dtype=np.float64)
    i = np.asarray(impostor, dtype=np.float64)
    g = g[np.isfinite(g)]
    i = i[np.isfinite(i)]
    if g.size == 0 or i.size == 0:
        return {
            "auc": float("nan"),
            "eer": float("nan"),
            "far": float("nan"),
            "frr": float("nan"),
            "threshold_eer": float("nan"),
            "d_prime": float("nan"),
            "median_margin": float("nan"),
            "robust_gap": float("nan"),
            "n_genuine": int(g.size),
            "n_impostor": int(i.size),
        }
    eer, thr = equal_error_rate_with_threshold(g, i)
    far = float(np.mean(i >= thr)) if np.isfinite(thr) else float("nan")
    frr = float(np.mean(g < thr)) if np.isfinite(thr) else float("nan")
    return {
        "auc": auc_roc(g, i),
        "eer": eer,
        "far": far,
        "frr": frr,
        "threshold_eer": thr,
        "d_prime": d_prime(g, i),
        "median_margin": margin(g, i),
        "robust_gap": robust_gap(g, i),
        "n_genuine": int(g.size),
        "n_impostor": int(i.size),
        **{f"genuine_{k}": v for k, v in score_quantiles(g).items()},
        **{f"impostor_{k}": v for k, v in score_quantiles(i).items()},
    }


def summarize_group_pair(
    scores_df: pd.DataFrame,
    genuine_group: str,
    impostor_group: str,
) -> dict[str, Any]:
    g = scores_df.loc[scores_df["group"] == genuine_group, "score"].to_numpy()
    i = scores_df.loc[scores_df["group"] == impostor_group, "score"].to_numpy()
    out = binary_metrics(g, i)
    out["genuine_group"] = genuine_group
    out["impostor_group"] = impostor_group
    return out


def red_identification_metrics(query_df: pd.DataFrame, *, n_devices: int) -> dict[str, Any]:
    if query_df.empty:
        return {"n_queries": 0}
    ranks = query_df["rank_correct"].to_numpy(dtype=float)
    top1 = float(np.mean(query_df["correct"].astype(bool)))
    top3 = (
        float(np.mean(ranks <= 3))
        if n_devices >= 3
        else float("nan")
    )
    mrr = float(np.mean(query_df["mrr_contribution"]))
    margins = query_df["identity_margin"].to_numpy(dtype=float)
    # Genuine = correct identity score; impostor = best wrong (per query) is not a full
    # distribution — use all same-device cross-state vs diff-device from scores separately.
    return {
        "n_queries": int(len(query_df)),
        "top1": top1,
        "top3": top3,
        "mrr": mrr,
        "median_identity_margin": float(np.nanmedian(margins)),
        "fraction_positive_margin": float(np.mean(margins > 0)) if margins.size else float("nan"),
        "chance_top1": 1.0 / max(n_devices, 1),
        "per_query": query_df.to_dict(orient="records"),
    }


def select_threshold_eer(genuine: np.ndarray, impostor: np.ndarray) -> float:
    _eer, thr = equal_error_rate_with_threshold(genuine, impostor)
    return float(thr)


def system_metrics(
    red_scores: pd.DataFrame,
    green_scores: pd.DataFrame,
    *,
    tau_r: float,
    tau_g: float,
) -> dict[str, Any]:
    def _p(df: pd.DataFrame, group: str, pred) -> float:
        s = df.loc[df["group"] == group, "score"].to_numpy(dtype=float)
        s = s[np.isfinite(s)]
        if s.size == 0:
            return float("nan")
        return float(np.mean(pred(s)))

    # Unlinkability: distinguish same-device cross-state green from different-device
    cross = green_scores.loc[
        green_scores["group"] == "same_device_diff_state_same_challenge", "score"
    ].to_numpy()
    diff = green_scores.loc[
        green_scores["group"] == "diff_device_same_state_same_challenge", "score"
    ].to_numpy()
    # AUC near 0.5 is desirable (hard to link). Use cross as "positive" class.
    unlink_auc = auc_roc(cross, diff) if cross.size and diff.size else float("nan")

    return {
        "tau_R": float(tau_r),
        "tau_G": float(tau_g),
        "identity_continuity": _p(
            red_scores, "same_device_diff_state", lambda s: s >= tau_r
        ),
        "replacement_rejection": _p(
            red_scores, "diff_device", lambda s: s < tau_r
        ),
        "reenrollment_success": _p(
            green_scores,
            "same_device_same_state_same_challenge",
            lambda s: s >= tau_g,
        ),
        "revocation_rate": _p(
            green_scores,
            "same_device_diff_state_same_challenge",
            lambda s: s < tau_g,
        ),
        "challenge_rejection": _p(
            green_scores,
            "same_device_same_state_diff_challenge",
            lambda s: s < tau_g,
        ),
        "device_rejection": _p(
            green_scores,
            "diff_device_same_state_same_challenge",
            lambda s: s < tau_g,
        ),
        "unlinkability_auc": unlink_auc,
    }


def two_channel_decisions(
    *,
    s_r: float,
    s_g: float,
    tau_r: float,
    tau_g: float,
) -> dict[str, Any]:
    red_pass = bool(s_r >= tau_r)
    green_pass = bool(s_g >= tau_g)
    if red_pass and green_pass:
        outcome = "original_device_enrolled_state"
        action = "accept"
    elif red_pass and not green_pass:
        outcome = "original_device_state_changed"
        action = "require_authorized_reenrollment"
    elif (not red_pass) and (not green_pass):
        outcome = "replacement_or_anomaly"
        action = "reject"
    else:
        outcome = "contradictory_or_replay_like"
        action = "reject"
    return {
        "red_pass": red_pass,
        "green_pass": green_pass,
        "outcome": outcome,
        "action": action,
        "S_R": s_r,
        "S_G": s_g,
    }


def cluster_bootstrap_mean(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    ok = np.isfinite(values)
    values, cluster_ids = values[ok], cluster_ids[ok]
    units = np.unique(cluster_ids)
    if values.size == 0 or units.size < 2:
        return {
            "mean": float(np.nanmean(values)) if values.size else float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_clusters": int(units.size),
            "reliable": False,
        }
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample_units = rng.choice(units, size=units.size, replace=True)
        parts = [values[cluster_ids == u] for u in sample_units]
        boots.append(float(np.mean(np.concatenate(parts))))
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "n_clusters": int(units.size),
        "reliable": True,
    }


def permutation_top1(
    query_df: pd.DataFrame,
    *,
    n_perm: int,
    seed: int,
) -> dict[str, float]:
    if query_df.empty or "rank_correct" not in query_df.columns:
        return {"observed_top1": float("nan"), "p_value": float("nan"), "n_perm": 0}
    observed = float(np.mean(query_df["correct"].astype(bool)))
    n_gallery = int(query_df["n_gallery"].iloc[0]) if "n_gallery" in query_df else 0
    if n_gallery < 2:
        return {"observed_top1": observed, "p_value": float("nan"), "n_perm": 0}
    rng = np.random.default_rng(seed)
    null = []
    n_q = len(query_df)
    for _ in range(n_perm):
        # Under label permutation, chance top-1 is 1/n_gallery per query
        hits = rng.random(n_q) < (1.0 / n_gallery)
        null.append(float(np.mean(hits)))
    p = float((np.sum(np.asarray(null) >= observed) + 1) / (n_perm + 1))
    return {"observed_top1": observed, "p_value": p, "n_perm": n_perm}


def rates_at_fixed_threshold(
    genuine: np.ndarray,
    impostor: np.ndarray,
    *,
    tau: float,
) -> dict[str, float]:
    """FAR/FRR/TPR at a frozen operational threshold (not EER-derived on this set)."""
    g = np.asarray(genuine, dtype=np.float64)
    i = np.asarray(impostor, dtype=np.float64)
    g = g[np.isfinite(g)]
    i = i[np.isfinite(i)]
    if g.size == 0 or i.size == 0 or not np.isfinite(tau):
        return {
            "tau_operational": float(tau),
            "far_at_dev_threshold": float("nan"),
            "frr_at_dev_threshold": float("nan"),
            "tpr_at_dev_threshold": float("nan"),
            "n_genuine": int(g.size),
            "n_impostor": int(i.size),
        }
    far = float(np.mean(i >= tau))
    frr = float(np.mean(g < tau))
    tpr = float(np.mean(g >= tau))
    return {
        "tau_operational": float(tau),
        "far_at_dev_threshold": far,
        "frr_at_dev_threshold": frr,
        "tpr_at_dev_threshold": tpr,
        "n_genuine": int(g.size),
        "n_impostor": int(i.size),
    }


def binary_metrics_with_operational_threshold(
    genuine: np.ndarray,
    impostor: np.ndarray,
    *,
    tau_operational: float,
) -> dict[str, float]:
    """AUC/EER as descriptive separability; FAR/FRR use development-fixed tau."""
    base = binary_metrics(genuine, impostor)
    rates = rates_at_fixed_threshold(genuine, impostor, tau=tau_operational)
    out = {
        **base,
        **rates,
        "eer_note": (
            "The cohort EER / threshold_eer is reported only as a threshold-free "
            "separability summary. Operational decisions use thresholds fixed on "
            "the development cohort."
        ),
        "operational_threshold": float(tau_operational),
        "frozen_eer_threshold_is_operational": False,
    }
    # Ensure descriptive EER threshold is not confused with operational
    out["threshold_eer_descriptive_only"] = out.get("threshold_eer")
    return out


def summarize_group_pair_at_tau(
    scores_df: pd.DataFrame,
    genuine_group: str,
    impostor_group: str,
    *,
    tau_operational: float,
) -> dict[str, Any]:
    g = scores_df.loc[scores_df["group"] == genuine_group, "score"].to_numpy()
    i = scores_df.loc[scores_df["group"] == impostor_group, "score"].to_numpy()
    out = binary_metrics_with_operational_threshold(g, i, tau_operational=tau_operational)
    out["genuine_group"] = genuine_group
    out["impostor_group"] = impostor_group
    return out


def device_bootstrap_system_metrics(
    red_scores: pd.DataFrame,
    green_scores: pd.DataFrame,
    *,
    devices: list[str],
    tau_r: float,
    tau_g: float,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Device-level bootstrap of system rates on a fixed device set."""
    from experiment3.splits import filter_score_pairs_by_devices

    devices = sorted(devices)
    if len(devices) < 2:
        return {
            "bootstrap_unit": "device",
            "number_of_devices": len(devices),
            "number_of_replicates": 0,
            "random_seed": seed,
            "reliable": False,
            "metrics": {},
        }
    rng = np.random.default_rng(seed)
    keys = [
        "identity_continuity",
        "replacement_rejection",
        "reenrollment_success",
        "revocation_rate",
        "challenge_rejection",
        "device_rejection",
        "unlinkability_auc",
    ]
    stores: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        sample = list(rng.choice(devices, size=len(devices), replace=True))
        # Unique devices for filtering pairs (with multiplicity ignored for pair filter)
        uniq = sorted(set(sample))
        red_b = filter_score_pairs_by_devices(red_scores, uniq)
        green_b = filter_score_pairs_by_devices(green_scores, uniq)
        m = system_metrics(red_b, green_b, tau_r=tau_r, tau_g=tau_g)
        for k in keys:
            stores[k].append(float(m.get(k, float("nan"))))
    summary = {}
    for k, vals in stores.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            summary[k] = {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
        else:
            summary[k] = {
                "mean": float(np.mean(arr)),
                "ci_low": float(np.percentile(arr, 2.5)),
                "ci_high": float(np.percentile(arr, 97.5)),
            }
    return {
        "bootstrap_unit": "device",
        "number_of_devices": len(devices),
        "number_of_replicates": n_boot,
        "random_seed": seed,
        "reliable": True,
        "metrics": summary,
    }


def pilot_diagnostics(
    *,
    query_df: pd.DataFrame,
    green_scores: pd.DataFrame,
    system: dict[str, Any],
    n_devices: int,
) -> dict[str, str]:
    """Core diagnostic statuses (caller must scope the input tables)."""
    if query_df.empty:
        red_status = "inconclusive"
    else:
        all_top1 = bool(query_df["correct"].all())
        all_pos = bool((query_df["identity_margin"] > 0).all())
        if all_top1 and all_pos:
            red_status = "encouraging"
        elif (not all_top1) and float(query_df["correct"].mean()) < 0.5:
            red_status = "no-go"
        else:
            red_status = "inconclusive"

    genu = green_scores.loc[
        green_scores["group"] == "same_device_same_state_same_challenge", "score"
    ].to_numpy()
    wrong_ch = green_scores.loc[
        green_scores["group"] == "same_device_same_state_diff_challenge", "score"
    ].to_numpy()
    wrong_dev = green_scores.loc[
        green_scores["group"] == "diff_device_same_state_same_challenge", "score"
    ].to_numpy()
    cross = green_scores.loc[
        green_scores["group"] == "same_device_diff_state_same_challenge", "score"
    ].to_numpy()

    if genu.size == 0:
        green_rel = "inconclusive"
    else:
        med_g = float(np.median(genu))
        med_wc = float(np.median(wrong_ch)) if wrong_ch.size else float("-inf")
        med_wd = float(np.median(wrong_dev)) if wrong_dev.size else float("-inf")
        if med_g > med_wc and med_g > med_wd:
            green_rel = "encouraging"
        elif med_g <= max(med_wc, med_wd):
            green_rel = "no-go"
        else:
            green_rel = "inconclusive"

    tau_g = system.get("tau_G", float("nan"))
    if genu.size == 0 or cross.size == 0:
        green_rev = "inconclusive"
    else:
        below_genuine = float(np.median(cross)) < float(np.median(genu))
        mostly_below = (
            float(np.mean(cross < tau_g)) >= 0.5 if np.isfinite(tau_g) else False
        )
        if below_genuine and mostly_below:
            green_rev = "encouraging"
        elif float(np.median(cross)) >= float(np.median(genu)):
            green_rev = "no-go"
        else:
            green_rev = "inconclusive"

    statuses = {red_status, green_rel, green_rev}
    if "no-go" in statuses:
        full = "no-go"
    elif statuses == {"encouraging"}:
        full = "proceed"
    elif red_status == "encouraging" and green_rel == "encouraging":
        full = (
            "proceed"
            if green_rev != "no-go"
            else "fix acquisition or processing before continuing"
        )
    else:
        full = "fix acquisition or processing before continuing"

    return {
        "RED_IDENTITY": red_status,
        "GREEN_SAME_STATE_RELIABILITY": green_rel,
        "GREEN_STATE_REVOCATION": green_rev,
        "FULL_IDENTITY_STATE_STORY": full,
    }


def development_readiness_gate(
    *,
    query_df: pd.DataFrame,
    green_scores: pd.DataFrame,
    system: dict[str, Any],
    n_devices: int,
) -> dict[str, str]:
    """Development-only readiness gate. Must not receive frozen devices."""
    out = pilot_diagnostics(
        query_df=query_df,
        green_scores=green_scores,
        system=system,
        n_devices=n_devices,
    )
    out["gate_scope"] = "development_only"
    out["proceed_key"] = out.get("FULL_IDENTITY_STATE_STORY")
    return out


def frozen_confirmation(
    *,
    query_df: pd.DataFrame,
    green_scores: pd.DataFrame,
    system: dict[str, Any],
    n_devices: int,
) -> dict[str, str]:
    """One-shot frozen confirmation. Must not be used to retune parameters."""
    out = pilot_diagnostics(
        query_df=query_df,
        green_scores=green_scores,
        system=system,
        n_devices=n_devices,
    )
    out["gate_scope"] = "frozen_formal"
    out["note"] = (
        "Frozen confirmation only; failures must not trigger automatic retuning "
        "of thresholds, mask, standardizer, or features."
    )
    return out


def build_scope_metric_pack(
    *,
    red_scores: pd.DataFrame,
    green_scores: pd.DataFrame,
    query_df: pd.DataFrame,
    devices: list[str],
    tau_r: float,
    tau_g: float,
    evaluation_scope: str,
    claim_label: str,
    include_bootstrap: bool,
    bootstrap_iterations: int,
    random_seed: int,
    include_permutation: bool,
    permutation_iterations: int,
) -> dict[str, Any]:
    """Compute red/green/system metrics for one evaluation scope."""
    from experiment3.splits import filter_queries_by_devices, filter_score_pairs_by_devices

    red_s = filter_score_pairs_by_devices(red_scores, devices)
    green_s = filter_score_pairs_by_devices(green_scores, devices)
    query_s = filter_queries_by_devices(query_df, devices)
    n_devices = len(devices)

    red_id = red_identification_metrics(query_s, n_devices=n_devices)
    red_genu = red_s.loc[red_s["group"] == "same_device_diff_state", "score"].to_numpy()
    red_imp = red_s.loc[red_s["group"] == "diff_device", "score"].to_numpy()
    red_sep = binary_metrics_with_operational_threshold(
        red_genu, red_imp, tau_operational=tau_r
    )

    green_metrics = {
        "same_vs_diff_challenge": summarize_group_pair_at_tau(
            green_s,
            "same_device_same_state_same_challenge",
            "same_device_same_state_diff_challenge",
            tau_operational=tau_g,
        ),
        "same_vs_diff_device": summarize_group_pair_at_tau(
            green_s,
            "same_device_same_state_same_challenge",
            "diff_device_same_state_same_challenge",
            tau_operational=tau_g,
        ),
        "reenrollment_vs_revocation": summarize_group_pair_at_tau(
            green_s,
            "same_device_same_state_same_challenge",
            "same_device_diff_state_same_challenge",
            tau_operational=tau_g,
        ),
    }

    sys_m = system_metrics(red_s, green_s, tau_r=float(tau_r), tau_g=float(tau_g))
    sys_m["evaluation_scope"] = evaluation_scope
    sys_m["devices"] = list(devices)
    sys_m["claim_label"] = claim_label
    sys_m["tau_R_dev"] = float(tau_r)
    sys_m["tau_G_dev"] = float(tau_g)

    bootstrap: dict[str, Any] = {}
    if include_bootstrap and n_devices >= 2:
        bootstrap = device_bootstrap_system_metrics(
            red_s,
            green_s,
            devices=devices,
            tau_r=float(tau_r),
            tau_g=float(tau_g),
            n_boot=min(bootstrap_iterations, 500),
            seed=random_seed,
        )
    elif include_bootstrap:
        bootstrap = {
            "warning": f"Device bootstrap not reliable with n_devices={n_devices}",
            "reliable": False,
        }

    perm: dict[str, Any] = {}
    if include_permutation and n_devices >= 2 and not query_s.empty:
        perm = permutation_top1(
            query_s, n_perm=permutation_iterations, seed=random_seed
        )
    else:
        perm = {"skipped": True, "reason": "insufficient devices or empty queries"}

    red_metrics = {
        "identification": {k: v for k, v in red_id.items() if k != "per_query"},
        "genuine_vs_impostor": red_sep,
        "bootstrap": bootstrap,
        "permutation": perm,
        "evaluation_scope": evaluation_scope,
        "devices": list(devices),
        "claim_label": claim_label,
    }
    green_metrics["evaluation_scope"] = evaluation_scope
    green_metrics["devices"] = list(devices)
    green_metrics["claim_label"] = claim_label

    return {
        "red_metrics": red_metrics,
        "green_metrics": green_metrics,
        "system_metrics": sys_m,
        "red_scores": red_s,
        "green_scores": green_s,
        "query_df": query_s,
        "devices": list(devices),
        "evaluation_scope": evaluation_scope,
        "claim_label": claim_label,
        "bootstrap": bootstrap,
    }


def per_device_metrics(
    red_scores: pd.DataFrame,
    green_scores: pd.DataFrame,
    query_df: pd.DataFrame,
) -> pd.DataFrame:
    devices = sorted(
        set(red_scores.get("device_id_a", pd.Series(dtype=str)).tolist())
        | set(green_scores.get("device_id_a", pd.Series(dtype=str)).tolist())
    )
    rows = []
    for d in devices:
        q = query_df[query_df["query_device"] == d] if not query_df.empty else query_df
        g = green_scores[
            (green_scores["device_id_a"] == d) & (green_scores["device_id_b"] == d)
        ]
        genu = g.loc[g["group"] == "same_device_same_state_same_challenge", "score"]
        cross = g.loc[g["group"] == "same_device_diff_state_same_challenge", "score"]
        rows.append(
            {
                "device_id": d,
                "n_queries": int(len(q)),
                "top1": float(q["correct"].mean()) if len(q) else float("nan"),
                "median_identity_margin": float(q["identity_margin"].median())
                if len(q)
                else float("nan"),
                "green_genuine_median": float(genu.median()) if len(genu) else float("nan"),
                "green_cross_state_median": float(cross.median())
                if len(cross)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def per_state_metrics(green_scores: pd.DataFrame, query_df: pd.DataFrame) -> pd.DataFrame:
    states = sorted(set(green_scores.get("state_id_a", pd.Series(dtype=str)).tolist()))
    rows = []
    for s in states:
        g = green_scores[green_scores["state_id_a"] == s]
        genu = g.loc[g["group"] == "same_device_same_state_same_challenge", "score"]
        q = query_df[query_df["query_state"] == s] if not query_df.empty else query_df
        rows.append(
            {
                "state_id": s,
                "green_genuine_median": float(genu.median()) if len(genu) else float("nan"),
                "n_queries": int(len(q)),
                "top1": float(q["correct"].mean()) if len(q) else float("nan"),
            }
        )
    return pd.DataFrame(rows)
