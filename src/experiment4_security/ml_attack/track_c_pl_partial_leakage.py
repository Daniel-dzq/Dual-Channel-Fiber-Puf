"""Track C-PL — partial-leakage registered-bank generalization to hidden challenges.

Threat model (conservative partial database-compromise, see
configs/partial_disclosure.yaml and
supplementary/partial_leakage/PARTIAL_LEAKAGE_ANALYSIS.md):

  Server (per device d, source state s), full registered bank C001..C128:
    common_full_A[d,s] = mean_c detail(A[d,s,c])                 (all 128)
    T_full[d,s,c]      = detail(A[d,s,c]) - common_full_A[d,s]   (all 128)
    Q[d,t,c|s]         = detail(B[d,t,c]) - common_full_A[d,s]

  Attacker (partial leakage): gets {bitmap_32[c] for c in C001..C128} (the
  challenge patterns are the server's *public*, fixed, already-registered
  bank) and {T_full[d,s,c] for c in LEAKED only}. Hidden-challenge
  *templates* never reach any fit/PCA/normalization/model-selection step —
  only their (public) challenge bitmaps are used, and only at *prediction*
  time, never at *fit* time.

This module fits PL0-PL5 on the leaked subset of one (device, source_state,
repetition, leak_size) cell and scores predictions for the hidden subset
against the SOURCE-STATE Round-B queries (Track C-PL, i.e. target == source).
Cross-state scoring against target != source lives in track_d_pl_transfer.py
and reuses the exact fitted predictors returned here (frozen, no refit).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiment4_security.ml_attack.batch_eval import (
    auc_eer_genuine_vs_negative,
    robust_gap_q05_minus_q95,
    row_zero_mean_ncc,
    score_distribution,
    stack_dict,
)
from experiment4_security.ml_attack.config import ModelsConfig
from experiment4_security.ml_attack.partial_leakage_models import (
    PL_ATTACK_METHOD_MAP,
    fit_pl0_mean,
    fit_pl1_nearest,
    fit_pl_regressors,
)
from experiment4_security.ml_attack.partial_leakage_splits import LeakSplit

logger = logging.getLogger(__name__)

PL_METHOD_ORDER = (
    "PL0_mean_leaked_response",
    "PL1_nearest_leaked_challenge",
    "PL2_ridge_clone",
    "PL3_kernel_ridge_clone",
    "PL4_random_fourier_ridge_clone",
    "PL5_small_mlp_clone",
)


# ---------------------------------------------------------------------------
# Shared-cache vector I/O (reads existing formal-run green vectors; never
# writes into the base run).
# ---------------------------------------------------------------------------

def green_vector_path(shared_cache_root: Path, device_id: str, state_id: str, round_id: str, challenge_id: str) -> Path:
    return Path(shared_cache_root) / "vectors" / f"{device_id}_{state_id}_{round_id}_{challenge_id}.npy"


def load_round_vectors(
    shared_cache_root: Path, device_id: str, state_id: str, round_id: str, challenge_ids: list[str]
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for cid in challenge_ids:
        p = green_vector_path(shared_cache_root, device_id, state_id, round_id, cid)
        if not p.exists():
            raise FileNotFoundError(f"Missing cached green vector: {p}")
        out[cid] = np.load(p).astype(np.float32)
    return out


def compute_common_full(a_vectors: dict[str, np.ndarray]) -> np.ndarray:
    """common_full_A[d,s] = mean over ALL registered challenges' Round-A detail."""
    stack = np.stack(list(a_vectors.values()), axis=0).astype(np.float64)
    return stack.mean(axis=0).astype(np.float32)


def to_templates(vectors: dict[str, np.ndarray], common: np.ndarray) -> dict[str, np.ndarray]:
    c64 = common.astype(np.float64)
    return {cid: (v.astype(np.float64) - c64).astype(np.float32) for cid, v in vectors.items()}


# ---------------------------------------------------------------------------
# Genuine reference (same-state; identical formula/value to Track A's S_G).
# ---------------------------------------------------------------------------

def compute_genuine_scores(T_full: dict[str, np.ndarray], Q_same_state: dict[str, np.ndarray]) -> dict[str, float]:
    ids = sorted(set(T_full) & set(Q_same_state))
    T = stack_dict(T_full, ids)
    Q = stack_dict(Q_same_state, ids)
    scores = row_zero_mean_ncc(T, Q)
    return {cid: float(scores[i]) for i, cid in enumerate(ids)}


# ---------------------------------------------------------------------------
# Fit PL0-PL5 on a leaked subset (never sees hidden templates / Round B).
# ---------------------------------------------------------------------------

def fit_all_pl_models(
    leaked_templates: dict[str, np.ndarray],
    challenge_features: dict[str, np.ndarray],
    hamming_matrix: pd.DataFrame,
    models_cfg: ModelsConfig,
    pca_dimension_max: int,
    *,
    which: tuple[str, ...] = PL_METHOD_ORDER,
) -> dict[str, Any]:
    leaked_ids = sorted(leaked_templates.keys())
    leaked_features = {cid: challenge_features[cid] for cid in leaked_ids}
    predictors: dict[str, Any] = {}
    if "PL0_mean_leaked_response" in which:
        predictors["PL0_mean_leaked_response"] = fit_pl0_mean(leaked_templates)
    if "PL1_nearest_leaked_challenge" in which:
        predictors["PL1_nearest_leaked_challenge"] = fit_pl1_nearest(leaked_templates, hamming_matrix)
    regressor_keys = tuple(k for k in which if k not in ("PL0_mean_leaked_response", "PL1_nearest_leaked_challenge"))
    if regressor_keys:
        predictors.update(
            fit_pl_regressors(
                leaked_features,
                leaked_templates,
                models_cfg,
                pca_dimension_max=pca_dimension_max,
                which=regressor_keys,
            )
        )
    return predictors


def predict_hidden_all(
    predictors: dict[str, Any], hidden_ids: list[str], challenge_features: dict[str, np.ndarray]
) -> dict[str, tuple[dict[str, np.ndarray], dict[str, Any]]]:
    """Returns {method_key: (predictions_by_hidden_id, extra_meta_by_hidden_id)}.

    Only ``challenge_features`` (challenge-side, public) are consulted for
    hidden ids — never any stored response for a hidden challenge.
    """
    out: dict[str, tuple[dict[str, np.ndarray], dict[str, Any]]] = {}
    for key, pred in predictors.items():
        if key == "PL0_mean_leaked_response":
            out[key] = (pred.predict_batch(hidden_ids), {})
        elif key == "PL1_nearest_leaked_challenge":
            preds, nearest = pred.predict_batch(hidden_ids)
            out[key] = (preds, {"nearest_leaked_challenge_id": nearest})
        else:
            hidden_feat = {cid: challenge_features[cid] for cid in hidden_ids}
            out[key] = (pred.predict_batch(hidden_feat, hidden_ids), {"effective_pca_dim": pred.effective_pca_dim})
    return out


# ---------------------------------------------------------------------------
# Score predictions against a target-state query set (source state == this
# module's C-PL; any target state == track_d_pl_transfer.py).
#
# Performance note: vectors here are ~1.68M-dim (fullres_detail_cm masked).
# With up to 46k (device x source x target x rep x leak x method) scoring
# calls in the full run, redundant float64 casts / repeated row-normalization
# dominate wall time. The helpers below do ONE float32 row-normalization
# pass and reuse it for both the matched (diagonal) score and the full
# n_hidden x n_hidden retrieval matrix, matching the same zero-mean NCC
# as batch_eval.row_zero_mean_ncc / metrics.compute_retrieval_metrics.
# ---------------------------------------------------------------------------

def _stack_f32(d: dict[str, np.ndarray], ids: list[str]) -> np.ndarray:
    return np.stack([d[i] for i in ids], axis=0).astype(np.float32, copy=False)


def _row_normalize_f32(mat: np.ndarray) -> np.ndarray:
    m = mat - mat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    m /= norms
    return m


def _retrieval_from_pairwise(sim: np.ndarray) -> dict[str, Any]:
    n = sim.shape[0]
    order = np.argsort(-sim, axis=1, kind="mergesort")  # descending-similarity order, stable
    inv = np.argsort(order, axis=1)  # inverse permutation: inv[i, j] = 0-indexed rank of column j
    ranks = (inv[np.arange(n), np.arange(n)] + 1).astype(np.float64)
    return {
        "n": n,
        "top1_rate": float(np.mean(ranks == 1)),
        "top5_rate": float(np.mean(ranks <= 5)),
        "median_rank": float(np.median(ranks)),
        "mean_reciprocal_rank": float(np.mean(1.0 / ranks)),
    }


def score_predictions_against_target(
    predictions: dict[str, np.ndarray],
    Q_target: dict[str, np.ndarray],
    hidden_ids: list[str],
    genuine_scores_source_state: dict[str, float],
    *,
    compute_residual_dominance: bool = True,
    pca_mean: np.ndarray | None = None,
) -> dict[str, Any]:
    ids = [c for c in hidden_ids if c in predictions and c in Q_target]
    P = _stack_f32(predictions, ids)
    Q = _stack_f32(Q_target, ids)

    Pn = _row_normalize_f32(P.copy())
    Qn = _row_normalize_f32(Q.copy())
    sim = Pn @ Qn.T  # (n_hidden, n_hidden) float32
    scores = np.diag(sim).astype(np.float64)
    retrieval = _retrieval_from_pairwise(sim)

    q_c = Q - Q.mean(axis=1, keepdims=True)
    denom = np.sum(q_c.astype(np.float64) ** 2, axis=1)
    num = np.sum((P.astype(np.float64) - Q.astype(np.float64)) ** 2, axis=1)
    nmse_vals = np.divide(num, denom, out=np.full_like(num, np.nan), where=denom > 0)

    g = np.array([genuine_scores_source_state[c] for c in ids], dtype=np.float64)
    s_sum = score_distribution(scores)
    g_sum = score_distribution(g)
    rg = robust_gap_q05_minus_q95(g, scores)
    auc_eer = auc_eer_genuine_vs_negative(g, scores)

    residual_ncc, dominance_flag = float("nan"), "NOT_COMPUTED_CROSS_STATE"
    if compute_residual_dominance:
        residual_ncc, dominance_flag = _residual_and_dominance_f32(P, Q, pca_mean)

    n_hidden = len(ids)
    return {
        "challenge_ids": ids,
        "s_a_partial": {cid: float(scores[i]) for i, cid in enumerate(ids)},
        "s_g_hidden": {cid: float(g[i]) for i, cid in enumerate(ids)},
        "nmse": {cid: float(nmse_vals[i]) for i, cid in enumerate(ids)},
        "score_summary": s_sum,
        "genuine_summary": g_sum,
        "rg_a_partial": rg,
        "auc_a_partial": auc_eer["auc"],
        "eer_a_partial": auc_eer["eer"],
        "retrieval": retrieval,
        "n_hidden": n_hidden,
        "chance_top1": (1.0 / n_hidden) if n_hidden else float("nan"),
        "chance_top5": min(5.0 / n_hidden, 1.0) if n_hidden else float("nan"),
        "residual_ncc": residual_ncc,
        "common_dominance_flag": dominance_flag,
    }


def _residual_and_dominance_f32(P: np.ndarray, Q: np.ndarray, pca_mean: np.ndarray | None) -> tuple[float, str]:
    if pca_mean is None or P.size == 0:
        return float("nan"), "NOT_APPLICABLE"
    mean = pca_mean.astype(np.float32)
    P_res = P - mean
    Q_res = Q - mean
    Pn = _row_normalize_f32(P_res.copy())
    Qn = _row_normalize_f32(Q_res.copy())
    residual_ncc = float(np.mean(np.sum(Pn * Qn, axis=1)))
    mean_bc = np.broadcast_to(mean, P.shape).copy()
    Pn2 = _row_normalize_f32(P.copy())
    Mn = _row_normalize_f32(mean_bc)
    pred_to_mean_ncc = float(np.mean(np.sum(Pn2 * Mn, axis=1)))
    dominated = np.isfinite(pred_to_mean_ncc) and pred_to_mean_ncc > 0.95 and (
        not np.isfinite(residual_ncc) or residual_ncc < 0.2
    )
    return residual_ncc, ("HIGH_NCC_DOMINATED_BY_COMMON_RESPONSE" if dominated else "OK")


def run_device_source_state(
    device_id: str,
    source_state: str,
    *,
    shared_cache_root: Path,
    all_challenge_ids: list[str],
    challenge_to_bank: dict[str, str],
    challenge_features: dict[str, np.ndarray],
    hamming_matrix: pd.DataFrame,
    splits_by_rep: dict[int, dict[int, LeakSplit]],
    models_cfg: ModelsConfig,
    pca_dimension_max: int,
    b_vectors_all_states: dict[str, dict[str, np.ndarray]],
    states_for_transfer: list[str],
    model_keys: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Fit PL0-PL5 once per (repetition, leak_size) on LEAKED templates only,
    predict each hidden challenge ONCE, then score that single frozen
    prediction set against every target state in ``states_for_transfer``
    (source state included => the diagonal == Track C-PL; all states =>
    Track D-PL). No target-state data is used for fitting or model
    selection at any point — only for the final NCC scoring pass.

    Returns one unified ``transfer_scores`` / ``transfer_summary`` table
    tagged with ``is_diagonal``; callers split C-PL (diagonal) vs D-PL (all
    rows) from this single pass to avoid refitting.
    """
    a_vectors = load_round_vectors(shared_cache_root, device_id, source_state, "A", all_challenge_ids)
    common_full = compute_common_full(a_vectors)
    T_full = to_templates(a_vectors, common_full)
    del a_vectors

    Q_by_target = {t: to_templates(b_vectors_all_states[t], common_full) for t in states_for_transfer}
    genuine_queries = Q_by_target.get(source_state)
    if genuine_queries is None:
        genuine_queries = to_templates(b_vectors_all_states[source_state], common_full)
    genuine_scores = compute_genuine_scores(T_full, genuine_queries)
    del genuine_queries

    summary_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []

    for rep, leak_map in splits_by_rep.items():
        for leak_size, split in leak_map.items():
            leaked_ids = list(split.leaked)
            hidden_ids = list(split.hidden)
            leaked_templates = {cid: T_full[cid] for cid in leaked_ids}

            predictors = fit_all_pl_models(
                leaked_templates,
                challenge_features,
                hamming_matrix,
                models_cfg,
                pca_dimension_max,
                which=tuple(model_keys) if model_keys else PL_METHOD_ORDER,
            )
            # Predict each hidden challenge exactly ONCE per method; reuse the
            # frozen prediction set for every target state below (no refit,
            # no re-prediction, per Section 7's "completely freeze" rule).
            predictions_all = predict_hidden_all(predictors, hidden_ids, challenge_features)

            for method_key, (preds, extra) in predictions_all.items():
                attack_method = PL_ATTACK_METHOD_MAP[method_key]
                pca_mean = getattr(predictors.get(method_key), "bundle", None)
                pca_mean = pca_mean.mu if pca_mean is not None else None
                effective_pca_dim = getattr(predictors.get(method_key), "effective_pca_dim", None)

                for target_state in states_for_transfer:
                    is_diag = target_state == source_state
                    # genuine reference is always the SOURCE-state genuine,
                    # fixed across all targets (matches Track D convention).
                    # Residual/common-dominance audit (an extra O(n_hidden x d)
                    # pass) is only computed on the diagonal to keep the full
                    # 6-device x 8-state x 8-target sweep tractable; it is a
                    # same-state diagnostic in Section 5, not a required D-PL
                    # field.
                    ev = score_predictions_against_target(
                        preds,
                        Q_by_target[target_state],
                        hidden_ids,
                        genuine_scores,
                        compute_residual_dominance=is_diag,
                        pca_mean=pca_mean,
                    )
                    for cid in ev["challenge_ids"]:
                        hidden_rows.append(
                            {
                                "device_id": device_id,
                                "source_state": source_state,
                                "target_state": target_state,
                                "is_diagonal": is_diag,
                                "repetition": rep,
                                "split_seed": split.seed,
                                "leak_size": leak_size,
                                "n_hidden": ev["n_hidden"],
                                "challenge_id": cid,
                                "bank_id": challenge_to_bank[cid],
                                "attack_method": attack_method,
                                "s_a_partial": ev["s_a_partial"][cid],
                                "s_g_hidden": ev["s_g_hidden"][cid],
                                "prediction_nmse": ev["nmse"][cid],
                                "nearest_leaked_challenge_id": (
                                    extra.get("nearest_leaked_challenge_id", {}).get(cid)
                                    if method_key == "PL1_nearest_leaked_challenge"
                                    else None
                                ),
                                "effective_pca_dim": effective_pca_dim,
                            }
                        )

                    residual_ncc, dominance_flag = ev["residual_ncc"], ev["common_dominance_flag"]
                    s_a = np.array(list(ev["s_a_partial"].values()), dtype=np.float64)
                    s_g = np.array(list(ev["s_g_hidden"].values()), dtype=np.float64)
                    nmse_vals = np.array(list(ev["nmse"].values()), dtype=np.float64)
                    summary_rows.append(
                        {
                            "device_id": device_id,
                            "source_state": source_state,
                            "target_state": target_state,
                            "is_diagonal": is_diag,
                            "repetition": rep,
                            "split_seed": split.seed,
                            "leak_size": leak_size,
                            "attack_method": attack_method,
                            "n_leaked": len(leaked_ids),
                            "n_hidden": ev["n_hidden"],
                            "median_S_A_partial": float(np.median(s_a)) if s_a.size else float("nan"),
                            "q05_S_A_partial": float(np.percentile(s_a, 5)) if s_a.size else float("nan"),
                            "q95_S_A_partial": float(np.percentile(s_a, 95)) if s_a.size else float("nan"),
                            "median_S_G_hidden": float(np.median(s_g)) if s_g.size else float("nan"),
                            "q05_S_G_hidden": float(np.percentile(s_g, 5)) if s_g.size else float("nan"),
                            "q95_S_G_hidden": float(np.percentile(s_g, 95)) if s_g.size else float("nan"),
                            "RG_A_partial": ev["rg_a_partial"],
                            "AUC_A_partial": ev["auc_a_partial"],
                            "EER_A_partial": ev["eer_a_partial"],
                            "Top1_hidden": ev["retrieval"]["top1_rate"],
                            "Top5_hidden": ev["retrieval"]["top5_rate"],
                            "MRR_hidden": ev["retrieval"]["mean_reciprocal_rank"],
                            "chance_top1": ev["chance_top1"],
                            "chance_top5": ev["chance_top5"],
                            "normalized_retrieval_lift": (
                                ev["retrieval"]["top1_rate"] / ev["chance_top1"]
                                if ev["chance_top1"] and np.isfinite(ev["retrieval"]["top1_rate"])
                                else float("nan")
                            ),
                            "median_NMSE": float(np.nanmedian(nmse_vals)) if nmse_vals.size else float("nan"),
                            "median_residual_ncc": residual_ncc,
                            "common_dominance_flag": dominance_flag,
                            "effective_pca_dim": effective_pca_dim,
                        }
                    )
            del predictions_all, predictors

    df = pd.DataFrame(summary_rows)
    if not df.empty:
        pl0 = df[(df["attack_method"] == "mean_leaked_response") & (df["is_diagonal"])].set_index(
            ["repetition", "leak_size"]
        )["median_S_A_partial"]
        df["gain_over_mean"] = df.apply(
            lambda r: r["median_S_A_partial"] - pl0.get((r["repetition"], r["leak_size"]), np.nan), axis=1
        )

    hidden_df = pd.DataFrame(hidden_rows)
    if not hidden_df.empty:
        c_pl_hidden = (
            hidden_df[hidden_df["is_diagonal"]]
            .drop(columns=["target_state", "is_diagonal"])
            .reset_index(drop=True)
        )
    else:
        c_pl_hidden = hidden_df

    return {
        "transfer_summary": df,
        "c_pl_summary": df[df["is_diagonal"]].drop(columns=["target_state", "is_diagonal"]).reset_index(drop=True),
        "transfer_hidden_scores": hidden_df,
        "c_pl_hidden_scores": c_pl_hidden,
        "genuine_scores": genuine_scores,
    }
