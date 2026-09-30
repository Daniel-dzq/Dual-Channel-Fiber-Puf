"""Batched zero-mean NCC and unified score evaluation (S_G/S_C/S_X/S_A)."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiment4_security.ml_attack.metric_schema import (
    COMPARISON_G_VS_A,
    COMPARISON_G_VS_C,
    COMPARISON_G_VS_X,
    POSITIVE_CLASS,
    REPRESENTATION,
    RETRIEVAL_CANDIDATE_COUNT,
    SCORE_CHALLENGE_MISMATCH,
    SCORE_CROSS_STATE_CREDENTIAL,
    SCORE_DIRECTION,
    SCORE_GENUINE,
    SCORE_SOFTWARE_CLONE,
    retrieval_meta,
    summary_base_fields,
)
from experiment4_security.ml_attack.metrics import compute_retrieval_metrics
from puf_common.ncc import zero_mean_ncc as _scalar_zero_mean_ncc
from puf_common.metrics import auc_roc, equal_error_rate


def center_normalize_rows(mat: np.ndarray) -> np.ndarray:
    m = mat.astype(np.float64)
    m = m - m.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return m / norms


def pairwise_zero_mean_ncc(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return center_normalize_rows(A) @ center_normalize_rows(B).T


def pairwise_zero_mean_ncc_blocked(A: np.ndarray, B: np.ndarray, *, block_size: int = 65536) -> np.ndarray:
    """Float64 NCC without materializing full float64 response matrices.

    Long float32 dot products can depend on the BLAS reduction path. Center,
    accumulate norms, and multiply in float64, using bounded coordinate blocks.
    Constant responses have score zero, matching the scalar NCC definition.
    """
    A, B = np.asarray(A), np.asarray(B)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        raise ValueError("NCC requires two matrices with matching vector lengths")
    if block_size <= 0 or A.shape[1] == 0:
        raise ValueError("NCC requires positive block size and nonempty vectors")
    am = A.mean(axis=1, dtype=np.float64)[:, None]
    bm = B.mean(axis=1, dtype=np.float64)[:, None]
    products = np.zeros((len(A), len(B)), dtype=np.float64)
    an = np.zeros(len(A), dtype=np.float64)
    bn = np.zeros(len(B), dtype=np.float64)
    for start in range(0, A.shape[1], block_size):
        ac = A[:, start:start + block_size].astype(np.float64) - am
        bc = B[:, start:start + block_size].astype(np.float64) - bm
        products += ac @ bc.T
        an += np.einsum("ij,ij->i", ac, ac)
        bn += np.einsum("ij,ij->i", bc, bc)
    denom = np.sqrt(an)[:, None] * np.sqrt(bn)[None, :]
    return np.divide(products, denom, out=np.zeros_like(products), where=denom > 0)


def row_zero_mean_ncc(preds: np.ndarray, meas: np.ndarray) -> np.ndarray:
    preds_c = preds - preds.mean(axis=1, keepdims=True)
    meas_c = meas - meas.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(preds_c, axis=1) * np.linalg.norm(meas_c, axis=1)
    num = np.sum(preds_c * meas_c, axis=1)
    return np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)


def stack_dict(d: dict[str, np.ndarray], ids: list[str]) -> np.ndarray:
    return np.stack([d[i] for i in ids], axis=0).astype(np.float64)


def score_distribution(vals: np.ndarray) -> dict[str, float]:
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "q05": float("nan"),
            "q95": float("nan"),
            "minimum": float("nan"),
            "maximum": float("nan"),
            "std": float("nan"),
        }
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "q05": float(np.percentile(vals, 5)),
        "q95": float(np.percentile(vals, 95)),
        "minimum": float(vals.min()),
        "maximum": float(vals.max()),
        "std": float(vals.std()),
    }


def robust_gap_q05_minus_q95(genuine: np.ndarray, negative: np.ndarray) -> float:
    """Unified RG_* = Q05(Genuine) - Q95(non-genuine)."""
    g = np.asarray(genuine, dtype=np.float64)
    n = np.asarray(negative, dtype=np.float64)
    g = g[np.isfinite(g)]
    n = n[np.isfinite(n)]
    if g.size == 0 or n.size == 0:
        return float("nan")
    return float(np.percentile(g, 5) - np.percentile(n, 95))


def auc_eer_genuine_vs_negative(genuine: np.ndarray, negative: np.ndarray) -> dict[str, Any]:
    g = np.asarray(genuine, dtype=np.float64)
    n = np.asarray(negative, dtype=np.float64)
    return {
        "positive_class": POSITIVE_CLASS,
        "score_direction": SCORE_DIRECTION,
        "auc": float(auc_roc(g, n)) if g.size and n.size else float("nan"),
        "eer": float(equal_error_rate(g, n)) if g.size and n.size else float("nan"),
    }


def evaluate_database_authentication(
    templates: dict[str, np.ndarray],
    queries: dict[str, np.ndarray],
    challenge_ids: list[str],
    *,
    device_id: str,
    state_id: str,
) -> dict[str, Any]:
    """Track A: S_G vs S_C on one mechanical state."""
    ids = [c for c in challenge_ids if c in templates and c in queries]
    T = stack_dict(templates, ids)
    Q = stack_dict(queries, ids)
    sims = pairwise_zero_mean_ncc(T, Q)
    genuine = np.diag(sims)
    mask = ~np.eye(len(ids), dtype=bool)
    challenge_mismatch = sims[mask]
    retrieval = compute_retrieval_metrics(
        {cid: templates[cid] for cid in ids},
        {cid: queries[cid] for cid in ids},
    )
    g_sum = score_distribution(genuine)
    c_sum = score_distribution(challenge_mismatch)
    rg_c = robust_gap_q05_minus_q95(genuine, challenge_mismatch)
    auc_eer = auc_eer_genuine_vs_negative(genuine, challenge_mismatch)
    auc_eer["negative_class"] = SCORE_CHALLENGE_MISMATCH

    score_rows: list[dict[str, Any]] = []
    for i, cid in enumerate(ids):
        score_rows.append(
            {
                "sample_id": f"{device_id}_{state_id}_A_{cid}",
                "device_id": device_id,
                "source_state": state_id,
                "target_state": state_id,
                "challenge_id": cid,
                "query_challenge_id": cid,
                "score_type": SCORE_GENUINE,
                "attack_method": None,
                "representation": REPRESENTATION,
                "score": float(genuine[i]),
                "is_genuine": True,
                "positive_class": POSITIVE_CLASS,
                "negative_class": SCORE_CHALLENGE_MISMATCH,
                "score_direction": SCORE_DIRECTION,
            }
        )
    # Store compact S_C sample for audit (full off-diagonal is huge).
    if challenge_mismatch.size:
        rng = np.random.default_rng(20260721)
        take = min(512, challenge_mismatch.size)
        for val in rng.choice(challenge_mismatch, size=take, replace=False):
            score_rows.append(
                {
                    "sample_id": None,
                    "device_id": device_id,
                    "source_state": state_id,
                    "target_state": state_id,
                    "challenge_id": None,
                    "query_challenge_id": None,
                    "score_type": SCORE_CHALLENGE_MISMATCH,
                    "attack_method": None,
                    "representation": REPRESENTATION,
                    "score": float(val),
                    "is_genuine": False,
                    "positive_class": POSITIVE_CLASS,
                    "negative_class": SCORE_CHALLENGE_MISMATCH,
                    "score_direction": SCORE_DIRECTION,
                }
            )

    summary = summary_base_fields(
        source_state=state_id,
        target_state=state_id,
        attack_method=None,
        comparison_type=COMPARISON_G_VS_C,
        n_positive=g_sum["n"],
        n_negative=c_sum["n"],
        genuine_summary=g_sum,
        negative_summary=c_sum,
        robust_gap=rg_c,
        auc=auc_eer["auc"],
        eer=auc_eer["eer"],
        retrieval=retrieval,
    )
    summary.update(
        {
            "state_id": state_id,
            "median_S_G": g_sum["median"],
            "q05_S_G": g_sum["q05"],
            "minimum_S_G": g_sum["minimum"],
            "mean_S_G": g_sum["mean"],
            "median_S_C": c_sum["median"],
            "q95_S_C": c_sum["q95"],
            "maximum_S_C": c_sum["maximum"],
            "mean_S_C": c_sum["mean"],
            "rg_challenge": rg_c,
            "auc_challenge": auc_eer["auc"],
            "eer_challenge": auc_eer["eer"],
            "negative_class": SCORE_CHALLENGE_MISMATCH,
        }
    )
    summary.update(retrieval_meta(RETRIEVAL_CANDIDATE_COUNT))
    return {
        "challenge_ids": ids,
        "genuine_scores": genuine,
        "challenge_mismatch_scores": challenge_mismatch,
        "genuine_summary": g_sum,
        "challenge_mismatch_summary": c_sum,
        "rg_challenge": rg_c,
        "retrieval": retrieval,
        "auc_eer": auc_eer,
        "summary_row": summary,
        "score_rows": score_rows,
    }


def evaluate_cross_state_credential(
    templates: dict[str, np.ndarray],
    queries: dict[str, np.ndarray],
    challenge_ids: list[str],
    *,
    device_id: str,
    source_state: str,
    target_state: str,
    genuine_same_state: np.ndarray | None,
) -> dict[str, Any]:
    """Track B: S_X(s→t) vs S_G(s)."""
    ids = [c for c in challenge_ids if c in templates and c in queries]
    T = stack_dict(templates, ids)
    Q = stack_dict(queries, ids)
    matched = row_zero_mean_ncc(T, Q)
    retrieval = compute_retrieval_metrics(
        {cid: templates[cid] for cid in ids},
        {cid: queries[cid] for cid in ids},
    )
    x_sum = score_distribution(matched)
    g = np.asarray(genuine_same_state, dtype=np.float64) if genuine_same_state is not None else np.array([])
    g_sum = score_distribution(g) if g.size else {"n": 0, "median": float("nan"), "q05": float("nan"), "q95": float("nan")}
    rg_x = robust_gap_q05_minus_q95(g, matched) if g.size else float("nan")
    auc_eer = auc_eer_genuine_vs_negative(g, matched) if g.size else {"auc": float("nan"), "eer": float("nan"), "positive_class": POSITIVE_CLASS, "score_direction": SCORE_DIRECTION}
    auc_eer["negative_class"] = SCORE_CROSS_STATE_CREDENTIAL

    score_rows = [
        {
            "sample_id": f"{device_id}_{source_state}_A_{cid}__vs__{device_id}_{target_state}_B_{cid}",
            "device_id": device_id,
            "source_state": source_state,
            "target_state": target_state,
            "challenge_id": cid,
            "query_challenge_id": cid,
            "score_type": SCORE_GENUINE if source_state == target_state else SCORE_CROSS_STATE_CREDENTIAL,
            "attack_method": None,
            "representation": REPRESENTATION,
            "score": float(matched[i]),
            "is_genuine": source_state == target_state,
            "positive_class": POSITIVE_CLASS,
            "negative_class": SCORE_CROSS_STATE_CREDENTIAL,
            "score_direction": SCORE_DIRECTION,
        }
        for i, cid in enumerate(ids)
    ]

    summary = summary_base_fields(
        source_state=source_state,
        target_state=target_state,
        attack_method=None,
        comparison_type=COMPARISON_G_VS_X,
        n_positive=int(g_sum.get("n", 0)),
        n_negative=x_sum["n"],
        genuine_summary=g_sum if g.size else {"median": float("nan"), "q05": float("nan")},
        negative_summary=x_sum,
        robust_gap=rg_x,
        auc=auc_eer["auc"],
        eer=auc_eer["eer"],
        retrieval=retrieval,
    )
    summary.update(
        {
            "is_diagonal": source_state == target_state,
            "median_S_X": x_sum["median"],
            "mean_S_X": x_sum["mean"],
            "q05_S_X": x_sum["q05"],
            "q95_S_X": x_sum["q95"],
            "minimum_S_X": x_sum["minimum"],
            "maximum_S_X": x_sum["maximum"],
            "rg_cross_state_credential": rg_x,
            "auc_cross_state_credential": auc_eer["auc"],
            "eer_cross_state_credential": auc_eer["eer"],
            "negative_class": SCORE_CROSS_STATE_CREDENTIAL,
            "threshold_note": "PILOT_STATE_SPECIFIC_THRESHOLD; DESCRIPTIVE_ONLY; NOT_LIFECYCLE_TAU_G",
        }
    )
    summary.update(retrieval_meta(RETRIEVAL_CANDIDATE_COUNT))
    return {
        "challenge_ids": ids,
        "transfer_scores": matched,
        "transfer_summary": x_sum,
        "rg_cross_state_credential": rg_x,
        "retrieval": retrieval,
        "auc_eer": auc_eer,
        "summary_row": summary,
        "score_rows": score_rows,
    }


def evaluate_software_clone(
    predictions: dict[str, np.ndarray],
    queries: dict[str, np.ndarray],
    challenge_ids: list[str],
    *,
    device_id: str,
    source_state: str,
    target_state: str,
    attack_method: str,
    genuine_reference: np.ndarray | None,
) -> dict[str, Any]:
    """Track C/D: S_A^(m)(s→t) vs S_G(s)."""
    ids = [c for c in challenge_ids if c in predictions and c in queries]
    P = stack_dict(predictions, ids)
    Q = stack_dict(queries, ids)
    scores = row_zero_mean_ncc(P, Q)
    # Vectorized NMSE (same formula as metrics.nmse per row).
    q_c = Q - Q.mean(axis=1, keepdims=True)
    denom = np.sum(q_c * q_c, axis=1)
    num = np.sum((P - Q) ** 2, axis=1)
    nmse_vals = np.divide(num, denom, out=np.full_like(num, np.nan), where=denom > 0)
    retrieval = compute_retrieval_metrics(
        {cid: predictions[cid] for cid in ids},
        {cid: queries[cid] for cid in ids},
    )
    s_sum = score_distribution(scores)
    g = np.asarray(genuine_reference, dtype=np.float64) if genuine_reference is not None else np.array([])
    g_sum = score_distribution(g) if g.size else {"n": 0, "median": float("nan"), "q05": float("nan")}
    rg_a = robust_gap_q05_minus_q95(g, scores) if g.size else float("nan")
    auc_eer = auc_eer_genuine_vs_negative(g, scores) if g.size else {"auc": float("nan"), "eer": float("nan"), "positive_class": POSITIVE_CLASS, "score_direction": SCORE_DIRECTION}
    auc_eer["negative_class"] = SCORE_SOFTWARE_CLONE

    score_rows = [
        {
            "sample_id": f"{device_id}_{source_state}_clone_{attack_method}_{cid}__vs__{target_state}_B",
            "device_id": device_id,
            "source_state": source_state,
            "target_state": target_state,
            "challenge_id": cid,
            "query_challenge_id": cid,
            "score_type": SCORE_SOFTWARE_CLONE,
            "attack_method": attack_method,
            "representation": REPRESENTATION,
            "score": float(scores[i]),
            "is_genuine": False,
            "positive_class": POSITIVE_CLASS,
            "negative_class": SCORE_SOFTWARE_CLONE,
            "score_direction": SCORE_DIRECTION,
        }
        for i, cid in enumerate(ids)
    ]

    summary = summary_base_fields(
        source_state=source_state,
        target_state=target_state,
        attack_method=attack_method,
        comparison_type=COMPARISON_G_VS_A,
        n_positive=int(g_sum.get("n", 0)),
        n_negative=s_sum["n"],
        genuine_summary=g_sum if g.size else {"median": float("nan"), "q05": float("nan")},
        negative_summary=s_sum,
        robust_gap=rg_a,
        auc=auc_eer["auc"],
        eer=auc_eer["eer"],
        retrieval=retrieval,
    )
    pred_var = float(np.var(P, axis=0).mean()) if P.size else float("nan")
    meas_var = float(np.var(Q, axis=0).mean()) if Q.size else float("nan")
    summary.update(
        {
            "is_diagonal": source_state == target_state,
            "median_S_A": s_sum["median"],
            "mean_S_A": s_sum["mean"],
            "q05_S_A": s_sum["q05"],
            "q95_S_A": s_sum["q95"],
            "rg_software_clone": rg_a,
            "auc_software_clone": auc_eer["auc"],
            "eer_software_clone": auc_eer["eer"],
            "nmse_mean": float(np.nanmean(nmse_vals)) if nmse_vals.size else float("nan"),
            "nmse_median": float(np.nanmedian(nmse_vals)) if nmse_vals.size else float("nan"),
            "prediction_variance": pred_var,
            "measured_variance": meas_var,
            "variance_ratio": float(pred_var / meas_var) if meas_var and np.isfinite(meas_var) and meas_var > 0 else float("nan"),
            "negative_class": SCORE_SOFTWARE_CLONE,
            # auxiliary only — not a robust gap
            "scores": scores,
        }
    )
    summary.update(retrieval_meta(RETRIEVAL_CANDIDATE_COUNT))
    return {
        "challenge_ids": ids,
        "scores": scores,
        "score_summary": s_sum,
        "rg_software_clone": rg_a,
        "retrieval": retrieval,
        "auc_eer": auc_eer,
        "summary_row": summary,
        "score_rows": score_rows,
        "nmse_vals": nmse_vals,
        "prediction_variance": pred_var,
        "measured_variance": meas_var,
    }


def scalar_vs_batched_ncc_max_abs_error(a: np.ndarray, b: np.ndarray, *, n_pairs: int = 8) -> float:
    n = min(n_pairs, a.shape[0], b.shape[0])
    batched = row_zero_mean_ncc(a[:n], b[:n])
    scalar = np.array([float(_scalar_zero_mean_ncc(a[i], b[i])) for i in range(n)], dtype=np.float64)
    return float(np.max(np.abs(batched - scalar)))
