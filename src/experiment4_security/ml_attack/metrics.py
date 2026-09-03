"""Formal evaluation metrics: prediction NCC, NMSE, challenge retrieval,
genuine/challenge-mismatch/attack score distributions, robust gap, AUC/EER,
and the gated absolute-ASR computation.

Score naming (per protocol, replacing ambiguous "intra"/"inter"):
- S_G: genuine match      -- same state, same challenge, Round A vs Round B.
- S_C: challenge mismatch -- same state, different challenge.
- S_A: attack match       -- model-predicted response vs. its measured response.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from puf_common.metrics import auc_roc, equal_error_rate, robust_gap as _robust_gap
from puf_common.ncc import zero_mean_ncc


def prediction_ncc(predicted: np.ndarray, measured: np.ndarray) -> float:
    return float(zero_mean_ncc(predicted, measured))


def nmse(predicted: np.ndarray, measured: np.ndarray) -> float:
    measured = measured.astype(np.float64)
    predicted = predicted.astype(np.float64)
    denom = float(np.sum((measured - measured.mean()) ** 2))
    if denom <= 0:
        return float("nan")
    return float(np.sum((predicted - measured) ** 2) / denom)


def _center_normalize_rows(mat: np.ndarray) -> np.ndarray:
    m = mat.astype(np.float64)
    m = m - m.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return m / norms


def pairwise_zero_mean_ncc(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Vectorized zero-mean NCC between every row of A and every row of B."""
    An = _center_normalize_rows(A)
    Bn = _center_normalize_rows(B)
    return An @ Bn.T


def compute_retrieval_metrics(
    predicted_by_cid: dict[str, np.ndarray], candidate_by_cid: dict[str, np.ndarray]
) -> dict[str, Any]:
    """For each predicted response, rank its true challenge among ALL
    candidate measured responses in the target state (by NCC, descending)."""
    pred_ids = sorted(predicted_by_cid.keys())
    cand_ids = sorted(candidate_by_cid.keys())
    if not pred_ids or not cand_ids:
        return {"n": 0, "top1_rate": float("nan"), "top5_rate": float("nan"), "median_rank": float("nan"), "mean_reciprocal_rank": float("nan"), "per_challenge": []}

    P = np.stack([predicted_by_cid[c] for c in pred_ids], axis=0)
    C = np.stack([candidate_by_cid[c] for c in cand_ids], axis=0)
    sims = pairwise_zero_mean_ncc(P, C)  # (n_pred, n_cand)

    cand_index = {c: j for j, c in enumerate(cand_ids)}
    # Rank of true challenge: 1 + #candidates with strictly higher similarity
    # (ties broken consistently with argsort(-row) then .index — rare for float NCC).
    order = np.argsort(-sims, axis=1, kind="mergesort")  # stable
    ranks = []
    per_challenge = []
    for i, cid in enumerate(pred_ids):
        j = cand_index.get(cid)
        if j is None:
            continue
        rank = int(np.where(order[i] == j)[0][0]) + 1
        ranks.append(rank)
        per_challenge.append({"challenge_id": cid, "rank": rank, "top1": rank == 1, "top5": rank <= 5})

    if not ranks:
        return {"n": 0, "top1_rate": float("nan"), "top5_rate": float("nan"), "median_rank": float("nan"), "mean_reciprocal_rank": float("nan"), "per_challenge": []}

    ranks_arr = np.asarray(ranks, dtype=np.float64)
    return {
        "n": len(ranks),
        "top1_rate": float(np.mean(ranks_arr == 1)),
        "top5_rate": float(np.mean(ranks_arr <= 5)),
        "median_rank": float(np.median(ranks_arr)),
        "mean_reciprocal_rank": float(np.mean(1.0 / ranks_arr)),
        "per_challenge": per_challenge,
    }


def robust_gap_attack(genuine_scores: np.ndarray, attack_scores: np.ndarray) -> float:
    """RG_A = Q05(S_G) - Q95(S_A). Positive => attack scores' upper tail is
    still below the genuine-match lower tail."""
    return _robust_gap(np.asarray(genuine_scores), np.asarray(attack_scores))


def attack_auc_eer(genuine_scores: np.ndarray, attack_scores: np.ndarray) -> dict[str, Any]:
    g = np.asarray(genuine_scores, dtype=np.float64)
    a = np.asarray(attack_scores, dtype=np.float64)
    return {
        "positive_label": "genuine_match_S_G",
        "negative_label": "attack_match_S_A",
        "score_direction": "higher_ncc_is_more_similar",
        "attack_auc": float(auc_roc(g, a)) if g.size and a.size else float("nan"),
        "attack_eer": float(equal_error_rate(g, a)) if g.size and a.size else float("nan"),
    }


def score_space_matches_lifecycle(representation_space: str, lifecycle_calibrated_repr: str = "fullres_detail_cm") -> bool:
    """Absolute ASR@tau_G is only meaningful if the score space used here is
    identical to the one tau_G was calibrated on.

    The pilot's state-specific 128-challenge Round-A enrollment common
    (`fullres_detail_cm` / `fullres_detail_cm_enrollment`) is a *different
    protocol* from the lifecycle's detail_cm construction, so absolute ASR is
    refused for all current pilot spaces.
    """
    # Explicit deny-list for pilot representations (protocol mismatch).
    if representation_space in {
        "fullres_detail",
        "fullres_detail_cm",
        "fullres_detail_cm_enrollment",
        "fullres_detail_cm_train_frozen",
    }:
        return False
    return representation_space == lifecycle_calibrated_repr and False  # keep locked off

def compute_gated_absolute_asr(
    attack_scores: np.ndarray, tau_g: float, *, representation_space: str
) -> dict[str, Any]:
    matches = score_space_matches_lifecycle(representation_space)
    if not matches:
        return {
            "absolute_asr": None,
            "asr_status": "SCORE_SPACE_MISMATCH",
            "tau_g_used": None,
            "representation_space": representation_space,
            "note": (
                "Lifecycle tau_G was calibrated on the lifecycle's own "
                "fullres detail_cm (all-challenge common template per "
                "device/state/round group). This pilot's representation "
                "space does not match, so absolute ASR@tau_G is refused "
                "rather than silently mis-scaled."
            ),
        }
    scores = np.asarray(attack_scores, dtype=np.float64)
    if scores.size == 0:
        return {"absolute_asr": None, "asr_status": "NO_SCORES", "tau_g_used": tau_g, "representation_space": representation_space}
    asr = float(np.mean(scores >= tau_g))
    return {"absolute_asr": asr, "asr_status": "COMPUTED", "tau_g_used": float(tau_g), "representation_space": representation_space}
