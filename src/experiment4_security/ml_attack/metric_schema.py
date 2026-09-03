"""Canonical Experiment-4 metric symbols, code fields, and score-type enums.

Paper convention (must appear once in Methods / reports):
  - Green challenge-specific response: \\widetilde{\\mathbf G}
  - Subscript G in S_G means Genuine match, NOT Green.

Unified score types:
  GENUINE, CHALLENGE_MISMATCH, DEVICE_MISMATCH,
  CROSS_STATE_CREDENTIAL, SOFTWARE_CLONE

Unified robust gaps (all Q05(Genuine) - Q95(non-genuine)):
  RG_C, RG_D, RG_X, RG_A

Representation for this pilot:
  fullres_detail_cm_enrollment_frozen
"""

from __future__ import annotations

from typing import Any

REPRESENTATION = "fullres_detail_cm_enrollment_frozen"

SCORE_GENUINE = "GENUINE"
SCORE_CHALLENGE_MISMATCH = "CHALLENGE_MISMATCH"
SCORE_DEVICE_MISMATCH = "DEVICE_MISMATCH"
SCORE_CROSS_STATE_CREDENTIAL = "CROSS_STATE_CREDENTIAL"
SCORE_SOFTWARE_CLONE = "SOFTWARE_CLONE"

COMPARISON_G_VS_C = "G_VS_C"
COMPARISON_G_VS_D = "G_VS_D"
COMPARISON_G_VS_X = "G_VS_X"
COMPARISON_G_VS_A = "G_VS_A"

SCORE_DIRECTION = "HIGHER_IS_MORE_GENUINE"
POSITIVE_CLASS = "GENUINE"

RETRIEVAL_CANDIDATE_COUNT = 128
TOP1_CHANCE = 1.0 / 128.0
TOP5_CHANCE = 5.0 / 128.0

# Forbidden formal-output tokens (grep targets for audit).
FORBIDDEN_METRIC_TOKENS = (
    "prediction_ncc",
    "transfer_ncc",
    "RG_revoke",
    "RG_model_same",
    "RG_model_revoke",
    "attack_auc",
    "attack_eer",
    "transfer_auc",
    "transfer_eer",
    "intra_score",
    "inter_challenge",
    "inter_device",
    "NO_SAME_STATE_SIGNAL",
    "same_state_signal",
)

METRIC_DEFINITIONS: dict[str, dict[str, Any]] = {
    "S_G": {
        "symbol": "S_G",
        "full_name": "Genuine match",
        "code_field": "score_genuine",
        "score_type": SCORE_GENUINE,
        "positive_class": POSITIVE_CLASS,
        "negative_class": None,
        "formula": "NCC(G~^A_s,c, G~^{B|s}_s,c)",
        "applicable_tracks": ["A", "B", "C", "D"],
        "score_direction": SCORE_DIRECTION,
        "note": "Subscript G means Genuine, not Green.",
    },
    "S_C": {
        "symbol": "S_C",
        "full_name": "Challenge mismatch",
        "code_field": "score_challenge_mismatch",
        "score_type": SCORE_CHALLENGE_MISMATCH,
        "positive_class": POSITIVE_CLASS,
        "negative_class": SCORE_CHALLENGE_MISMATCH,
        "formula": "NCC(G~^A_s,c, G~^{B|s}_s,c') with c≠c'",
        "applicable_tracks": ["A"],
        "score_direction": SCORE_DIRECTION,
    },
    "S_D": {
        "symbol": "S_D",
        "full_name": "Device mismatch",
        "code_field": "score_device_mismatch",
        "score_type": SCORE_DEVICE_MISMATCH,
        "positive_class": POSITIVE_CLASS,
        "negative_class": SCORE_DEVICE_MISMATCH,
        "formula": "NCC(G~^A_i,s,c, G~^B_j,s,c) with i≠j",
        "applicable_tracks": [],
        "score_direction": SCORE_DIRECTION,
        "f01_status": "NOT_AVAILABLE_SINGLE_DEVICE",
    },
    "S_X": {
        "symbol": "S_X",
        "full_name": "Cross-state old-credential match",
        "code_field": "score_cross_state_credential",
        "score_type": SCORE_CROSS_STATE_CREDENTIAL,
        "positive_class": POSITIVE_CLASS,
        "negative_class": SCORE_CROSS_STATE_CREDENTIAL,
        "formula": "NCC(G~^A_s,c, G~^{B|s}_t,c) with s≠t",
        "applicable_tracks": ["B"],
        "score_direction": SCORE_DIRECTION,
    },
    "S_A": {
        "symbol": "S_A",
        "full_name": "Software-clone attack match",
        "code_field": "score_software_clone",
        "score_type": SCORE_SOFTWARE_CLONE,
        "positive_class": POSITIVE_CLASS,
        "negative_class": SCORE_SOFTWARE_CLONE,
        "formula": "NCC(Ĝ^(m)_s,c, G~^{B|s}_t,c)",
        "applicable_tracks": ["C", "D"],
        "score_direction": SCORE_DIRECTION,
    },
    "RG_C": {
        "symbol": "RG_C",
        "full_name": "Challenge robust gap",
        "code_field": "rg_challenge",
        "formula": "Q05(S_G) - Q95(S_C)",
        "applicable_tracks": ["A"],
        "score_direction": SCORE_DIRECTION,
    },
    "RG_D": {
        "symbol": "RG_D",
        "full_name": "Device robust gap",
        "code_field": "rg_device",
        "formula": "Q05(S_G) - Q95(S_D)",
        "applicable_tracks": [],
        "f01_status": "NOT_AVAILABLE_SINGLE_DEVICE",
    },
    "RG_X": {
        "symbol": "RG_X",
        "full_name": "Cross-state credential robust gap",
        "code_field": "rg_cross_state_credential",
        "formula": "Q05(S_G(s)) - Q95(S_X(s→t))",
        "applicable_tracks": ["B"],
    },
    "RG_A": {
        "symbol": "RG_A",
        "full_name": "Software-clone robust gap",
        "code_field": "rg_software_clone",
        "formula": "Q05(S_G(s)) - Q95(S_A^(m)(s→t))",
        "applicable_tracks": ["C", "D"],
    },
    "AUC_C": {"symbol": "AUC_C", "code_field": "auc_challenge", "comparison_type": COMPARISON_G_VS_C},
    "EER_C": {"symbol": "EER_C", "code_field": "eer_challenge", "comparison_type": COMPARISON_G_VS_C},
    "AUC_X": {"symbol": "AUC_X", "code_field": "auc_cross_state_credential", "comparison_type": COMPARISON_G_VS_X},
    "EER_X": {"symbol": "EER_X", "code_field": "eer_cross_state_credential", "comparison_type": COMPARISON_G_VS_X},
    "AUC_A": {"symbol": "AUC_A", "code_field": "auc_software_clone", "comparison_type": COMPARISON_G_VS_A},
    "EER_A": {"symbol": "EER_A", "code_field": "eer_software_clone", "comparison_type": COMPARISON_G_VS_A},
}

NAME_MAPPING = [
    {"old_name": "prediction_ncc", "new_name": "score_software_clone", "meaning": "S_A software-clone NCC"},
    {"old_name": "transfer_ncc", "new_name": "score_cross_state_credential", "meaning": "S_X when object is old template DB"},
    {"old_name": "RG_revoke", "new_name": "rg_cross_state_credential", "meaning": "RG_X"},
    {"old_name": "RG_model_same", "new_name": "rg_software_clone", "meaning": "RG_A (same or cross via source/target)"},
    {"old_name": "RG_model_revoke", "new_name": "rg_software_clone", "meaning": "RG_A"},
    {"old_name": "attack_auc", "new_name": "auc_software_clone", "meaning": "AUC_A"},
    {"old_name": "attack_eer", "new_name": "eer_software_clone", "meaning": "EER_A"},
    {"old_name": "transfer_auc", "new_name": "auc_cross_state_credential", "meaning": "AUC_X"},
    {"old_name": "transfer_eer", "new_name": "eer_cross_state_credential", "meaning": "EER_A"},
    {"old_name": "RG_C (unchanged symbol)", "new_name": "rg_challenge", "meaning": "code field for RG_C"},
    {"old_name": "intra", "new_name": "GENUINE / score_genuine", "meaning": "S_G"},
    {"old_name": "inter_challenge", "new_name": "CHALLENGE_MISMATCH", "meaning": "S_C"},
    {"old_name": "fullres_detail_cm / detail_cm_train_frozen", "new_name": "fullres_detail_cm_enrollment_frozen", "meaning": "enrollment common representation"},
    {"old_name": "same_to_cross_drop", "new_name": "delta_clone_same_to_cross", "meaning": "auxiliary median(S_A same)-median(S_A cross); not a robust gap"},
]


def retrieval_meta(n_candidates: int = RETRIEVAL_CANDIDATE_COUNT) -> dict[str, float | int]:
    return {
        "retrieval_candidate_count": int(n_candidates),
        "top1_chance": float(1.0 / n_candidates) if n_candidates else float("nan"),
        "top5_chance": float(min(5, n_candidates) / n_candidates) if n_candidates else float("nan"),
    }


def summary_base_fields(
    *,
    source_state: str,
    target_state: str,
    attack_method: str | None,
    comparison_type: str,
    n_positive: int,
    n_negative: int,
    genuine_summary: dict[str, float],
    negative_summary: dict[str, float],
    robust_gap: float,
    auc: float,
    eer: float,
    retrieval: dict[str, Any],
) -> dict[str, Any]:
    meta = retrieval_meta(int(retrieval.get("n", RETRIEVAL_CANDIDATE_COUNT) or RETRIEVAL_CANDIDATE_COUNT))
    # Prefer fixed bank size of 128 for formal reporting when retrieval used full bank.
    if int(retrieval.get("n", 0) or 0) == RETRIEVAL_CANDIDATE_COUNT:
        meta = retrieval_meta(RETRIEVAL_CANDIDATE_COUNT)
    return {
        "source_state": source_state,
        "target_state": target_state,
        "attack_method": attack_method,
        "representation": REPRESENTATION,
        "comparison_type": comparison_type,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "median_genuine": genuine_summary.get("median"),
        "q05_genuine": genuine_summary.get("q05"),
        "median_negative": negative_summary.get("median"),
        "q95_negative": negative_summary.get("q95"),
        "robust_gap": robust_gap,
        "auc": auc,
        "eer": eer,
        "top1": retrieval.get("top1_rate"),
        "top5": retrieval.get("top5_rate"),
        "median_rank": retrieval.get("median_rank"),
        "mean_reciprocal_rank": retrieval.get("mean_reciprocal_rank"),
        **meta,
        "positive_class": POSITIVE_CLASS,
        "score_direction": SCORE_DIRECTION,
        "device_mismatch_status": "NOT_AVAILABLE_SINGLE_DEVICE",
        "rg_device": None,
        "rg_device_status": "NOT_AVAILABLE_SINGLE_DEVICE",
    }
