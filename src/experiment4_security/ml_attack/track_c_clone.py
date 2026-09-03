"""Track C — same-state response reconstruction (S_intra vs S_A, RG_A).

Exact replay is a digital enrollment-database replay, not a physical clone.
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from experiment4_security.ml_attack.baselines import fit_exact_template_replay, fit_mean_baseline
from experiment4_security.ml_attack.batch_eval import evaluate_software_clone, row_zero_mean_ncc, stack_dict
from experiment4_security.ml_attack.config import ModelsConfig
from experiment4_security.ml_attack.enrollment import EnrollmentCommon, enrollment_templates, query_vectors
from experiment4_security.ml_attack.models import (
    fit_enrollment_representation,
    fit_kernel_ridge_fixed,
    fit_mlp_fixed,
    fit_rff_ridge_fixed,
    fit_ridge_fixed,
)

logger = logging.getLogger(__name__)

ATTACK_METHOD_MAP = {
    "C0_mean_response": "mean_response",
    "C1_exact_template_replay": "exact_template_replay",
    "C2_ridge_clone": "ridge_clone",
    "C3_kernel_ridge_clone": "kernel_ridge_clone",
    "C4_random_fourier_ridge_clone": "random_fourier_ridge_clone",
    "C5_small_mlp_clone": "small_mlp_clone",
}


def _predict_all(predictor, features, challenge_ids, model_key: str) -> dict[str, np.ndarray]:
    if model_key == "C0_mean_response":
        return {cid: predictor.predict(features[cid]) for cid in challenge_ids}
    if model_key == "C1_exact_template_replay":
        return {cid: predictor.predict_by_challenge_id(cid) for cid in challenge_ids}
    if hasattr(predictor, "predict_batch"):
        return predictor.predict_batch(features, challenge_ids)
    return {cid: predictor.predict(features[cid]) for cid in challenge_ids}


def _residual_audit(predictions, queries, challenge_ids, pca_mean):
    ids = challenge_ids
    P = stack_dict(predictions, ids)
    Q = stack_dict(queries, ids)
    full_ncc = float(np.mean(row_zero_mean_ncc(P, Q)))
    if pca_mean is None:
        return {
            "residual_ncc": float("nan"),
            "pca_mean_ncc": float("nan"),
            "prediction_to_pca_mean_ncc": float("nan"),
            "predicted_residual_energy": float("nan"),
            "measured_residual_energy": float("nan"),
            "residual_energy_ratio": float("nan"),
            "score_software_clone_full": full_ncc,
        }
    mean = pca_mean.astype(np.float64)
    P_res = P - mean
    Q_res = Q - mean
    return {
        "score_software_clone_full": full_ncc,
        "pca_mean_ncc": float(np.mean(row_zero_mean_ncc(np.broadcast_to(mean, P.shape), Q))),
        "residual_ncc": float(np.mean(row_zero_mean_ncc(P_res, Q_res))),
        "prediction_to_pca_mean_ncc": float(np.mean(row_zero_mean_ncc(P, np.broadcast_to(mean, P.shape)))),
        "predicted_residual_energy": float(np.mean(P_res ** 2)),
        "measured_residual_energy": float(np.mean(Q_res ** 2)),
        "residual_energy_ratio": float(np.mean(P_res ** 2) / max(np.mean(Q_res ** 2), 1e-12)),
    }


def run_track_c(
    states: list[str],
    challenge_ids: list[str],
    commons: dict[str, EnrollmentCommon],
    challenge_features: dict[str, np.ndarray],
    *,
    detail_lookup: Callable[[str, str, str], np.ndarray],
    models_cfg: ModelsConfig,
    pca_dimension: int,
    genuine_by_state: dict[str, np.ndarray],
    device_id: str = "F01",
    checkpoint_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    dominance_rows: list[dict[str, Any]] = []
    models_by_state: dict[str, dict[str, Any]] = {}
    t0 = time.time()
    spill = checkpoint_dir is not None
    if spill:
        from experiment4_security.identity_credential.cd_checkpoint import save_track_c_state_predictors

    for state in states:
        logger.info("[Track C] state %s", state)
        templates = enrollment_templates(commons[state], challenge_ids, detail_lookup=detail_lookup)
        queries = query_vectors(commons[state], state, challenge_ids, detail_lookup=detail_lookup)
        g_ref = genuine_by_state.get(state)
        bundle = fit_enrollment_representation(templates, pca_dimension=pca_dimension)
        pca_mean = bundle.mu

        predictors: dict[str, Any] = {}
        if models_cfg.mean_baseline:
            predictors["C0_mean_response"] = fit_mean_baseline(templates)
        if models_cfg.exact_template_replay:
            predictors["C1_exact_template_replay"] = fit_exact_template_replay(templates)
        if models_cfg.ridge:
            predictors["C2_ridge_clone"] = fit_ridge_fixed(challenge_features, templates, bundle, models_cfg)
        if models_cfg.kernel_ridge:
            predictors["C3_kernel_ridge_clone"] = fit_kernel_ridge_fixed(
                challenge_features, templates, bundle, models_cfg
            )
        if models_cfg.random_fourier_ridge:
            predictors["C4_random_fourier_ridge_clone"] = fit_rff_ridge_fixed(
                challenge_features, templates, bundle, models_cfg
            )
        if models_cfg.small_mlp:
            predictors["C5_small_mlp_clone"] = fit_mlp_fixed(challenge_features, templates, bundle, models_cfg)

        for model_key, pred in predictors.items():
            attack_method = ATTACK_METHOD_MAP[model_key]
            logger.info("[Track C] attack_method=%s on Round B (same state)", attack_method)
            preds = _predict_all(pred, challenge_features, challenge_ids, model_key)
            ev = evaluate_software_clone(
                preds,
                queries,
                challenge_ids,
                device_id=device_id,
                source_state=state,
                target_state=state,
                attack_method=attack_method,
                genuine_reference=g_ref,
            )
            row = dict(ev["summary_row"])
            # source validity for challenge-specific clones
            source_valid = bool(
                np.isfinite(row["median_S_A"]) and row["median_S_A"] >= 0.5 and row["top1"] >= 0.5
            )
            if attack_method == "exact_template_replay":
                source_valid = bool(np.isfinite(row["median_S_A"]) and row["median_S_A"] >= 0.5)
            if attack_method == "mean_response":
                source_valid = False
            row["source_state_valid"] = source_valid
            row["source_validity_note"] = "OK" if source_valid else "MODEL_NOT_VALID_IN_SOURCE_STATE"
            row["interpretation"] = (
                "The software clone reproduces the enrolled credential in its source state."
                if source_valid
                else "Software-clone attack scores do not reach the Genuine distribution in source state."
            )

            dom = _residual_audit(preds, queries, challenge_ids, pca_mean)
            dominated = (
                np.isfinite(dom["prediction_to_pca_mean_ncc"])
                and dom["prediction_to_pca_mean_ncc"] > 0.95
                and (not np.isfinite(dom["residual_ncc"]) or dom["residual_ncc"] < 0.2)
            )
            row["residual_ncc"] = dom["residual_ncc"]
            row["common_dominance_flag"] = (
                "HIGH_NCC_DOMINATED_BY_COMMON_RESPONSE" if dominated else "OK"
            )
            dominance_rows.append(
                {"source_state": state, "attack_method": attack_method, **dom, "flag": row["common_dominance_flag"]}
            )
            # drop internal array from summary
            row.pop("scores", None)
            summary_rows.append(row)
            score_rows.extend(ev["score_rows"])
            logger.info(
                "[Track C] %s/%s median_S_A=%.4f RG_A=%.4f AUC_A=%.4f EER_A=%.4f Top1=%.3f residual_ncc=%.4f",
                state,
                attack_method,
                row["median_S_A"],
                row["rg_software_clone"],
                row["auc_software_clone"],
                row["eer_software_clone"],
                row["top1"],
                row["residual_ncc"] if np.isfinite(row.get("residual_ncc", float("nan"))) else float("nan"),
            )
            del preds

        if spill:
            # Persist this state then free ~1–2GB before fitting the next state.
            save_track_c_state_predictors(checkpoint_dir, device_id, state, predictors)
            models_by_state[state] = {"predictors": None, "lazy_checkpoint": True}
            del templates, queries, predictors, bundle
            gc.collect()
        else:
            models_by_state[state] = {"predictors": predictors, "bundle": bundle, "templates": templates}

    summary = pd.DataFrame(summary_rows)
    scores = pd.DataFrame(score_rows)
    dominance = pd.DataFrame(dominance_rows)
    meta = {
        "models_by_state": models_by_state,
        "genuine_by_state": genuine_by_state,
        "elapsed_s": time.time() - t0,
        "note": "No frozen_model_name; all attack methods reported. Round B not used for selection.",
        "lazy_predictors": spill,
    }
    logger.info("[Track C] done elapsed=%.1fs spill_per_state=%s", meta["elapsed_s"], spill)
    return summary, scores, dominance, meta
