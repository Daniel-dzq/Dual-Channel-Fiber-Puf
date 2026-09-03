"""Track D — cross-state transfer of frozen source-state reconstructions (S_intra vs S_A, RG_A).

Performance notes (logic unchanged):
- Preload all Round-B detail vectors once (avoid repeated disk loads).
- Build Q_{t|s} once per (source, target), then score all models against it.
- Parallelize over targets (numpy BLAS releases GIL), same pattern as Track B.
"""

from __future__ import annotations

import gc
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from experiment4_security.ml_attack.batch_eval import evaluate_software_clone
from experiment4_security.ml_attack.enrollment import EnrollmentCommon
from experiment4_security.ml_attack.track_c_clone import ATTACK_METHOD_MAP, _predict_all

logger = logging.getLogger(__name__)

TRANSFER_MODEL_KEYS = [
    "C1_exact_template_replay",
    "C2_ridge_clone",
    "C3_kernel_ridge_clone",
    "C4_random_fourier_ridge_clone",
    "C5_small_mlp_clone",
]


def _preload_round_b_details(
    states: list[str],
    challenge_ids: list[str],
    detail_lookup: Callable[[str, str, str], np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    """Load detail(B_t, c) once per (target, challenge)."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for state in states:
        t0 = time.time()
        out[state] = {cid: detail_lookup(state, "B", cid) for cid in challenge_ids}
        logger.info("[Track D] preloaded Round-B details for %s (n=%d) in %.1fs", state, len(challenge_ids), time.time() - t0)
    return out


def _queries_from_preloaded(
    source_common: EnrollmentCommon,
    target_state: str,
    challenge_ids: list[str],
    detail_b: dict[str, dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Q_t,c|s = detail(B_t,c) - common_s_A (identical to enrollment.query_vectors)."""
    b = detail_b[target_state]
    return {cid: source_common.to_detail_cm(b[cid]) for cid in challenge_ids}


def run_track_d(
    states: list[str],
    challenge_ids: list[str],
    commons: dict[str, EnrollmentCommon],
    challenge_features: dict[str, np.ndarray],
    *,
    detail_lookup: Callable[[str, str, str], np.ndarray],
    track_c_meta: dict[str, Any],
    track_c_summary: pd.DataFrame,
    device_id: str = "F01",
    n_parallel_targets: int = 4,
    checkpoint_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    from experiment4_security.identity_credential.cd_checkpoint import (
        load_track_c_predictors_state,
        load_track_d_partial,
        save_track_d_partial,
    )

    models_by_state = track_c_meta["models_by_state"]
    lazy_predictors = bool(track_c_meta.get("lazy_predictors")) or any(
        bool(models_by_state.get(s, {}).get("lazy_checkpoint"))
        or models_by_state.get(s, {}).get("predictors") is None
        for s in states
        if s in models_by_state
    )
    summary_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    sources_done: list[str] = []
    t0 = time.time()

    if checkpoint_dir is not None:
        summary_rows, score_rows, sources_done = load_track_d_partial(checkpoint_dir, device_id)

    same_median = {}
    same_q05 = {}
    source_valid = {}
    for _, r in track_c_summary.iterrows():
        key = (r["source_state"], r["attack_method"])
        same_median[key] = r["median_S_A"]
        same_q05[key] = r["q05_S_A"] if "q05_S_A" in r else r.get("q05")
        v = r["source_state_valid"]
        if isinstance(v, str):
            source_valid[key] = v.strip().lower() in {"1", "true", "t", "yes"}
        else:
            source_valid[key] = bool(v)

    detail_b = _preload_round_b_details(states, challenge_ids, detail_lookup)
    n_workers = max(1, int(n_parallel_targets))
    logger.info("[Track D] parallel_targets=%d lazy_predictors=%s", n_workers, lazy_predictors)

    total_cells = len(states) * len(states) * len(TRANSFER_MODEL_KEYS)
    done_cells = len(summary_rows)
    if sources_done:
        logger.info(
            "[Track D] resume: skipping sources %s | progress %d/%d",
            ",".join(sources_done),
            done_cells,
            total_cells,
        )

    for si, source in enumerate(states, start=1):
        if source in sources_done:
            logger.info("[Track D] source %s %d/%d already checkpointed — skip", source, si, len(states))
            continue
        t_src = time.time()

        # Load only this source's predictors (avoids holding all 8 states ≈ jetsam).
        if lazy_predictors:
            if checkpoint_dir is None:
                raise RuntimeError("lazy Track C predictors require checkpoint_dir")
            predictors = load_track_c_predictors_state(checkpoint_dir, device_id, source)
        else:
            predictors = models_by_state[source]["predictors"]
        g_ref = track_c_meta.get("genuine_by_state", {}).get(source)

        # Predict once per model for this source (shared across all targets).
        preds_by_model: dict[str, dict[str, np.ndarray]] = {}
        for model_key in TRANSFER_MODEL_KEYS:
            if model_key not in predictors:
                continue
            preds_by_model[model_key] = _predict_all(
                predictors[model_key], challenge_features, challenge_ids, model_key
            )
        # Predictors no longer needed after materializing source predictions.
        del predictors
        gc.collect()
        logger.info(
            "[Track D] source %s %d/%d | models=%d",
            source,
            si,
            len(states),
            len(preds_by_model),
        )

        def _one_target(target: str) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
            queries = _queries_from_preloaded(commons[source], target, challenge_ids, detail_b)
            out: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
            for model_key, preds in preds_by_model.items():
                attack_method = ATTACK_METHOD_MAP[model_key]
                ev = evaluate_software_clone(
                    preds,
                    queries,
                    challenge_ids,
                    device_id=device_id,
                    source_state=source,
                    target_state=target,
                    attack_method=attack_method,
                    genuine_reference=g_ref,
                )
                row = dict(ev["summary_row"])
                row.pop("scores", None)
                key = (source, attack_method)
                valid = source_valid.get(key, False)
                row["source_state_valid"] = valid
                row["validity_note"] = "OK" if valid else "MODEL_NOT_VALID_IN_SOURCE_STATE"
                med_same = same_median.get(key, float("nan"))
                row["delta_clone_same_to_cross"] = (
                    float(med_same - row["median_S_A"])
                    if np.isfinite(med_same) and np.isfinite(row["median_S_A"]) and source != target
                    else (0.0 if source == target else float("nan"))
                )
                if valid:
                    row["interpretation"] = (
                        "The software clone reproduces the enrolled credential in its source state "
                        "but does not transfer across mechanical reconfiguration."
                        if source != target
                        and np.isfinite(row["rg_software_clone"])
                        and row["rg_software_clone"] > 0
                        else (
                            "Software-clone attack scores do not reach the Genuine distribution."
                            if np.isfinite(row["rg_software_clone"]) and row["rg_software_clone"] > 0
                            else "Cross-state software-clone scores overlap Genuine tail."
                        )
                    )
                else:
                    row["interpretation"] = "MODEL_NOT_VALID_IN_SOURCE_STATE — cannot discuss revocation."
                out.append((row, ev["score_rows"]))
            return out

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_one_target, t): t for t in states}
            for fut in as_completed(futs):
                target = futs[fut]
                for row, scores in fut.result():
                    summary_rows.append(row)
                    score_rows.extend(scores)
                    done_cells += 1
                logger.info(
                    "[Track D] %s→%s done | progress %d/%d",
                    source,
                    target,
                    done_cells,
                    total_cells,
                )
        del preds_by_model
        if source in models_by_state:
            models_by_state[source]["predictors"] = None
            models_by_state[source].pop("templates", None)
            models_by_state[source].pop("bundle", None)
        gc.collect()
        sources_done.append(source)
        if checkpoint_dir is not None:
            save_track_d_partial(
                checkpoint_dir,
                device_id,
                summary_rows=summary_rows,
                score_rows=score_rows,
                sources_done=sources_done,
            )
        logger.info("[Track D] source %s completed in %.1fs", source, time.time() - t_src)

    del detail_b
    gc.collect()

    # Stable order (parallel completion is nondeterministic).
    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["source_state", "attack_method", "target_state"])
        .reset_index(drop=True)
    )
    scores = pd.DataFrame(score_rows)
    if not scores.empty and {"source_state", "attack_method", "target_state", "challenge_id"}.issubset(scores.columns):
        scores = scores.sort_values(
            ["source_state", "attack_method", "target_state", "challenge_id"]
        ).reset_index(drop=True)

    conclusions = []
    for model_key in TRANSFER_MODEL_KEYS:
        attack_method = ATTACK_METHOD_MAP[model_key]
        sub = summary[(summary["attack_method"] == attack_method) & (summary["source_state_valid"])]
        if sub.empty:
            conclusions.append({"attack_method": attack_method, "conclusion": "MODEL_NOT_VALID_IN_SOURCE_STATE"})
            continue
        diag = sub[sub["is_diagonal"]]["median_S_A"].median()
        off = sub[~sub["is_diagonal"]]["median_S_A"].median()
        drop = float(diag - off) if np.isfinite(diag) and np.isfinite(off) else float("nan")
        if np.isfinite(drop) and drop >= 0.15 and (not np.isfinite(off) or off < 0.4):
            conclusions.append(
                {
                    "attack_method": attack_method,
                    "conclusion": "SOFTWARE_CLONE_VALID_SAME_STATE_ONLY",
                    "delta_clone_same_to_cross": drop,
                }
            )
        elif np.isfinite(off) and off >= 0.5:
            conclusions.append(
                {
                    "attack_method": attack_method,
                    "conclusion": "SOFTWARE_CLONE_RECONFIGURATION_RISK",
                    "delta_clone_same_to_cross": drop,
                }
            )
        else:
            conclusions.append(
                {
                    "attack_method": attack_method,
                    "conclusion": "PARTIAL_CROSS_STATE_CLONE_TRANSFER",
                    "delta_clone_same_to_cross": drop,
                }
            )

    meta = {
        "per_model_conclusions": conclusions,
        "elapsed_s": time.time() - t0,
        "absolute_asr": None,
        "asr_status": "SCORE_SPACE_OR_PROTOCOL_MISMATCH",
        "n_parallel_targets": n_workers,
    }
    logger.info("[Track D] done elapsed=%.1fs", meta["elapsed_s"])
    return summary, scores, meta
