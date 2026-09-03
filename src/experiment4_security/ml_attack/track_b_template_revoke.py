"""Track B — Cross-state revocation of registered credentials (S_G vs S_X, RG_X)."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import numpy as np
import pandas as pd

from experiment4_security.ml_attack.batch_eval import evaluate_cross_state_credential
from experiment4_security.ml_attack.enrollment import EnrollmentCommon, enrollment_templates, query_vectors

logger = logging.getLogger(__name__)


def run_track_b(
    states: list[str],
    challenge_ids: list[str],
    commons: dict[str, EnrollmentCommon],
    *,
    detail_lookup: Callable[[str, str, str], np.ndarray],
    genuine_by_state: dict[str, np.ndarray],
    device_id: str = "F01",
    n_parallel_targets: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    t0 = time.time()

    for si, source in enumerate(states, start=1):
        t_src = time.time()
        logger.info("[Track B] source %s %d/%d", source, si, len(states))
        templates = enrollment_templates(commons[source], challenge_ids, detail_lookup=detail_lookup)
        g_same = genuine_by_state.get(source)

        def _one_target(target: str) -> dict[str, Any]:
            queries = query_vectors(commons[source], target, challenge_ids, detail_lookup=detail_lookup)
            return evaluate_cross_state_credential(
                templates,
                queries,
                challenge_ids,
                device_id=device_id,
                source_state=source,
                target_state=target,
                genuine_same_state=g_same,
            )

        # Thread pool: numpy releases GIL in BLAS; good for overlapping I/O+NCC on M4 Pro.
        with ThreadPoolExecutor(max_workers=max(1, n_parallel_targets)) as ex:
            futs = {ex.submit(_one_target, t): t for t in states}
            for fut in as_completed(futs):
                target = futs[fut]
                ev = fut.result()
                row = dict(ev["summary_row"])
                row["interpretation"] = (
                    "Cross-state old-credential scores remain separated from Genuine scores."
                    if np.isfinite(row["rg_cross_state_credential"]) and row["rg_cross_state_credential"] > 0
                    else "Cross-state old-credential scores overlap Genuine tail."
                )
                summary_rows.append(row)
                score_rows.extend(ev["score_rows"])
                logger.info(
                    "[Track B] %s→%s median_S_X=%.4f q95_S_X=%.4f RG_X=%.4f AUC_X=%.4f EER_X=%.4f Top1=%.3f",
                    source,
                    target,
                    row["median_S_X"],
                    row["q95_S_X"],
                    row["rg_cross_state_credential"],
                    row["auc_cross_state_credential"],
                    row["eer_cross_state_credential"],
                    row["top1"],
                )
        logger.info("[Track B] source %s completed in %.1fs", source, time.time() - t_src)

    summary = pd.DataFrame(summary_rows).sort_values(["source_state", "target_state"]).reset_index(drop=True)
    scores = pd.DataFrame(score_rows)

    diag = summary[summary["is_diagonal"]]
    off = summary[~summary["is_diagonal"]]
    diagonal_median = float(diag["median_S_X"].median()) if not diag.empty else float("nan")
    off_diagonal_median = float(off["median_S_X"].median()) if not off.empty else float("nan")
    worst_off = off.loc[off["q95_S_X"].idxmax()] if not off.empty else None
    min_rg = float(off["rg_cross_state_credential"].min()) if not off.empty and off["rg_cross_state_credential"].notna().any() else float("nan")

    if (
        np.isfinite(diagonal_median)
        and np.isfinite(off_diagonal_median)
        and diagonal_median - off_diagonal_median >= 0.15
        and (not np.isfinite(min_rg) or min_rg > 0.0)
    ):
        conclusion = "STATE_BOUND_CREDENTIALS_CONFIRMED"
        if np.isfinite(min_rg) and min_rg > 0.05:
            conclusion = "OLD_TEMPLATE_DATABASE_REVOKED"
    elif np.isfinite(off_diagonal_median) and off_diagonal_median > 0.5:
        conclusion = "CREDENTIAL_RECONFIGURATION_NOT_SECURE"
    elif np.isfinite(off_diagonal_median) and off_diagonal_median > 0.3:
        conclusion = "PARTIAL_CROSS_STATE_CREDENTIAL_TRANSFER"
    else:
        conclusion = "OLD_TEMPLATE_DATABASE_REVOKED"

    meta = {
        "diagonal_median": diagonal_median,
        "off_diagonal_median": off_diagonal_median,
        "minimum_rg_cross_state_credential": min_rg,
        "worst_off_diagonal_q95": float(worst_off["q95_S_X"]) if worst_off is not None else float("nan"),
        "source_state_worst_target": (
            f"{worst_off['source_state']}->{worst_off['target_state']}" if worst_off is not None else None
        ),
        "conclusion": conclusion,
        "elapsed_s": time.time() - t0,
        "absolute_asr": None,
        "asr_status": "SCORE_SPACE_OR_PROTOCOL_MISMATCH",
    }
    logger.info(
        "[Track B] done diag_med=%.4f off_med=%.4f min_RG_X=%.4f conclusion=%s",
        diagonal_median,
        off_diagonal_median,
        min_rg,
        conclusion,
    )
    return summary, scores, meta
