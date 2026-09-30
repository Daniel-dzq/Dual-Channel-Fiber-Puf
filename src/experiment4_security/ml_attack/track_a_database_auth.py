"""Track A — Same-state registered database authentication (S_G vs S_C, RG_C)."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from experiment4_security.ml_attack.batch_eval import evaluate_database_authentication
from experiment4_security.ml_attack.enrollment import EnrollmentCommon, enrollment_templates, query_vectors

logger = logging.getLogger(__name__)


def database_authentication_conclusion(rg_c: float, top1: float, eer: float) -> str:
    """Unit quality class of a device-state unit (Fig. 7a): Valid / Partial / Failed.

    Valid requires RG_C >= 0.05, Top-1 >= 0.90 and EER <= 0.10 over the 128 challenges.
    This rule is independent of the threshold-development operating point (T_G, n_req).
    """
    if not np.isfinite(rg_c) or not np.isfinite(top1) or not np.isfinite(eer):
        return "DATABASE_AUTHENTICATION_FAILED"
    if rg_c >= 0.05 and top1 >= 0.90 and eer <= 0.10:
        return "DATABASE_AUTHENTICATION_VALID"
    if rg_c > 0.0 and top1 >= 0.50:
        return "DATABASE_AUTHENTICATION_PARTIAL"
    return "DATABASE_AUTHENTICATION_FAILED"


def run_track_a(
    states: list[str],
    challenge_ids: list[str],
    commons: dict[str, EnrollmentCommon],
    *,
    detail_lookup: Callable[[str, str, str], np.ndarray],
    device_id: str = "F01",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    genuine_by_state: dict[str, np.ndarray] = {}
    t0 = time.time()

    for i, state in enumerate(states, start=1):
        t_state = time.time()
        logger.info("[Track A] state %s %d/%d", state, i, len(states))
        common = commons[state]
        templates = enrollment_templates(common, challenge_ids, detail_lookup=detail_lookup)
        queries = query_vectors(common, state, challenge_ids, detail_lookup=detail_lookup)
        ev = evaluate_database_authentication(
            templates, queries, challenge_ids, device_id=device_id, state_id=state
        )
        genuine_by_state[state] = ev["genuine_scores"]
        row = dict(ev["summary_row"])
        row["state_conclusion"] = database_authentication_conclusion(ev["rg_challenge"], row["top1"], row["eer_challenge"])
        # Explicit narrative fields
        row["interpretation"] = (
            "Genuine match is separated from Challenge mismatch."
            if row["state_conclusion"] == "DATABASE_AUTHENTICATION_VALID"
            else "Genuine vs Challenge mismatch separation incomplete."
        )
        summary_rows.append(row)
        score_rows.extend(ev["score_rows"])
        logger.info(
            "[Track A] state %s done in %.1fs | median_S_G=%.4f q05_S_G=%.4f median_S_C=%.4f "
            "q95_S_C=%.4f RG_C=%.4f AUC_C=%.4f EER_C=%.4f Top1=%.3f",
            state,
            time.time() - t_state,
            row["median_S_G"],
            row["q05_S_G"],
            row["median_S_C"],
            row["q95_S_C"],
            row["rg_challenge"],
            row["auc_challenge"],
            row["eer_challenge"],
            row["top1"],
        )

    summary = pd.DataFrame(summary_rows)
    scores = pd.DataFrame(score_rows)
    if (summary["state_conclusion"] == "DATABASE_AUTHENTICATION_VALID").all():
        overall = "DATABASE_AUTHENTICATION_VALID"
    elif (summary["state_conclusion"] == "DATABASE_AUTHENTICATION_FAILED").all():
        overall = "DATABASE_AUTHENTICATION_FAILED"
    else:
        overall = "DATABASE_AUTHENTICATION_PARTIAL"
    meta = {"overall_conclusion": overall, "elapsed_s": time.time() - t0, "genuine_by_state": genuine_by_state}
    logger.info("[Track A] done overall=%s elapsed=%.1fs", overall, meta["elapsed_s"])
    return summary, scores, meta
