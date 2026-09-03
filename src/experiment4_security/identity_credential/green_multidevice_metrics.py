"""Multi-device green metrics: true S_D / RG_D across devices (same state+challenge)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from experiment4_security.ml_attack.batch_eval import (
    auc_eer_genuine_vs_negative,
    robust_gap_q05_minus_q95,
    row_zero_mean_ncc,
    score_distribution,
)
from experiment4_security.ml_attack.enrollment import EnrollmentCommon, enrollment_templates, query_vectors
from experiment4_security.ml_attack.metric_schema import (
    SCORE_DEVICE_MISMATCH,
    SCORE_GENUINE,
)


def evaluate_device_mismatch_for_state(
    *,
    state_id: str,
    challenge_ids: list[str],
    devices: list[str],
    commons: dict[tuple[str, str], EnrollmentCommon],
    detail_lookups: dict[str, Callable[[str, str, str], np.ndarray]],
    genuine_by_device: dict[str, np.ndarray],
) -> dict[str, Any]:
    """S_D: template of device d vs query of device e≠d, same state, matched challenge_id.

    Uses source-device enrollment common on source templates; queries use
    target Round B minus *source* common (same cross-entity convention as S_X).
    """
    rows = []
    scores_s_d: list[float] = []
    for src in devices:
        templates = enrollment_templates(
            commons[(src, state_id)], challenge_ids, detail_lookup=detail_lookups[src]
        )
        g = genuine_by_device.get(src)
        for tgt in devices:
            if tgt == src:
                continue
            # Q on target Round B with SOURCE common
            queries = {
                cid: commons[(src, state_id)].to_detail_cm(
                    detail_lookups[tgt](state_id, "B", cid)
                )
                for cid in challenge_ids
            }
            ids = [c for c in challenge_ids if c in templates and c in queries]
            if not ids:
                continue
            T = np.stack([templates[c] for c in ids], axis=0).astype(np.float64)
            Q = np.stack([queries[c] for c in ids], axis=0).astype(np.float64)
            s = row_zero_mean_ncc(T, Q)
            scores_s_d.extend(float(x) for x in s)
            for i, cid in enumerate(ids):
                rows.append(
                    {
                        "source_device": src,
                        "target_device": tgt,
                        "state_id": state_id,
                        "challenge_id": cid,
                        "score_type": SCORE_DEVICE_MISMATCH,
                        "S_D": float(s[i]),
                        "positive_class": SCORE_GENUINE,
                    }
                )
            if g is not None and len(g):
                rg = robust_gap_q05_minus_q95(g, s)
                ae = auc_eer_genuine_vs_negative(g, s)
            else:
                rg, ae = float("nan"), {"auc": float("nan"), "eer": float("nan")}
            # per pair summary accumulated below

    sd_arr = np.asarray(scores_s_d, dtype=np.float64)
    # Aggregate genuine across devices for RG_D
    g_all = np.concatenate([np.asarray(genuine_by_device[d], dtype=np.float64) for d in devices if d in genuine_by_device]) if genuine_by_device else np.array([])
    summary = {
        "state_id": state_id,
        "n_S_D": int(sd_arr.size),
        "median_S_D": float(np.median(sd_arr)) if sd_arr.size else float("nan"),
        "q95_S_D": float(np.percentile(sd_arr, 95)) if sd_arr.size else float("nan"),
        "rg_device": robust_gap_q05_minus_q95(g_all, sd_arr) if g_all.size and sd_arr.size else float("nan"),
        "device_mismatch_status": "AVAILABLE_MULTI_DEVICE",
    }
    if g_all.size and sd_arr.size:
        ae = auc_eer_genuine_vs_negative(g_all, sd_arr)
        summary["auc_device"] = ae["auc"]
        summary["eer_device"] = ae["eer"]
    else:
        summary["auc_device"] = float("nan")
        summary["eer_device"] = float("nan")
    return {"summary": summary, "score_rows": rows, "S_D": sd_arr}
