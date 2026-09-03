"""Held-out closed-set green challenge retrieval for the fixed-state experiment (Fig. 6f).

Protocol (manuscript Fig. 6f): temporal windows W1 and W2 form the enrollment
templates, W3 is the held-out query. The green common component is estimated
from the enrollment representatives only and applied unchanged to the query.
Scores are masked zero-mean NCC values; the query is assigned to the template
with the highest score (Top-1).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from puf_common.features import build_common_from_templates, mean_template, subtract_common
from puf_common.ncc import zero_mean_ncc


def build_enrollment_templates(
    detail_blocks: dict[str, list[np.ndarray]],
    enroll_windows: list[int],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Enrollment-only common component and common-removed templates.

    ``detail_blocks[cid]`` holds the per-window detail images (pre-common, index 0 = W1).
    ``enroll_windows`` are 1-indexed window numbers used for enrollment.
    """
    reps = {cid: mean_template([blocks[w - 1] for w in enroll_windows]) for cid, blocks in detail_blocks.items()}
    common = build_common_from_templates(list(reps.values()))
    return {cid: subtract_common(rep, common) for cid, rep in reps.items()}, common


def query_template(detail_blocks: dict[str, list[np.ndarray]], cid: str, query_window: int, common: np.ndarray) -> np.ndarray:
    return subtract_common(detail_blocks[cid][query_window - 1], common)


def retrieve_one(
    query: np.ndarray,
    templates: dict[str, np.ndarray],
    true_cid: str,
    mask: np.ndarray,
) -> dict[str, Any]:
    scores = {cid: float(zero_mean_ncc(query, tpl, mask=mask)) for cid, tpl in templates.items()}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    predicted = ranked[0][0]
    wrong = [s for cid, s in scores.items() if cid != true_cid]
    best_wrong = max(wrong) if wrong else float("nan")
    rank = next(i for i, (cid, _) in enumerate(ranked, start=1) if cid == true_cid)
    return {
        "correct_template_score": scores[true_cid],
        "best_wrong_template_score": best_wrong,
        "retrieval_margin": scores[true_cid] - best_wrong,
        "predicted_challenge": predicted,
        "rank": rank,
        "correct": int(predicted == true_cid),
        "scores": scores,
    }


def heldout_retrieval(
    detail_blocks: dict[str, list[np.ndarray]],
    mask: np.ndarray,
    *,
    enroll_windows: list[int] = (1, 2),
    query_window: int = 3,
) -> list[dict[str, Any]]:
    """Run the W1+W2 enrollment / W3 query protocol for one device."""
    templates, common = build_enrollment_templates(detail_blocks, list(enroll_windows))
    rows = []
    for cid in detail_blocks:
        result = retrieve_one(query_template(detail_blocks, cid, query_window, common), templates, cid, mask)
        rows.append({"query_challenge": cid, **result})
    return rows
