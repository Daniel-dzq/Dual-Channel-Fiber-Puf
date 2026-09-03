"""Multi-device Tracks A–B by wrapping ml_attack track runners (read-only)."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import numpy as np
import pandas as pd

from experiment4_security.identity_credential.green_multidevice_metrics import (
    evaluate_device_mismatch_for_state,
)
from experiment4_security.identity_credential.green_result_adapter import build_device_commons
from experiment4_security.ml_attack.enrollment import EnrollmentCommon
from experiment4_security.ml_attack.track_a_database_auth import run_track_a
from experiment4_security.ml_attack.track_b_template_revoke import run_track_b

logger = logging.getLogger(__name__)


def run_green_tracks_ab_multidevice(
    devices: list[str],
    states: list[str],
    challenge_ids: list[str],
    *,
    detail_lookups: dict[str, Callable[[str, str, str], np.ndarray]],
    n_parallel_targets: int = 8,
    green_device_parallel: int = 2,
    run_s_d: bool = True,
) -> dict[str, Any]:
    """Run Track A/B per device; optionally aggregate S_D across devices."""

    def _one_device(device_id: str) -> dict[str, Any]:
        logger.info("[green A/B] device %s", device_id)
        commons = build_device_commons(
            device_id, states, challenge_ids, detail_lookup=detail_lookups[device_id]
        )
        ta_sum, ta_scores, ta_meta = run_track_a(
            states,
            challenge_ids,
            commons,
            detail_lookup=detail_lookups[device_id],
            device_id=device_id,
        )
        tb_sum, tb_scores, tb_meta = run_track_b(
            states,
            challenge_ids,
            commons,
            detail_lookup=detail_lookups[device_id],
            genuine_by_state=ta_meta["genuine_by_state"],
            device_id=device_id,
            n_parallel_targets=n_parallel_targets,
        )
        for df in (ta_sum, ta_scores, tb_sum, tb_scores):
            if df is not None and not df.empty:
                if "device_id" in df.columns:
                    df["device_id"] = device_id
                else:
                    df.insert(0, "device_id", device_id)
        return {
            "device_id": device_id,
            "commons": commons,
            "ta_sum": ta_sum,
            "ta_scores": ta_scores,
            "ta_meta": ta_meta,
            "tb_sum": tb_sum,
            "tb_scores": tb_scores,
            "tb_meta": tb_meta,
        }

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, green_device_parallel)) as ex:
        futs = {ex.submit(_one_device, d): d for d in devices}
        for fut in as_completed(futs):
            results.append(fut.result())
    results.sort(key=lambda r: r["device_id"])

    ta_sum = pd.concat([r["ta_sum"] for r in results], ignore_index=True)
    tb_sum = pd.concat([r["tb_sum"] for r in results], ignore_index=True)
    ta_scores = pd.concat([r["ta_scores"] for r in results], ignore_index=True)
    tb_scores = pd.concat([r["tb_scores"] for r in results], ignore_index=True)

    out: dict[str, Any] = {
        "track_a_summary": ta_sum,
        "track_b_summary": tb_sum,
        "track_a_scores": ta_scores,
        "track_b_scores": tb_scores,
        "per_device": {r["device_id"]: r for r in results},
    }

    if run_s_d and len(devices) >= 2:
        logger.info(
            "[Track S_D] starting device-mismatch evaluation (%d devices × %d states × %d challenges)",
            len(devices),
            len(states),
            len(challenge_ids),
        )
        commons_keyed: dict[tuple[str, str], EnrollmentCommon] = {}
        genuine_by_device_state: dict[str, dict[str, np.ndarray]] = {}
        for r in results:
            d = r["device_id"]
            genuine_by_device_state[d] = r["ta_meta"]["genuine_by_state"]
            for s, c in r["commons"].items():
                commons_keyed[(d, s)] = c
        sd_summaries = []
        sd_scores = []
        for i, state in enumerate(states, start=1):
            logger.info("[Track S_D] state %s %d/%d", state, i, len(states))
            genuine_by_device = {d: genuine_by_device_state[d][state] for d in devices}
            ev = evaluate_device_mismatch_for_state(
                state_id=state,
                challenge_ids=challenge_ids,
                devices=devices,
                commons=commons_keyed,
                detail_lookups=detail_lookups,
                genuine_by_device=genuine_by_device,
            )
            sd_summaries.append(ev["summary"])
            sd_scores.extend(ev["score_rows"])
        logger.info("[Track S_D] done n_score_rows=%d", len(sd_scores))
        out["track_s_d_summary"] = pd.DataFrame(sd_summaries)
        out["track_s_d_scores"] = pd.DataFrame(sd_scores)
    else:
        out["track_s_d_summary"] = pd.DataFrame(
            [{"device_mismatch_status": "NOT_AVAILABLE_SINGLE_DEVICE_OR_SKIPPED"}]
        )
        out["track_s_d_scores"] = pd.DataFrame()

    return out
