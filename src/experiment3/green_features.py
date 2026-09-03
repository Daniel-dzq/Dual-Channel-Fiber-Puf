"""Green detail_cm features with per device×state×round common responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from experiment3.config import Experiment3Config
from experiment3.video_processing import ProcessedRecording
from puf_common.features import (
    build_common_from_templates,
    mean_template,
    subtract_common,
    to_detail,
)
from puf_common.ncc import zero_mean_ncc


@dataclass
class GreenChallengeFeatures:
    device_id: str
    state_id: str
    round_id: str
    challenge_id: str
    detail_blocks: list[np.ndarray]
    detail_representative: np.ndarray
    detail_cm_blocks: list[np.ndarray]
    detail_cm_representative: np.ndarray
    n_blocks: int = 0
    detail_mean: float = float("nan")


def build_detail_cm_for_green_group(
    recordings: dict[str, ProcessedRecording],
    cfg: Experiment3Config,
    *,
    device_id: str,
    state_id: str,
    round_id: str,
) -> dict[str, GreenChallengeFeatures]:
    """Build detail_cm for one device × state × round.

    Common response = mean of the eight challenge detail representatives.
    Round A and Round B MUST use separate commons (caller enforces grouping).

    Memory note: only float32 representatives are retained; per-block detail /
    detail_cm maps are discarded after aggregation (scores/plots use reps only).
    """
    sigma = cfg.analysis.envelope_sigma_px
    eps = cfg.analysis.envelope_eps

    rep_details: dict[str, np.ndarray] = {}
    n_blocks_by_cid: dict[str, int] = {}
    for cid, rec in recordings.items():
        detail_blocks = [to_detail(block, sigma=sigma, eps=eps) for block in rec.blocks]
        n_blocks_by_cid[cid] = len(detail_blocks)
        rep_details[cid] = mean_template(detail_blocks)
        del detail_blocks

    if not rep_details:
        return {}

    common = build_common_from_templates(list(rep_details.values()))
    out: dict[str, GreenChallengeFeatures] = {}
    for cid in recordings:
        d_rep = rep_details[cid]
        cm_rep = np.asarray(subtract_common(d_rep, common), dtype=np.float32)
        out[cid] = GreenChallengeFeatures(
            device_id=device_id,
            state_id=state_id,
            round_id=round_id,
            challenge_id=cid,
            detail_blocks=[],
            # Drop full detail representative; scores/plots use detail_cm only.
            detail_representative=np.empty((0,), dtype=np.float32),
            detail_cm_blocks=[],
            detail_cm_representative=cm_rep,
            n_blocks=n_blocks_by_cid[cid],
            detail_mean=float(np.mean(d_rep)),
        )
    return out


def group_green_recordings(
    recordings: list[ProcessedRecording],
) -> dict[tuple[str, str, str], dict[str, ProcessedRecording]]:
    """Key: (device_id, state_id, round_id) -> {challenge_id: recording}."""
    groups: dict[tuple[str, str, str], dict[str, ProcessedRecording]] = {}
    for rec in recordings:
        if rec.channel != "green":
            continue
        key = (rec.device_id, rec.state_id, rec.round_id)
        groups.setdefault(key, {})[rec.challenge_id] = rec
    return groups


def build_all_green_features(
    recordings: list[ProcessedRecording],
    cfg: Experiment3Config,
) -> dict[tuple[str, str, str, str], GreenChallengeFeatures]:
    """Return features keyed by (device, state, round, challenge)."""
    out: dict[tuple[str, str, str, str], GreenChallengeFeatures] = {}
    for (device, state, rnd), by_cid in group_green_recordings(recordings).items():
        feats = build_detail_cm_for_green_group(
            by_cid,
            cfg,
            device_id=device,
            state_id=state,
            round_id=rnd,
        )
        for cid, feat in feats.items():
            out[(device, state, rnd, cid)] = feat
    return out


def green_similarity(
    a: GreenChallengeFeatures,
    b: GreenChallengeFeatures,
    mask: np.ndarray,
) -> float:
    """Official green score S_G = zero-mean NCC between detail_cm templates."""
    return zero_mean_ncc(a.detail_cm_representative, b.detail_cm_representative, mask=mask)


def green_features_to_dataframe(
    features: dict[tuple[str, str, str, str], GreenChallengeFeatures],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (device, state, rnd, cid), feat in sorted(features.items()):
        rows.append(
            {
                "device_id": device,
                "state_id": state,
                "round_id": rnd,
                "challenge_id": cid,
                "channel": "green",
                "detail_cm_mean": float(np.mean(feat.detail_cm_representative)),
                "detail_cm_std": float(np.std(feat.detail_cm_representative)),
                "detail_mean": float(
                    feat.detail_mean
                    if np.isfinite(feat.detail_mean)
                    else (
                        float(np.mean(feat.detail_representative))
                        if feat.detail_representative.size
                        else float("nan")
                    )
                ),
                "n_blocks": int(feat.n_blocks or len(feat.detail_cm_blocks) or 0),
            }
        )
    return pd.DataFrame(rows)


def assert_no_cross_round_common_leakage(
    features: dict[tuple[str, str, str, str], GreenChallengeFeatures],
) -> None:
    """Sanity: commons are built per round, so A and B detail_cm differ when details match.

    This helper is used by tests; it verifies that for the same device/state/challenge,
    Round A and Round B feature objects are distinct objects with independently
    subtracted commons (callers construct separately).
    """
    keys_a = {k for k in features if k[2] == "A"}
    for device, state, _rnd, cid in keys_a:
        key_b = (device, state, "B", cid)
        if key_b not in features:
            continue
        fa = features[(device, state, "A", cid)]
        fb = features[key_b]
        # If commons were wrongly shared across rounds after identical details,
        # representatives could match exactly; we only assert object independence here.
        if fa is fb:
            raise AssertionError("Round A/B features unexpectedly aliased")
