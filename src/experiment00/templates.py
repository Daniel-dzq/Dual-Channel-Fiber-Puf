"""Detail transform and challenge common-mode removal (reuse puf_common)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from experiment00.acquisition import EXPECTED_CHALLENGES, ORDER_SOURCE
from experiment00.config import Experiment00Config
from experiment00.video_processing import ProcessedVideo
from puf_common.features import (
    build_common_from_templates,
    subtract_common,
    to_detail,
)


@dataclass
class GreenTemplates:
    length_cm: int
    fiber_id: str
    round: str
    challenge: str
    challenge_id: str
    sequence_id: str | None
    acquisition_position: int | None
    order_source: str | None
    blocks_raw: list[np.ndarray]
    blocks_detail: list[np.ndarray]
    blocks_detail_cm: list[np.ndarray]
    video_raw: np.ndarray
    video_detail: np.ndarray
    video_detail_cm: np.ndarray
    common_component: np.ndarray | None = None


def _challenge_id_of(rec: ProcessedVideo) -> str:
    ch = getattr(rec, "challenge_id", None) or rec.challenge
    if not ch:
        raise ValueError(f"Missing challenge_id for {rec.path}")
    return str(ch)


def build_green_detail_cm_group(
    recordings: list[ProcessedVideo],
    cfg: Experiment00Config,
    *,
    require_full_challenge_set: bool = True,
) -> dict[str, GreenTemplates]:
    """Build detail_cm for one (length, fiber, round) group across challenges.

    Common-mode is estimated ONLY within this group. Round A and Round B never share
    a common component. Different fibers/lengths never share.

    Templates are keyed by challenge_id. Input list order must not affect results
    (common-mode is order-invariant across the challenge set).
    """
    if not recordings:
        return {}
    sigma = cfg.preprocessing.detail_sigma_px
    eps = cfg.preprocessing.detail_epsilon
    expected = list(cfg.dataset.expected_challenges or EXPECTED_CHALLENGES)

    detail_reps: dict[str, np.ndarray] = {}
    block_details: dict[str, list[np.ndarray]] = {}
    raw_reps: dict[str, np.ndarray] = {}
    raw_blocks: dict[str, list[np.ndarray]] = {}
    meta: dict[str, ProcessedVideo] = {}
    for rec in recordings:
        assert rec.illumination == "green" and rec.round
        ch = _challenge_id_of(rec)
        if ch in detail_reps:
            raise ValueError(
                f"Duplicate challenge_id {ch} in common-mode group "
                f"({rec.length_cm}cm {rec.fiber_id} {rec.round})"
            )
        d_blocks = [to_detail(b, sigma, eps) for b in rec.blocks_raw]
        v_detail = to_detail(rec.representative_raw, sigma, eps)
        detail_reps[ch] = v_detail
        block_details[ch] = d_blocks
        raw_reps[ch] = rec.representative_raw
        raw_blocks[ch] = rec.blocks_raw
        meta[ch] = rec

    present = set(detail_reps)
    if require_full_challenge_set:
        missing = sorted(set(expected) - present)
        extra = sorted(present - set(expected))
        if missing or extra:
            raise ValueError(
                f"Common-mode group must contain exactly {expected}; "
                f"missing={missing} extra={extra}"
            )
    # Deterministic challenge-id order for reproducible (order-invariant) common
    ch_order = [c for c in expected if c in detail_reps] + sorted(
        present - set(expected)
    )

    if cfg.preprocessing.green_common_mode:
        common = build_common_from_templates([detail_reps[c] for c in ch_order])
    else:
        common = np.zeros_like(next(iter(detail_reps.values())))

    n_blocks = len(next(iter(block_details.values())))
    block_commons: list[np.ndarray] = []
    for bi in range(n_blocks):
        temps = [block_details[c][bi] for c in ch_order]
        if cfg.preprocessing.green_common_mode:
            block_commons.append(build_common_from_templates(temps))
        else:
            block_commons.append(np.zeros_like(temps[0]))

    out: dict[str, GreenTemplates] = {}
    for ch, rec in meta.items():
        v_cm = subtract_common(detail_reps[ch], common)
        b_cm = [
            subtract_common(block_details[ch][bi], block_commons[bi]) for bi in range(n_blocks)
        ]
        out[ch] = GreenTemplates(
            length_cm=rec.length_cm,
            fiber_id=rec.fiber_id,
            round=rec.round or "",
            challenge=ch,
            challenge_id=ch,
            sequence_id=getattr(rec, "sequence_id", None),
            acquisition_position=getattr(rec, "acquisition_position", None),
            order_source=getattr(rec, "order_source", None) or ORDER_SOURCE,
            blocks_raw=raw_blocks[ch],
            blocks_detail=block_details[ch],
            blocks_detail_cm=b_cm,
            video_raw=raw_reps[ch],
            video_detail=detail_reps[ch],
            video_detail_cm=v_cm,
            common_component=common,
        )
    return out


def group_green_recordings(
    green: list[ProcessedVideo],
) -> dict[tuple[int, str, str], list[ProcessedVideo]]:
    groups: dict[tuple[int, str, str], list[ProcessedVideo]] = defaultdict(list)
    for rec in green:
        if rec.round is None:
            continue
        groups[(rec.length_cm, rec.fiber_id, rec.round)].append(rec)
    return groups

