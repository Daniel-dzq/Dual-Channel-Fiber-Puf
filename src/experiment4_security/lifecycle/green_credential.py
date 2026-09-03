"""Green credential: detail_cm per device × state × round."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiment4_security.common.frozen_protocol import ENVELOPE_EPS, ENVELOPE_SIGMA_PX
from experiment4_security.common.video_io import ProcessedRecording
from puf_common.features import (
    build_common_from_templates,
    mean_template,
    subtract_common,
    to_detail,
)
from puf_common.ncc import zero_mean_ncc


@dataclass
class GreenFeat:
    device_id: str
    state_id: str
    round_id: str
    challenge_id: str
    detail_cm: np.ndarray


def build_detail_cm_group(
    recordings: dict[str, ProcessedRecording],
    *,
    device_id: str,
    state_id: str,
    round_id: str,
    sigma: float = ENVELOPE_SIGMA_PX,
    eps: float = ENVELOPE_EPS,
) -> dict[str, GreenFeat]:
    reps: dict[str, np.ndarray] = {}
    for cid, rec in recordings.items():
        details = [to_detail(b, sigma=sigma, eps=eps) for b in rec.blocks]
        reps[cid] = mean_template(details)
    common = build_common_from_templates(list(reps.values()))
    out: dict[str, GreenFeat] = {}
    for cid, d_rep in reps.items():
        out[cid] = GreenFeat(
            device_id=device_id,
            state_id=state_id,
            round_id=round_id,
            challenge_id=cid,
            detail_cm=np.asarray(subtract_common(d_rep, common), dtype=np.float32),
        )
    return out


def green_score(a: GreenFeat, b: GreenFeat, mask: np.ndarray) -> float:
    return float(zero_mean_ncc(a.detail_cm, b.detail_cm, mask=mask))
