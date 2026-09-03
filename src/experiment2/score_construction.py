"""NCC score construction for Experiment 2."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np

from experiment2.config import Experiment2Config
from experiment2.video_processing import ProcessedRecording
from puf_common.features import build_common_from_templates, mean_template, subtract_common, to_detail
from puf_common.masks import combine_pair_mask
from puf_common.ncc import zero_mean_ncc


@dataclass
class DetailCmFeatures:
    detail_blocks: list[np.ndarray]
    detail_representative: np.ndarray
    detail_cm_blocks: list[np.ndarray]
    detail_cm_representative: np.ndarray


def build_detail_cm_for_green_device(
    green_recordings: dict[str, ProcessedRecording],
    cfg: Experiment2Config,
) -> dict[str, DetailCmFeatures]:
    """Build detail_cm features for all green challenges on one device."""
    sigma = cfg.analysis.envelope_sigma_px
    eps = cfg.analysis.envelope_eps

    rep_details: dict[str, np.ndarray] = {}
    block_details: dict[str, list[np.ndarray]] = {}

    for cid, rec in green_recordings.items():
        detail_blocks = [
            to_detail(block, sigma=sigma, eps=eps) for block in rec.blocks
        ]
        block_details[cid] = detail_blocks
        rep_details[cid] = mean_template(detail_blocks)

    common = build_common_from_templates(list(rep_details.values()))

    out: dict[str, DetailCmFeatures] = {}
    for cid, rec in green_recordings.items():
        d_blocks = block_details[cid]
        d_rep = rep_details[cid]
        out[cid] = DetailCmFeatures(
            detail_blocks=d_blocks,
            detail_representative=d_rep,
            detail_cm_blocks=[subtract_common(b, common) for b in d_blocks],
            detail_cm_representative=subtract_common(d_rep, common),
        )
    return out


def build_detail_for_red_recording(
    recording: ProcessedRecording,
    cfg: Experiment2Config,
) -> DetailCmFeatures:
    sigma = cfg.analysis.envelope_sigma_px
    eps = cfg.analysis.envelope_eps
    detail_blocks = [
        to_detail(block, sigma=sigma, eps=eps) for block in recording.blocks
    ]
    d_rep = mean_template(detail_blocks)
    return DetailCmFeatures(
        detail_blocks=detail_blocks,
        detail_representative=d_rep,
        detail_cm_blocks=detail_blocks,
        detail_cm_representative=d_rep,
    )


def intra_block_pair_indices(num_blocks: int = 3) -> list[tuple[int, int]]:
    return [(i, j) for i in range(num_blocks) for j in range(i + 1, num_blocks)]


def compute_intra_scores(
    *,
    device_id: str,
    challenge_id: str,
    session_id: str,
    features: DetailCmFeatures,
    mask: np.ndarray,
    cfg: Experiment2Config,
    dark_hash: str,
    mask_id: str,
) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    pairs = intra_block_pair_indices(len(features.detail_cm_blocks))
    for bi, bj in pairs:
        ncc = zero_mean_ncc(
            features.detail_cm_blocks[bi],
            features.detail_cm_blocks[bj],
            mask=mask,
        )
        scores.append(
            _score_row(
                cfg=cfg,
                score_type="intra_repeatability",
                device_id_a=device_id,
                device_id_b=device_id,
                challenge_id_a=challenge_id,
                challenge_id_b=challenge_id,
                block_id_a=bi + 1,
                block_id_b=bj + 1,
                session_id_a=session_id,
                session_id_b=session_id,
                ncc=ncc,
                dark_hash=dark_hash,
                mask_id=mask_id,
            )
        )
    return scores


def compute_inter_challenge_scores(
    *,
    device_id: str,
    session_id: str,
    features_by_challenge: dict[str, DetailCmFeatures],
    challenge_ids: list[str],
    mask: np.ndarray,
    cfg: Experiment2Config,
    dark_hash: str,
    mask_id: str,
) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    for ci, cj in itertools.combinations(challenge_ids, 2):
        ncc = zero_mean_ncc(
            features_by_challenge[ci].detail_cm_representative,
            features_by_challenge[cj].detail_cm_representative,
            mask=mask,
        )
        scores.append(
            _score_row(
                cfg=cfg,
                score_type="inter_challenge",
                device_id_a=device_id,
                device_id_b=device_id,
                challenge_id_a=ci,
                challenge_id_b=cj,
                block_id_a=0,
                block_id_b=0,
                session_id_a=session_id,
                session_id_b=session_id,
                ncc=ncc,
                dark_hash=dark_hash,
                mask_id=mask_id,
            )
        )
    return scores


def compute_inter_device_scores(
    *,
    challenge_id: str,
    device_features: dict[str, DetailCmFeatures],
    device_masks: dict[str, np.ndarray],
    device_ids: list[str],
    cfg: Experiment2Config,
    dark_hash: str,
) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    for di, dj in itertools.combinations(device_ids, 2):
        pair_mask = combine_pair_mask(device_masks[di], device_masks[dj])
        mask_id = f"pair_{di}_{dj}"
        ncc = zero_mean_ncc(
            device_features[di].detail_cm_representative,
            device_features[dj].detail_cm_representative,
            mask=pair_mask,
        )
        scores.append(
            _score_row(
                cfg=cfg,
                score_type="inter_device",
                device_id_a=di,
                device_id_b=dj,
                challenge_id_a=challenge_id,
                challenge_id_b=challenge_id,
                block_id_a=0,
                block_id_b=0,
                session_id_a="",
                session_id_b="",
                ncc=ncc,
                dark_hash=dark_hash,
                mask_id=mask_id,
            )
        )
    return scores


def compute_short_term_ncc(
    features: DetailCmFeatures,
    mask: np.ndarray,
) -> list[float]:
    scores: list[float] = []
    for bi, bj in intra_block_pair_indices(len(features.detail_cm_blocks)):
        scores.append(
            zero_mean_ncc(
                features.detail_cm_blocks[bi],
                features.detail_cm_blocks[bj],
                mask=mask,
            )
        )
    return scores


def compute_red_drift_metrics(
    before: ProcessedRecording,
    after: ProcessedRecording,
    before_features: DetailCmFeatures,
    after_features: DetailCmFeatures,
    mask: np.ndarray,
) -> dict[str, float]:
    ncc = zero_mean_ncc(
        before_features.detail_representative,
        after_features.detail_representative,
        mask=mask,
    )
    sel = mask.astype(bool)
    mean_before = float(before.representative[sel].mean())
    mean_after = float(after.representative[sel].mean())
    intensity_change = mean_after - mean_before

    ys, xs = np.where(sel)
    if ys.size == 0:
        centroid_shift = float("nan")
    else:
        w_before = before.representative[sel]
        w_after = after.representative[sel]
        cy_b = float(np.average(ys, weights=w_before))
        cx_b = float(np.average(xs, weights=w_before))
        cy_a = float(np.average(ys, weights=w_after))
        cx_a = float(np.average(xs, weights=w_after))
        centroid_shift = float(np.hypot(cx_a - cx_b, cy_a - cy_b))

    return {
        "red_drift_ncc": ncc,
        "red_mean_intensity_change": intensity_change,
        "red_centroid_shift_px": centroid_shift,
    }


def _score_row(
    *,
    cfg: Experiment2Config,
    score_type: str,
    device_id_a: str,
    device_id_b: str,
    challenge_id_a: str,
    challenge_id_b: str,
    block_id_a: int,
    block_id_b: int,
    session_id_a: str,
    session_id_b: str,
    ncc: float,
    dark_hash: str,
    mask_id: str,
) -> dict[str, Any]:
    return {
        "experiment": cfg.experiment.name,
        "feature_type": cfg.analysis.feature_type,
        "score_type": score_type,
        "device_id_a": device_id_a,
        "device_id_b": device_id_b,
        "challenge_id_a": challenge_id_a,
        "challenge_id_b": challenge_id_b,
        "block_id_a": block_id_a,
        "block_id_b": block_id_b,
        "session_id_a": session_id_a,
        "session_id_b": session_id_b,
        "ncc": ncc,
        "mask_id": mask_id,
        "dark_artifact_hash": dark_hash,
        "preprocessing_version": cfg.analysis.preprocessing_version,
    }
