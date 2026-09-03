"""Multi-device Tracks C–D wrapping ml_attack clone runners (read-only)."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from experiment4_security.identity_credential import cd_checkpoint as ckpt
from experiment4_security.identity_credential.green_result_adapter import build_device_commons
from experiment4_security.ml_attack.config import ModelsConfig
from experiment4_security.ml_attack.track_c_clone import run_track_c
from experiment4_security.ml_attack.track_d_clone_transfer import run_track_d

logger = logging.getLogger(__name__)


def _drop_heavy_track_c_models(tc_meta: dict[str, Any]) -> dict[str, Any]:
    """Keep genuine scores / flags; drop in-RAM predictors (reload per-state for Track D)."""
    states = list(tc_meta.get("models_by_state", {}).keys())
    return {
        "models_by_state": {s: {"predictors": None, "lazy_checkpoint": True} for s in states},
        "genuine_by_state": tc_meta.get("genuine_by_state", {}),
        "elapsed_s": tc_meta.get("elapsed_s"),
        "note": tc_meta.get("note"),
        "resumed_from_checkpoint": bool(tc_meta.get("resumed_from_checkpoint")),
        "lazy_predictors": True,
    }


def run_green_tracks_cd_multidevice(
    devices: list[str],
    states: list[str],
    challenge_ids: list[str],
    challenge_features: dict[str, np.ndarray],
    *,
    detail_lookups: dict[str, Callable[[str, str, str], np.ndarray]],
    genuine_by_device_state: dict[str, dict[str, np.ndarray]],
    models_cfg: ModelsConfig | None = None,
    pca_dimension: int = 64,
    n_parallel_targets: int = 6,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    models_cfg = models_cfg or ModelsConfig()
    per_device = {}
    tc_frames = []
    td_frames = []
    tc_score_frames = []
    td_score_frames = []
    td_conclusions = []

    for device_id in devices:
        logger.info("[green C/D] device %s", device_id)

        if checkpoint_dir is not None and ckpt.is_device_done(checkpoint_dir, device_id):
            tc_sum, tc_scores, td_sum, td_scores, td_meta = ckpt.load_device_done(
                checkpoint_dir, device_id
            )
            tc_meta = {"resumed_from_checkpoint": True, "models_by_state": {}}
            for df in (tc_sum, tc_scores, td_sum, td_scores):
                if df is not None and not df.empty:
                    if "device_id" in df.columns:
                        df["device_id"] = device_id
                    else:
                        df.insert(0, "device_id", device_id)
            per_device[device_id] = {
                "tc_meta": tc_meta,
                "td_meta": td_meta,
                "dominance": pd.DataFrame(),
            }
            tc_frames.append(tc_sum)
            td_frames.append(td_sum)
            tc_score_frames.append(tc_scores)
            td_score_frames.append(td_scores)
            for c in td_meta.get("per_model_conclusions", []):
                td_conclusions.append({"device_id": device_id, **c})
            continue

        commons = build_device_commons(
            device_id, states, challenge_ids, detail_lookup=detail_lookups[device_id]
        )

        if checkpoint_dir is not None and ckpt.has_track_c(checkpoint_dir, device_id):
            logger.info("[green C/D] %s: resume Track C from checkpoint (skip training)", device_id)
            tc_sum, tc_scores, tc_meta = ckpt.load_track_c(checkpoint_dir, device_id)
            # Prefer live genuine scores from this formal resume over frozen checkpoint copy.
            if device_id in genuine_by_device_state:
                tc_meta["genuine_by_state"] = genuine_by_device_state[device_id]
            tc_dom = pd.DataFrame()
        else:
            tc_sum, tc_scores, tc_dom, tc_meta = run_track_c(
                states,
                challenge_ids,
                commons,
                challenge_features,
                detail_lookup=detail_lookups[device_id],
                models_cfg=models_cfg,
                pca_dimension=pca_dimension,
                genuine_by_state=genuine_by_device_state[device_id],
                device_id=device_id,
                checkpoint_dir=checkpoint_dir,
            )
            if checkpoint_dir is not None:
                ckpt.save_track_c(
                    checkpoint_dir,
                    device_id,
                    tc_sum=tc_sum,
                    tc_scores=tc_scores,
                    tc_meta=tc_meta,
                )
                # Spill predictors to disk and drop RAM before Track D.
                tc_meta = _drop_heavy_track_c_models(tc_meta)
                gc.collect()
                logger.info("[green C/D] %s: spilled Track C predictors to disk before Track D", device_id)

        td_sum, td_scores, td_meta = run_track_d(
            states,
            challenge_ids,
            commons,
            challenge_features,
            detail_lookup=detail_lookups[device_id],
            track_c_meta=tc_meta,
            track_c_summary=tc_sum,
            device_id=device_id,
            n_parallel_targets=n_parallel_targets,
            checkpoint_dir=checkpoint_dir,
        )
        if checkpoint_dir is not None:
            ckpt.save_device_done(
                checkpoint_dir,
                device_id,
                td_sum=td_sum,
                td_scores=td_scores,
                td_meta=td_meta,
            )

        for df in (tc_sum, tc_scores, td_sum, td_scores):
            if df is not None and not df.empty:
                if "device_id" in df.columns:
                    df["device_id"] = device_id
                else:
                    df.insert(0, "device_id", device_id)
        # Never retain predictors across devices (jetsam root cause).
        per_device[device_id] = {
            "tc_meta": _drop_heavy_track_c_models(tc_meta) if tc_meta else {"models_by_state": {}},
            "td_meta": td_meta,
            "dominance": tc_dom,
        }
        del tc_meta, commons
        gc.collect()
        tc_frames.append(tc_sum)
        td_frames.append(td_sum)
        tc_score_frames.append(tc_scores)
        td_score_frames.append(td_scores)
        for c in td_meta.get("per_model_conclusions", []):
            td_conclusions.append({"device_id": device_id, **c})

    return {
        "track_c_summary": pd.concat(tc_frames, ignore_index=True) if tc_frames else pd.DataFrame(),
        "track_d_summary": pd.concat(td_frames, ignore_index=True) if td_frames else pd.DataFrame(),
        "track_c_scores": pd.concat(tc_score_frames, ignore_index=True) if tc_score_frames else pd.DataFrame(),
        "track_d_scores": pd.concat(td_score_frames, ignore_index=True) if td_score_frames else pd.DataFrame(),
        "per_device": per_device,
        "track_d_conclusions": td_conclusions,
    }
