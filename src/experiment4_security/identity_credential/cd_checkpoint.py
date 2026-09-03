"""Disk checkpoints for green Tracks C/D (per-device, crash-safe resume).

Science-preserving: only serializes already-computed Track C predictors / tables
and Track D partial results. Does not change fit or scoring logic.

Memory note: predictors are stored **per state** so Track D can load one source
at a time. A legacy monolithic ``track_c_predictors.joblib`` is migrated on first
access (split + delete) to avoid ~80GB jetsam kills when all 8 states are loaded.
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

TRACK_C_SUMMARY = "track_c_summary.csv"
TRACK_C_SCORES = "track_c_scores.csv"
TRACK_C_MODELS = "track_c_predictors.joblib"  # legacy monolithic
TRACK_C_MODELS_STATE = "track_c_predictors_{state}.joblib"
TRACK_C_META = "track_c_meta.json"
TRACK_C_GENUINE = "track_c_genuine.joblib"
TRACK_D_SUMMARY = "track_d_summary.csv"
TRACK_D_SCORES = "track_d_scores.csv"
TRACK_D_SOURCES_DONE = "track_d_sources_done.json"
DEVICE_DONE = "device_cd_done.json"


def device_dir(checkpoint_root: Path, device_id: str) -> Path:
    d = Path(checkpoint_root) / device_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _state_predictor_path(d: Path, state: str) -> Path:
    return d / TRACK_C_MODELS_STATE.format(state=state)


def _list_state_predictor_files(d: Path) -> list[Path]:
    return sorted(d.glob("track_c_predictors_*.joblib"))


def is_device_done(checkpoint_root: Path, device_id: str) -> bool:
    d = Path(checkpoint_root) / device_id
    return (d / DEVICE_DONE).is_file() and (d / TRACK_C_SUMMARY).is_file() and (d / TRACK_D_SUMMARY).is_file()


def has_track_c(checkpoint_root: Path, device_id: str) -> bool:
    d = Path(checkpoint_root) / device_id
    if not (d / TRACK_C_SUMMARY).is_file():
        return False
    if (d / TRACK_C_MODELS).is_file():
        return True
    return bool(_list_state_predictor_files(d))


def save_track_c_state_predictors(
    checkpoint_root: Path,
    device_id: str,
    state: str,
    predictors: dict[str, Any],
) -> Path:
    """Spill one state's predictors during Track C (keeps RAM flat across 8 states)."""
    d = device_dir(checkpoint_root, device_id)
    path = _state_predictor_path(d, state)
    joblib.dump(predictors, path, compress=3)
    logger.info("[cd checkpoint] spilled Track C predictors %s/%s → %s", device_id, state, path.name)
    return path


def save_track_c(
    checkpoint_root: Path,
    device_id: str,
    *,
    tc_sum: pd.DataFrame,
    tc_scores: pd.DataFrame,
    tc_meta: dict[str, Any],
) -> Path:
    d = device_dir(checkpoint_root, device_id)
    tc_sum.to_csv(d / TRACK_C_SUMMARY, index=False)
    tc_scores.to_csv(d / TRACK_C_SCORES, index=False)

    states = sorted(tc_meta.get("models_by_state", {}).keys())
    for state in states:
        entry = tc_meta["models_by_state"][state]
        predictors = entry.get("predictors")
        path = _state_predictor_path(d, state)
        if predictors is not None:
            joblib.dump(predictors, path, compress=3)
        elif not path.is_file():
            raise FileNotFoundError(
                f"Track C predictors missing for {device_id}/{state} (not in RAM and not on disk: {path})"
            )

    joblib.dump(tc_meta.get("genuine_by_state", {}), d / TRACK_C_GENUINE, compress=3)
    meta_light = {
        "elapsed_s": tc_meta.get("elapsed_s"),
        "note": tc_meta.get("note"),
        "states": states,
        "predictor_layout": "per_state_v1",
    }
    (d / TRACK_C_META).write_text(json.dumps(meta_light, indent=2, default=str) + "\n", encoding="utf-8")

    # Remove legacy monolith if present (avoid double storage / accidental full load).
    legacy = d / TRACK_C_MODELS
    if legacy.is_file():
        legacy.unlink()
        logger.info("[cd checkpoint] removed legacy monolithic predictors for %s", device_id)

    logger.info("[cd checkpoint] saved Track C for %s (%d states, per-state) → %s", device_id, len(states), d)
    return d


def _migrate_monolithic_predictors(d: Path) -> None:
    """Split legacy track_c_predictors.joblib into per-state files, then delete it."""
    legacy = d / TRACK_C_MODELS
    if not legacy.is_file():
        return
    if _list_state_predictor_files(d):
        # Already split; drop leftover monolith.
        legacy.unlink()
        logger.info("[cd checkpoint] dropped leftover monolithic predictors under %s", d)
        return

    logger.info("[cd checkpoint] migrating monolithic predictors → per-state under %s", d)
    payload = joblib.load(legacy)
    predictors_by_state = payload.get("predictors_by_state", {})
    states = sorted(predictors_by_state.keys())
    for state in states:
        joblib.dump(predictors_by_state[state], _state_predictor_path(d, state), compress=3)
        predictors_by_state[state] = None
        gc.collect()
    joblib.dump(payload.get("genuine_by_state", {}), d / TRACK_C_GENUINE, compress=3)
    meta_light = {
        "elapsed_s": payload.get("elapsed_s"),
        "note": payload.get("note", "migrated_from_monolithic"),
        "states": states,
        "predictor_layout": "per_state_v1",
    }
    (d / TRACK_C_META).write_text(json.dumps(meta_light, indent=2, default=str) + "\n", encoding="utf-8")
    del payload, predictors_by_state
    gc.collect()
    legacy.unlink()
    logger.info("[cd checkpoint] migration complete under %s", d)


def load_track_c_predictors_state(
    checkpoint_root: Path, device_id: str, state: str
) -> dict[str, Any]:
    """Load predictors for a single source state (Track D memory-safe path)."""
    d = Path(checkpoint_root) / device_id
    _migrate_monolithic_predictors(d)
    path = _state_predictor_path(d, state)
    if not path.is_file():
        raise FileNotFoundError(f"Missing per-state Track C predictors: {path}")
    predictors = joblib.load(path)
    logger.info("[cd checkpoint] loaded Track C predictors %s/%s", device_id, state)
    return predictors


def load_track_c(
    checkpoint_root: Path, device_id: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load Track C tables + light meta. Predictors stay on disk (lazy per state)."""
    d = Path(checkpoint_root) / device_id
    _migrate_monolithic_predictors(d)

    tc_sum = pd.read_csv(d / TRACK_C_SUMMARY)
    tc_scores = pd.read_csv(d / TRACK_C_SCORES) if (d / TRACK_C_SCORES).is_file() else pd.DataFrame()

    genuine_by_state: dict[str, Any] = {}
    elapsed_s = None
    note = "loaded_from_checkpoint"
    states: list[str] = []
    if (d / TRACK_C_META).is_file():
        meta_light = json.loads((d / TRACK_C_META).read_text(encoding="utf-8"))
        elapsed_s = meta_light.get("elapsed_s")
        note = meta_light.get("note", note)
        states = list(meta_light.get("states") or [])
    if (d / TRACK_C_GENUINE).is_file():
        genuine_by_state = joblib.load(d / TRACK_C_GENUINE) or {}
    if not states:
        states = sorted(
            {
                p.name[len("track_c_predictors_") : -len(".joblib")]
                for p in _list_state_predictor_files(d)
            }
        )

    # Empty predictors placeholders — Track D loads one state at a time.
    models_by_state = {state: {"predictors": None, "lazy_checkpoint": True} for state in states}
    tc_meta = {
        "models_by_state": models_by_state,
        "genuine_by_state": genuine_by_state,
        "elapsed_s": elapsed_s,
        "note": note,
        "resumed_from_checkpoint": True,
        "lazy_predictors": True,
    }
    logger.info(
        "[cd checkpoint] loaded Track C tables for %s (lazy predictors, %d states)",
        device_id,
        len(states),
    )
    return tc_sum, tc_scores, tc_meta


def save_track_d_partial(
    checkpoint_root: Path,
    device_id: str,
    *,
    summary_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    sources_done: list[str],
) -> None:
    d = device_dir(checkpoint_root, device_id)
    pd.DataFrame(summary_rows).to_csv(d / TRACK_D_SUMMARY, index=False)
    pd.DataFrame(score_rows).to_csv(d / TRACK_D_SCORES, index=False)
    (d / TRACK_D_SOURCES_DONE).write_text(
        json.dumps({"sources_done": list(sources_done)}, indent=2) + "\n", encoding="utf-8"
    )


def load_track_d_partial(
    checkpoint_root: Path, device_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    d = Path(checkpoint_root) / device_id
    sources_done: list[str] = []
    if (d / TRACK_D_SOURCES_DONE).is_file():
        sources_done = list(json.loads((d / TRACK_D_SOURCES_DONE).read_text(encoding="utf-8")).get("sources_done", []))
    summary_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    if (d / TRACK_D_SUMMARY).is_file():
        df = pd.read_csv(d / TRACK_D_SUMMARY)
        if not df.empty:
            summary_rows = df.to_dict(orient="records")
    if (d / TRACK_D_SCORES).is_file():
        df = pd.read_csv(d / TRACK_D_SCORES)
        if not df.empty:
            score_rows = df.to_dict(orient="records")
    if sources_done:
        logger.info(
            "[cd checkpoint] Track D partial for %s: %d sources done, %d summary rows",
            device_id,
            len(sources_done),
            len(summary_rows),
        )
    return summary_rows, score_rows, sources_done


def save_device_done(
    checkpoint_root: Path,
    device_id: str,
    *,
    td_sum: pd.DataFrame,
    td_scores: pd.DataFrame,
    td_meta: dict[str, Any],
) -> None:
    d = device_dir(checkpoint_root, device_id)
    td_sum.to_csv(d / TRACK_D_SUMMARY, index=False)
    td_scores.to_csv(d / TRACK_D_SCORES, index=False)
    (d / DEVICE_DONE).write_text(
        json.dumps(
            {
                "device_id": device_id,
                "track_d_conclusions": td_meta.get("per_model_conclusions", []),
                "elapsed_s": td_meta.get("elapsed_s"),
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("[cd checkpoint] device %s C/D complete", device_id)


def load_device_done(
    checkpoint_root: Path, device_id: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    d = Path(checkpoint_root) / device_id
    tc_sum = pd.read_csv(d / TRACK_C_SUMMARY)
    tc_scores = pd.read_csv(d / TRACK_C_SCORES) if (d / TRACK_C_SCORES).is_file() else pd.DataFrame()
    td_sum = pd.read_csv(d / TRACK_D_SUMMARY)
    td_scores = pd.read_csv(d / TRACK_D_SCORES) if (d / TRACK_D_SCORES).is_file() else pd.DataFrame()
    meta = json.loads((d / DEVICE_DONE).read_text(encoding="utf-8"))
    td_meta = {
        "per_model_conclusions": meta.get("track_d_conclusions", []),
        "elapsed_s": meta.get("elapsed_s"),
        "resumed_from_checkpoint": True,
    }
    logger.info("[cd checkpoint] skipping completed device %s", device_id)
    return tc_sum, tc_scores, td_sum, td_scores, td_meta
