"""Vector / feature cache paths for identity_credential (separate from ml_attack pilot)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from experiment4_security.ml_attack import video_preprocessing as vp


def cache_root(run_dir: Path) -> Path:
    return Path(run_dir) / "_cache"


def vector_dir(run_dir: Path) -> Path:
    d = cache_root(run_dir) / "vectors"
    d.mkdir(parents=True, exist_ok=True)
    return d


def red_feature_dir(run_dir: Path) -> Path:
    d = cache_root(run_dir) / "red_features"
    d.mkdir(parents=True, exist_ok=True)
    return d


def green_sample_id(device_id: str, state_id: str, round_id: str, challenge_id: str) -> str:
    return f"{device_id}_{state_id}_{round_id}_{challenge_id}"


def red_sample_id(device_id: str, state_id: str, phase: str) -> str:
    return f"{device_id}_{state_id}_R_{phase}"


def make_detail_lookup(
    vector_cache_dir: Path,
    device_id: str,
) -> Callable[[str, str, str], np.ndarray]:
    """Closure matching ml_attack track signatures: (state, round, cid) -> detail."""

    def detail_lookup(state: str, round_id: str, challenge_id: str) -> np.ndarray:
        sid = green_sample_id(device_id, state, round_id, challenge_id)
        return vp.load_vector_cache(vector_cache_dir, sid)

    return detail_lookup


def save_red_vector(run_dir: Path, sample_id: str, vec: np.ndarray) -> Path:
    path = red_feature_dir(run_dir) / f"{sample_id}.npy"
    np.save(path, np.asarray(vec, dtype=np.float64))
    return path


def load_red_vector(run_dir: Path, sample_id: str) -> np.ndarray:
    path = red_feature_dir(run_dir) / f"{sample_id}.npy"
    return np.load(path)
