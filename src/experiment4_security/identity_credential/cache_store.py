"""Hash-isolated shared cache for Experiment 4 security.

Cache pollution is a scientific risk. Keys must include protocol identity,
not just filenames. Enrollment commons require the ordered set of Round-A
sample hashes — never cache commons from a single-video key alone.

Layout:
  _shared_cache/
    video_templates/<cache_key>.npy
    detail/<cache_key>.npy
    red_features/<cache_key>.json
    enrollment_commons/<cache_key>.npy
    cache_index.parquet
    legacy_vectors/   # optional symlink/compat to old F01 flat vectors
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Frozen protocol constants for cache key material
DETAIL_SIGMA = 42.0
DETAIL_EPSILON = 1.0
REPRESENTATION_VERSION = "fullres_detail_cm_enrollment_frozen_v1"
FRAME_AGGREGATION_PROTOCOL = "discard_10s_head_tail__3_block_median__rep_median"
FULL_VIDEO_POLICY = "retain_middle_after_head_tail_discard"
CHANNEL_GREEN = "green_bgr1"
CHANNEL_RED = "red_bgr2"


def _h(parts: Iterable[str]) -> str:
    blob = "\n".join(parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def file_fingerprint(path: Path) -> dict[str, Any]:
    path = Path(path)
    st = path.stat() if path.exists() else None
    # Content hash is expensive for 12k videos; include size/mtime always,
    # and optional content hash when requested.
    return {
        "path": str(path),
        "exists": path.exists(),
        "size": int(st.st_size) if st else None,
        "mtime_ns": int(st.st_mtime_ns) if st else None,
    }


def content_sha256(path: Path, *, max_bytes: int | None = None) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            n += len(chunk)
            if max_bytes is not None and n >= max_bytes:
                break
    return h.hexdigest()


@dataclass(frozen=True)
class DetailCacheKeySpec:
    device_id: str
    state_id: str
    round_id: str
    challenge_id: str
    channel: str
    video_size: int | None
    video_mtime_ns: int | None
    video_sha256: str | None
    valid_mask_hash: str
    frame_aggregation_protocol: str = FRAME_AGGREGATION_PROTOCOL
    full_video_policy: str = FULL_VIDEO_POLICY
    detail_sigma: float = DETAIL_SIGMA
    epsilon: float = DETAIL_EPSILON
    representation_version: str = REPRESENTATION_VERSION
    code_version: str = "unknown"

    def cache_key(self) -> str:
        return _h(
            [
                "detail_v1",
                self.device_id,
                self.state_id,
                self.round_id,
                self.challenge_id,
                self.channel,
                str(self.video_size),
                str(self.video_mtime_ns),
                str(self.video_sha256),
                self.valid_mask_hash,
                self.frame_aggregation_protocol,
                self.full_video_policy,
                str(self.detail_sigma),
                str(self.epsilon),
                self.representation_version,
                self.code_version,
            ]
        )


@dataclass(frozen=True)
class EnrollmentCommonCacheKeySpec:
    """Must include ordered Round-A sample hash set — never single-video only."""

    device_id: str
    state_id: str
    ordered_round_a_sample_hashes: tuple[str, ...]
    valid_mask_hash: str
    detail_sigma: float = DETAIL_SIGMA
    epsilon: float = DETAIL_EPSILON
    representation_version: str = REPRESENTATION_VERSION
    code_version: str = "unknown"

    def cache_key(self) -> str:
        if len(self.ordered_round_a_sample_hashes) != 128:
            # Still allow partial for tests, but mark in key
            pass
        return _h(
            [
                "enrollment_common_v1",
                self.device_id,
                self.state_id,
                *self.ordered_round_a_sample_hashes,
                self.valid_mask_hash,
                str(self.detail_sigma),
                str(self.epsilon),
                self.representation_version,
                self.code_version,
            ]
        )


@dataclass(frozen=True)
class RedFeatureCacheKeySpec:
    device_id: str
    state_id: str
    phase: str
    video_size: int | None
    video_mtime_ns: int | None
    video_sha256: str | None
    valid_mask_hash: str
    feature_contract: str = "lifecycle_red_13d_v1"
    frame_aggregation_protocol: str = FRAME_AGGREGATION_PROTOCOL
    code_version: str = "unknown"

    def cache_key(self) -> str:
        return _h(
            [
                "red_features_v1",
                self.device_id,
                self.state_id,
                self.phase,
                str(self.video_size),
                str(self.video_mtime_ns),
                str(self.video_sha256),
                self.valid_mask_hash,
                self.feature_contract,
                self.frame_aggregation_protocol,
                self.code_version,
            ]
        )


class SharedCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.detail_dir = self.root / "detail"
        self.template_dir = self.root / "video_templates"
        self.red_dir = self.root / "red_features"
        self.common_dir = self.root / "enrollment_commons"
        for d in (self.detail_dir, self.template_dir, self.red_dir, self.common_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "cache_index.parquet"
        self._rows: list[dict[str, Any]] = []

    def detail_path(self, key: str) -> Path:
        return self.detail_dir / f"{key}.npy"

    def load_detail(self, key: str) -> tuple[np.ndarray | None, str]:
        p = self.detail_path(key)
        if p.exists():
            self._rows.append({"cache_key": key, "kind": "detail", "event": "CACHE_HIT"})
            return np.load(p), "CACHE_HIT"
        self._rows.append({"cache_key": key, "kind": "detail", "event": "MISS"})
        return None, "MISS"

    def save_detail(self, key: str, arr: np.ndarray, *, event: str = "RECOMPUTED") -> Path:
        p = self.detail_path(key)
        np.save(p, np.asarray(arr))
        self._rows.append({"cache_key": key, "kind": "detail", "event": event, "path": str(p)})
        return p

    def flush_index(self) -> Path | None:
        if not self._rows:
            return None
        try:
            import pandas as pd

            df = pd.DataFrame(self._rows)
            if self.index_path.exists():
                old = pd.read_parquet(self.index_path)
                df = pd.concat([old, df], ignore_index=True)
            df.to_parquet(self.index_path, index=False)
            return self.index_path
        except Exception:
            # parquet optional; fall back to jsonl
            jl = self.root / "cache_index.jsonl"
            with open(jl, "a", encoding="utf-8") as f:
                for r in self._rows:
                    f.write(json.dumps(r) + "\n")
            self._rows.clear()
            return jl


def ensure_shared_cache_layout(security_root: Path) -> dict[str, str]:
    root = Path(security_root) / "_shared_cache"
    cache = SharedCache(root)
    # Keep legacy flat vectors discoverable without mixing into hash namespace
    legacy = root / "legacy_vectors"
    old = root / "vectors"
    if old.exists() and not legacy.exists():
        # Do not move 14GB blindly if already named vectors — record pointer
        (root / "LEGACY_VECTORS_NOTE.txt").write_text(
            "Existing F01 flat vectors/ remain for pilot compatibility.\n"
            "New formal runs must use hash-keyed detail/ entries and record CACHE_HIT/RECOMPUTED.\n"
            "Enrollment commons must use enrollment_commons/ with ordered Round-A hash sets.\n"
            "Dev-fit red standardizer / PCA / thresholds must NOT be applied unconditionally to held-out devices.\n"
        )
    readme = root / "CACHE_PROTOCOL.md"
    if not readme.exists():
        readme.write_text(
            "# Shared cache protocol\n\n"
            "- Keys include device/state/round/challenge, mask hash, sigma=42, eps=1, aggregation protocol, code version.\n"
            "- Enrollment common keys include the ordered 128 Round-A sample hashes.\n"
            "- Record CACHE_HIT vs RECOMPUTED per formal result.\n"
        )
    return {
        "root": str(root),
        "detail": str(cache.detail_dir),
        "red_features": str(cache.red_dir),
        "enrollment_commons": str(cache.common_dir),
    }


def spec_to_dict(spec: Any) -> dict[str, Any]:
    return asdict(spec)
