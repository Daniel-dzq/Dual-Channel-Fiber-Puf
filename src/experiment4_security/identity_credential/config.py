"""Configuration for identity–credential multi-device pipeline (M4 Pro tuned)."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from experiment4_security.identity_credential.schemas import (
    DEVELOPMENT_DEVICES,
    DEVICES,
    HELDOUT_DEVICES,
    STATES,
)


def _default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 2)


def new_run_id(suffix: str) -> str:
    """Local-timezone run id (Asia/Singapore by default). Prefer run_registry.new_run_id."""
    from experiment4_security.identity_credential.run_registry import new_run_id as _new

    return _new(suffix)


@dataclass
class PerformanceConfig:
    """Throughput knobs — do not change scientific formulas."""

    n_video_workers: int = field(default_factory=_default_workers)
    n_io_threads: int = 8
    track_b_parallel_targets: int = 8
    track_d_parallel_targets: int = 6
    green_device_parallel: int = 2
    preload_round_b: bool = True
    batch_size: int = 64
    blas_threads_hint: int = 8


@dataclass
class IdentityCredentialConfig:
    project_root: Path
    videos_root: Path
    output_root: Path
    devices: tuple[str, ...] = DEVICES
    states: tuple[str, ...] = STATES
    challenge_id_start: int = 1
    challenge_id_end: int = 128
    development_devices: tuple[str, ...] = DEVELOPMENT_DEVICES
    heldout_devices: tuple[str, ...] = HELDOUT_DEVICES
    lifecycle_src_guard: str = "src/experiment4_security/lifecycle"
    lifecycle_out_guard: str = "outputs/experiment4/lifecycle"
    f01_green_pilot_guard: str = "outputs/experiment4/security/runs/20260728_213919_f01_green_pilot"
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    make_figures: bool = False
    absolute_asr: None = None
    asr_status: str = "SCORE_SPACE_OR_PROTOCOL_MISMATCH"
    mode: str = "prepare_only"
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "IdentityCredentialConfig":
        path = Path(path)
        raw = yaml.safe_load(path.read_text()) or {}
        default_root = path.resolve().parent.parent
        pr = raw.get("project_root", None)
        if pr is None or str(pr).strip() in {".", ""}:
            project_root = default_root
        else:
            project_root = Path(pr)
            if not project_root.is_absolute():
                project_root = (default_root / project_root).resolve()
            else:
                project_root = project_root.resolve()
        videos_root = Path(raw.get("videos_root", "videos/identity_credential"))
        if not videos_root.is_absolute():
            videos_root = (project_root / videos_root).resolve()
        output_root = Path(raw.get("output_root", "outputs/experiment4/security"))
        if not output_root.is_absolute():
            output_root = (project_root / output_root).resolve()
        perf_raw = raw.get("performance", {}) or {}
        perf = PerformanceConfig(
            n_video_workers=int(perf_raw.get("n_video_workers", _default_workers())),
            n_io_threads=int(perf_raw.get("n_io_threads", 8)),
            track_b_parallel_targets=int(perf_raw.get("track_b_parallel_targets", 8)),
            track_d_parallel_targets=int(perf_raw.get("track_d_parallel_targets", 6)),
            green_device_parallel=int(perf_raw.get("green_device_parallel", 2)),
            preload_round_b=bool(perf_raw.get("preload_round_b", True)),
            batch_size=int(perf_raw.get("batch_size", 64)),
            blas_threads_hint=int(perf_raw.get("blas_threads_hint", 8)),
        )
        return cls(
            project_root=project_root,
            videos_root=videos_root,
            output_root=output_root,
            devices=tuple(raw.get("devices", list(DEVICES))),
            states=tuple(raw.get("states", list(STATES))),
            challenge_id_start=int(raw.get("challenge_id_start", 1)),
            challenge_id_end=int(raw.get("challenge_id_end", 128)),
            development_devices=tuple(
                raw.get("development_devices", list(DEVELOPMENT_DEVICES))
            ),
            heldout_devices=tuple(raw.get("heldout_devices", list(HELDOUT_DEVICES))),
            performance=perf,
            make_figures=bool(raw.get("make_figures", False)),
            mode=str(raw.get("mode", "prepare_only")),
            raw=raw,
        )

    def companion_path(self, key: str, default_relative: str) -> Path:
        """Resolve an auxiliary file (companion config, mask) relative to project_root."""
        value = Path(str(self.raw.get(key) or default_relative))
        return value if value.is_absolute() else (self.project_root / value).resolve()

    def runs_root(self) -> Path:
        return self.output_root / "runs"

    def run_dir(self, run_id: str) -> Path:
        return self.runs_root() / run_id

    def prepared_dir(self, run_id: str | None = None) -> Path:
        return self.run_dir(run_id or new_run_id("prepare"))

    def formal_dir(self, run_id: str | None = None) -> Path:
        return self.run_dir(run_id or new_run_id("formal"))

    def current_dir(self, run_id: str | None = None) -> Path:
        """Alias for formal_dir (timestamped formal run)."""
        return self.formal_dir(run_id)

    def to_resolved_dict(self) -> dict[str, Any]:
        return {
            "project_root": str(self.project_root),
            "videos_root": str(self.videos_root),
            "output_root": str(self.output_root),
            "runs_root": str(self.runs_root()),
            "devices": list(self.devices),
            "states": list(self.states),
            "challenge_id_start": self.challenge_id_start,
            "challenge_id_end": self.challenge_id_end,
            "development_devices": list(self.development_devices),
            "heldout_devices": list(self.heldout_devices),
            "performance": asdict(self.performance),
            "make_figures": self.make_figures,
            "absolute_asr": None,
            "asr_status": self.asr_status,
            "mode": self.mode,
            "protocol": "identity_credential_decoupling_v1",
            "guards": {
                "lifecycle_src": self.lifecycle_src_guard,
                "lifecycle_out": self.lifecycle_out_guard,
                "f01_green_pilot": self.f01_green_pilot_guard,
            },
        }
