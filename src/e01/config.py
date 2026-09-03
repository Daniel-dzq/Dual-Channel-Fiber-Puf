"""Configuration loading and derived SLM geometry."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SlmConfig:
    canvas_width: int = 1024
    canvas_height: int = 768
    center_x: int = 512
    center_y: int = 384
    active_width: int = 512
    active_height: int = 512
    gray_off: int = 0
    gray_on: int = 255
    monitor_index: int = 1
    backend: str = "save_only"

    @property
    def x_start(self) -> int:
        return int(self.center_x - self.active_width // 2)

    @property
    def x_end(self) -> int:
        return int(self.x_start + self.active_width)

    @property
    def y_start(self) -> int:
        return int(self.center_y - self.active_height // 2)

    @property
    def y_end(self) -> int:
        return int(self.y_start + self.active_height)

    def validate(self) -> None:
        if self.canvas_width <= 0 or self.canvas_height <= 0:
            raise ValueError("SLM canvas dimensions must be positive")
        if self.active_width <= 0 or self.active_height <= 0:
            raise ValueError("Active region dimensions must be positive")
        if not (0 <= self.x_start < self.x_end <= self.canvas_width):
            raise ValueError("Invalid active x range")
        if not (0 <= self.y_start < self.y_end <= self.canvas_height):
            raise ValueError("Invalid active y range")


@dataclass
class ExperimentConfig:
    macro_pixel_sizes: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64])
    challenge_ids: list[str] = field(
        default_factory=lambda: [str(i) for i in range(1, 9)]
    )
    challenge_names: list[str] = field(default_factory=lambda: ["A", "B"])
    num_loads_per_challenge: int = 2
    global_seed: int = 20260711
    input_hd_min: float = 0.45
    input_hd_max: float = 0.55
    settle_time_ms: int = 500
    video_duration_seconds: float = 28.0
    root_dir: str = "."


@dataclass
class CameraConfig:
    backend: str = "manual_video"
    camera_index: int = 0
    roi: list[int] | None = None
    requested_fps: float | None = 30.0
    requested_width: int | None = None
    requested_height: int | None = None
    exposure: float | None = None
    gain: float | None = None
    output_mode: str = "frames"
    output_codec: str = "FFV1"


@dataclass
class AnalysisConfig:
    # Green-only macro-pixel screening
    mode: str = "green_only"
    window_edges_seconds: list[float] = field(default_factory=lambda: [3.0, 11.0, 19.0, 27.0])
    frames_per_window: int = 70
    dark_artifact_path: str | None = None
    dark_video_path: str | None = None
    dark_green_scalar: float = 0.0
    valid_mask_path: str | None = None
    valid_mask_percentile: float = 10.0
    valid_mask_abs_floor: float = 5.0
    envelope_sigma_px: float | None = None
    envelope_sigma_factor: float = 4.0
    envelope_eps: float = 1.0
    sensitivity_sigma_factors: list[float] = field(default_factory=lambda: [3.0, 4.0, 5.0])
    saturation_threshold: float = 250.0
    max_green_saturation: float = 0.005
    min_intra_median_detail: float = 0.80
    allow_partial_macros: bool = True
    # Legacy fields kept for YAML compatibility
    discard_initial_seconds: float = 3.0
    sample_interval_seconds: float = 1.0
    template_method: str = "mean"
    dark_frame_path: str | None = None


@dataclass
class AppConfig:
    slm: SlmConfig
    experiment: ExperimentConfig
    camera: CameraConfig
    analysis: AnalysisConfig
    config_path: Path

    @property
    def root(self) -> Path:
        root = Path(self.experiment.root_dir)
        if not root.is_absolute():
            root = (self.config_path.parent / root).resolve()
        return root


def _merge_dataclass(cls: type, raw: dict[str, Any] | None) -> Any:
    raw = raw or {}
    valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in raw.items() if k in valid}
    return cls(**filtered)


def load_config(path: Path | str) -> AppConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    cfg = AppConfig(
        slm=_merge_dataclass(SlmConfig, raw.get("slm")),
        experiment=_merge_dataclass(ExperimentConfig, raw.get("experiment")),
        camera=_merge_dataclass(CameraConfig, raw.get("camera")),
        analysis=_merge_dataclass(AnalysisConfig, raw.get("analysis")),
        config_path=config_path,
    )
    cfg.slm.validate()
    if len(cfg.analysis.window_edges_seconds) != 4:
        raise ValueError("analysis.window_edges_seconds must have 4 edges for 3 windows")
    return cfg
