"""Load Experiment 2 YAML configuration into dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from puf_common.dark import CameraSettings


@dataclass
class ExperimentInfo:
    name: str = "experiment2_dual_channel_characterization"
    num_devices: int = 15
    device_ids: list[str] = field(
        default_factory=lambda: [f"F{i:02d}" for i in range(1, 16)]
    )


@dataclass
class OpticsConfig:
    red_wavelength_nm: int = 650
    green_wavelength_nm: int = 532


@dataclass
class SlmConfig:
    display_width_px: int = 1024
    display_height_px: int = 768
    challenge_width_px: int = 512
    challenge_height_px: int = 512
    active_x: int = 256
    active_y: int = 128
    macro_size_px: int = 2
    challenge_centering: str = "centered"
    interpolation: str = "nearest"
    num_challenges: int = 8
    challenge_ids: list[str] = field(
        default_factory=lambda: [f"C{i:02d}" for i in range(1, 9)]
    )
    representative_challenge_id: str = "C01"
    representative_figure_challenges: list[str] = field(
        default_factory=lambda: ["C01", "C03", "C05", "C07"]
    )

    @property
    def active_region(self) -> tuple[int, int, int, int]:
        return (
            self.active_x,
            self.active_y,
            self.challenge_width_px,
            self.challenge_height_px,
        )


@dataclass
class AcquisitionConfig:
    expected_frame_rate_fps: float = 30.0
    recording_duration_s: float = 28.0
    discard_head_s: float = 10.0
    discard_tail_s: float = 10.0
    effective_duration_s: float = 8.0
    num_time_blocks: int = 3
    block_aggregation: str = "median"
    recording_aggregation: str = "median"


@dataclass
class ChannelSpec:
    illumination_mode: str = "red_only"
    source_channel: str = "red"


@dataclass
class ChannelsConfig:
    red: ChannelSpec = field(
        default_factory=lambda: ChannelSpec("red_only", "red")
    )
    green: ChannelSpec = field(
        default_factory=lambda: ChannelSpec("green_only", "green")
    )


@dataclass
class AlignmentConfig:
    enabled: bool = False
    mode: str = "fixed_geometry_only"


@dataclass
class StatisticsConfig:
    random_seed: int = 42
    bootstrap_iterations: int = 10000


@dataclass
class AnalysisConfig:
    envelope_sigma_px: float = 42.0
    envelope_eps: float = 1.0
    feature_type: str = "detail_cm"
    preprocessing_version: str = "fixed_state_v1"
    saturation_threshold: float = 250.0
    max_saturation_fraction: float = 0.02
    min_mask_coverage: float = 0.05
    min_image_variance: float = 1.0e-6
    valid_mask_percentile: float = 10.0
    valid_mask_abs_floor: float = 5.0


@dataclass
class DarkFrameConfig:
    reuse_experiment1: bool = True
    experiment1_dark_artifact: str | None = None
    experiment1_dark_video: str | None = None
    experiment1_config: str | None = None
    experiment1_valid_mask: str | None = None
    aggregation: str = "mean"
    strict_parameter_match: bool = True
    allow_missing_dark: bool = False
    cache_path: str | None = None
    camera_settings: CameraSettings = field(default_factory=CameraSettings)


@dataclass
class ChallengesConfig:
    source_dir: str = ""
    pattern_glob: str = "mp002_C*.png"
    macro_size_px: int = 2
    duty_cycle_target: float = 0.5
    duty_cycle_tolerance: float = 0.02
    allowed_levels: list[int] = field(default_factory=lambda: [0, 255])


@dataclass
class CameraConfig:
    roi: list[int] | None = None
    color_format: str = "opencv_bgr"
    channel_convention: str = "opencv_bgr"


@dataclass
class FiguresConfig:
    representative_device_id: str = "F01"
    representative_red_record: str = "red_before"
    representative_green_challenge_id: str = "C01"
    challenge_example_ids: list[str] = field(
        default_factory=lambda: ["C01", "C03", "C05", "C07"]
    )


@dataclass
class QualityControlConfig:
    strict_mode: bool = False


@dataclass
class PathsConfig:
    root: str = "."
    metadata_csv: str = "data/experiment2_metadata.csv"
    output_dir: str = "outputs/experiment2"
    videos_root: str = "data/videos"


@dataclass
class Experiment2Config:
    experiment: ExperimentInfo
    optics: OpticsConfig
    slm: SlmConfig
    acquisition: AcquisitionConfig
    channels: ChannelsConfig
    alignment: AlignmentConfig
    statistics: StatisticsConfig
    analysis: AnalysisConfig
    dark_frame: DarkFrameConfig
    challenges: ChallengesConfig
    camera: CameraConfig
    figures: FiguresConfig
    quality_control: QualityControlConfig
    paths: PathsConfig
    config_path: Path

    @property
    def root(self) -> Path:
        root = Path(self.paths.root)
        if not root.is_absolute():
            root = (self.config_path.parent.parent / root).resolve()
        return root

    def resolve_path(self, value: str | Path | None) -> Path | None:
        if value is None or str(value).strip() == "":
            return None
        path = Path(value)
        if not path.is_absolute():
            path = (self.root / path).resolve()
        return path

    @property
    def metadata_path(self) -> Path:
        return self.resolve_path(self.paths.metadata_csv)  # type: ignore[return-value]

    @property
    def output_dir(self) -> Path:
        return self.resolve_path(self.paths.output_dir)  # type: ignore[return-value]

    @property
    def challenges_dir(self) -> Path:
        path = Path(self.challenges.source_dir)
        if not path.is_absolute():
            path = (self.root / path).resolve()
        return path

    @property
    def dark_artifact_path(self) -> Path | None:
        return self.resolve_path(self.dark_frame.experiment1_dark_artifact)

    @property
    def dark_video_path(self) -> Path | None:
        return self.resolve_path(self.dark_frame.experiment1_dark_video)

    @property
    def dark_cache_path(self) -> Path | None:
        return self.resolve_path(self.dark_frame.cache_path)

    @property
    def experiment1_valid_mask_path(self) -> Path | None:
        return self.resolve_path(self.dark_frame.experiment1_valid_mask)


def _merge(cls: type, raw: dict[str, Any] | None) -> Any:
    raw = raw or {}
    valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in raw.items() if k in valid}
    if cls is DarkFrameConfig and "camera_settings" in (raw or {}):
        filtered["camera_settings"] = CameraSettings(**(raw.get("camera_settings") or {}))
    return cls(**filtered)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(
    path: Path | str,
    *,
    paths_override: Path | str | None = None,
) -> Experiment2Config:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if paths_override is not None:
        with Path(paths_override).open("r", encoding="utf-8") as handle:
            override = yaml.safe_load(handle) or {}
        raw = _deep_merge(raw, override)

    channels_raw = raw.get("channels") or {}
    channels = ChannelsConfig(
        red=_merge(ChannelSpec, channels_raw.get("red")),
        green=_merge(ChannelSpec, channels_raw.get("green")),
    )

    cfg = Experiment2Config(
        experiment=_merge(ExperimentInfo, raw.get("experiment")),
        optics=_merge(OpticsConfig, raw.get("optics")),
        slm=_merge(SlmConfig, raw.get("slm")),
        acquisition=_merge(AcquisitionConfig, raw.get("acquisition")),
        channels=channels,
        alignment=_merge(AlignmentConfig, raw.get("alignment")),
        statistics=_merge(StatisticsConfig, raw.get("statistics")),
        analysis=_merge(AnalysisConfig, raw.get("analysis")),
        dark_frame=_merge(DarkFrameConfig, raw.get("dark_frame")),
        challenges=_merge(ChallengesConfig, raw.get("challenges")),
        camera=_merge(CameraConfig, raw.get("camera")),
        figures=_merge(FiguresConfig, raw.get("figures")),
        quality_control=_merge(QualityControlConfig, raw.get("quality_control")),
        paths=_merge(PathsConfig, raw.get("paths")),
        config_path=config_path,
    )

    if cfg.slm.macro_size_px != 2:
        raise ValueError("Experiment 2 requires macro_size_px = 2")
    if cfg.acquisition.num_time_blocks != 3:
        raise ValueError("Experiment 2 requires exactly 3 time blocks")
    if cfg.acquisition.block_aggregation != "median":
        raise ValueError("Experiment 2 requires block_aggregation = median")

    return cfg
