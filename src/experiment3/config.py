"""Load Experiment 3 YAML configuration into dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentInfo:
    name: str = "experiment3_identity_state_puf"
    num_devices: int = 15
    device_ids: list[str] = field(
        default_factory=lambda: [f"F{i:02d}" for i in range(1, 16)]
    )
    state_ids: list[str] = field(default_factory=lambda: ["S0", "S1", "S2"])
    round_ids: list[str] = field(default_factory=lambda: ["A", "B"])
    challenge_ids: list[str] = field(
        default_factory=lambda: [f"C{i:02d}" for i in range(1, 9)]
    )
    enrollment_state: str = "S0"
    query_states: list[str] = field(default_factory=lambda: ["S1", "S2"])


@dataclass
class SplitsConfig:
    development_devices: list[str] = field(
        default_factory=lambda: [f"F{i:02d}" for i in range(1, 6)]
    )
    frozen_test_devices: list[str] = field(
        default_factory=lambda: [f"F{i:02d}" for i in range(6, 16)]
    )
    pilot_development_only: bool = True


@dataclass
class AcquisitionConfig:
    expected_frame_rate_fps: float = 10.0
    recording_duration_s: float = 25.0
    discard_head_s: float = 10.0
    discard_tail_s: float = 10.0
    num_time_blocks: int = 3
    block_aggregation: str = "median"
    recording_aggregation: str = "median"
    expected_frame_width_px: int | None = None
    expected_frame_height_px: int | None = None


@dataclass
class AnalysisConfig:
    envelope_sigma_px: float = 42.0
    envelope_eps: float = 1.0
    feature_type: str = "detail_cm"
    preprocessing_version: str = "threshold_development_v1"
    saturation_threshold: float = 250.0
    max_saturation_fraction: float = 0.02
    min_mean_intensity: float = 1.0
    min_mask_coverage: float = 0.05
    min_image_variance: float = 1.0e-6
    valid_mask_percentile: float = 10.0
    valid_mask_abs_floor: float = 5.0
    red_primary_repr: str = "lowdim_stats"
    red_primary_distance: str = "euclidean"
    red_primary_pca_dim: int | None = None
    psd_radial_bins: int = 24
    frequency_bands: dict[str, list[float]] = field(
        default_factory=lambda: {
            "low": [0.00, 0.15],
            "mid": [0.15, 0.35],
            "high": [0.35, 0.50],
        }
    )


@dataclass
class DarkFrameConfig:
    allow_missing_dark: bool = True
    dark_video: str | None = None
    dark_artifact: str | None = None
    aggregation: str = "mean"
    cache_path: str = "outputs/threshold_development/templates/dark/dark_cache.npz"


@dataclass
class CameraConfig:
    roi: list[int] | None = None
    color_format: str = "opencv_bgr"
    channel_convention: str = "opencv_bgr"


@dataclass
class StatisticsConfig:
    random_seed: int = 42
    bootstrap_iterations: int = 2000
    permutation_iterations: int = 1000
    min_devices_for_frozen_test: int = 6
    min_devices_for_top3: int = 3
    min_devices_for_bootstrap: int = 5
    min_devices_for_permutation: int = 5


@dataclass
class QualityControlConfig:
    strict_mode: bool = False
    reject_on_saturation: bool = True
    reject_on_absent_signal: bool = True
    reject_on_wrong_resolution: bool = True
    reject_on_insufficient_frames: bool = True


@dataclass
class ThresholdsConfig:
    selection_mode: str = "eer_development"
    tau_R: float | None = None
    tau_G: float | None = None


@dataclass
class PathsConfig:
    root: str = "."
    metadata_csv: str = "data/experiment3_metadata.csv"
    output_dir: str = "outputs/threshold_development"
    videos_root: str = "videos"


@dataclass
class Experiment3Config:
    experiment: ExperimentInfo
    splits: SplitsConfig
    acquisition: AcquisitionConfig
    analysis: AnalysisConfig
    dark_frame: DarkFrameConfig
    camera: CameraConfig
    statistics: StatisticsConfig
    quality_control: QualityControlConfig
    thresholds: ThresholdsConfig
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
    def videos_root(self) -> Path:
        return self.resolve_path(self.paths.videos_root)  # type: ignore[return-value]

    @property
    def dark_video_path(self) -> Path | None:
        return self.resolve_path(self.dark_frame.dark_video)

    @property
    def dark_artifact_path(self) -> Path | None:
        return self.resolve_path(self.dark_frame.dark_artifact)

    @property
    def dark_cache_path(self) -> Path | None:
        return self.resolve_path(self.dark_frame.cache_path)


def _merge(cls: type, raw: dict[str, Any] | None) -> Any:
    raw = raw or {}
    valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in raw.items() if k in valid}
    return cls(**filtered)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(
    path: Path | str,
    *,
    paths_override: Path | str | None = None,
) -> Experiment3Config:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if paths_override is not None:
        with Path(paths_override).open("r", encoding="utf-8") as handle:
            override = yaml.safe_load(handle) or {}
        raw = _deep_merge(raw, override)

    cfg = Experiment3Config(
        experiment=_merge(ExperimentInfo, raw.get("experiment")),
        splits=_merge(SplitsConfig, raw.get("splits")),
        acquisition=_merge(AcquisitionConfig, raw.get("acquisition")),
        analysis=_merge(AnalysisConfig, raw.get("analysis")),
        dark_frame=_merge(DarkFrameConfig, raw.get("dark_frame")),
        camera=_merge(CameraConfig, raw.get("camera")),
        statistics=_merge(StatisticsConfig, raw.get("statistics")),
        quality_control=_merge(QualityControlConfig, raw.get("quality_control")),
        thresholds=_merge(ThresholdsConfig, raw.get("thresholds")),
        paths=_merge(PathsConfig, raw.get("paths")),
        config_path=config_path,
    )

    if cfg.acquisition.num_time_blocks != 3:
        raise ValueError("Experiment 3 requires exactly 3 time blocks")
    if cfg.acquisition.block_aggregation != "median":
        raise ValueError("Experiment 3 requires block_aggregation = median")
    if cfg.analysis.red_primary_pca_dim is not None:
        raise ValueError("Primary red analysis forbids PCA (red_primary_pca_dim must be null)")
    if abs(cfg.analysis.envelope_sigma_px - 42.0) > 1e-9:
        raise ValueError("Experiment 3 requires envelope_sigma_px = 42")
    if abs(cfg.analysis.envelope_eps - 1.0) > 1e-9:
        raise ValueError("Experiment 3 requires envelope_eps = 1")
    if abs(cfg.acquisition.discard_head_s - 10.0) > 1e-9:
        raise ValueError("Experiment 3 requires discard_head_s = 10")
    if abs(cfg.acquisition.discard_tail_s - 10.0) > 1e-9:
        raise ValueError("Experiment 3 requires discard_tail_s = 10")
    return cfg


def config_to_dict(cfg: Experiment3Config) -> dict[str, Any]:
    """Serialize config for run_config_snapshot.yaml (no Path objects)."""
    from dataclasses import asdict

    d = asdict(cfg)
    d["config_path"] = str(cfg.config_path)
    return d
