"""YAML config loader (same style as experiment_02 / experiment_03)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class PathsConfig:
    root: Path = Path(".")
    videos_root: Path = Path("videos")
    output_dir: Path = Path("outputs")
    dark_video: Path | None = None


@dataclass
class AttributionConfig:
    fixed_input_pattern_confirmed: bool = False
    same_camera_geometry_confirmed: bool = False
    same_package_state_confirmed: bool = False
    comparable_polarization_confirmed: bool = False
    comparable_optical_path_documented: bool = False

    @property
    def causal_factorial_interpretation_valid(self) -> bool:
        return all(
            [
                self.fixed_input_pattern_confirmed,
                self.same_camera_geometry_confirmed,
                self.same_package_state_confirmed,
                self.comparable_polarization_confirmed,
                self.comparable_optical_path_documented,
            ]
        )


@dataclass
class AnalysisConfig:
    analysis_start_s: float = 5.0
    analysis_end_margin_s: float = 5.0
    max_sampled_frames: int = 60
    minimum_retained_frames: int = 24
    n_temporal_blocks: int = 3
    crop_size_px: int = 512
    analysis_inner_size_px: int = 384
    envelope_sigma_px: float = 42.0
    epsilon: float = 1.0
    background_edge_fraction: float = 0.08
    centroid_smooth_sigma_px: float = 24.0
    high_frequency_threshold_cyc_per_px: float = 0.1
    bootstrap_iterations: int = 10000
    bootstrap_seed: int = 42
    mixed_model_enabled: bool = True


@dataclass
class QCConfig:
    saturation_warning_fraction: float = 0.02
    minimum_retained_frames: int = 24
    minimum_roi_coverage: float = 0.95
    minimum_signal_to_background_ratio: float = 5.0
    dark_fraction_warning: float = 0.50
    max_centroid_border_margin_px: float = 16.0


@dataclass
class SensitivityConfig:
    crop_inner_pairs: list[list[int]] = field(
        default_factory=lambda: [[384, 320], [640, 512]]
    )
    envelope_sigmas_px: list[float] = field(default_factory=lambda: [32.0, 42.0, 64.0])


@dataclass
class ExpectedConfig:
    n_fibers: int = 5
    n_wavelengths: int = 2
    n_geometries: int = 2
    n_repeats: int = 3
    n_videos: int = 60


@dataclass
class Experiment02bConfig:
    experiment_id: str = "experiment_02b_factorial_optical_control"
    paths: PathsConfig = field(default_factory=PathsConfig)
    attribution: AttributionConfig = field(default_factory=AttributionConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)
    expected: ExpectedConfig = field(default_factory=ExpectedConfig)
    package_root: Path = field(default_factory=lambda: Path("."))

    def resolve(self) -> "Experiment02bConfig":
        root = self.package_root.resolve()
        p = self.paths
        p.root = (root / p.root).resolve() if not Path(p.root).is_absolute() else Path(p.root)
        p.videos_root = (
            (root / p.videos_root).resolve()
            if not Path(p.videos_root).is_absolute()
            else Path(p.videos_root)
        )
        p.output_dir = (
            (root / p.output_dir).resolve()
            if not Path(p.output_dir).is_absolute()
            else Path(p.output_dir)
        )
        if p.dark_video is not None:
            dv = Path(p.dark_video)
            p.dark_video = dv if dv.is_absolute() else (root / dv).resolve()
        return self


def _fill_dataclass(cls: type, data: dict[str, Any] | None):
    data = data or {}
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        if f.name.endswith("_px") or f.name in {"root", "videos_root", "output_dir", "dark_video"}:
            if val is not None and f.type in (Path, "Path", "Path | None") or "Path" in str(f.type):
                if f.name == "dark_video" and val in (None, "null", ""):
                    kwargs[f.name] = None
                else:
                    kwargs[f.name] = Path(val) if val is not None else None
                continue
        kwargs[f.name] = val
    return cls(**kwargs)


def load_config(
    path: str | Path,
    *,
    paths_override: str | Path | None = None,
    package_root: str | Path | None = None,
) -> Experiment02bConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    if paths_override is not None:
        ov = yaml.safe_load(Path(paths_override).read_text()) or {}
        raw = _deep_merge(raw, ov)

    pkg = Path(package_root) if package_root else path.resolve().parents[1]
    cfg = Experiment02bConfig(
        experiment_id=raw.get("experiment_id", "experiment_02b_factorial_optical_control"),
        paths=_fill_dataclass(PathsConfig, raw.get("paths")),
        attribution=_fill_dataclass(AttributionConfig, raw.get("attribution")),
        analysis=_fill_dataclass(AnalysisConfig, raw.get("analysis")),
        qc=_fill_dataclass(QCConfig, raw.get("qc")),
        sensitivity=_fill_dataclass(SensitivityConfig, raw.get("sensitivity")),
        expected=_fill_dataclass(ExpectedConfig, raw.get("expected")),
        package_root=pkg,
    )
    return cfg.resolve()
