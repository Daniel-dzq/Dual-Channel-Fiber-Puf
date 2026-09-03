"""Configuration loading for the fiber-length optimization pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

PKG_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PKG_ROOT.parent


@dataclass
class ExperimentMeta:
    name: str = "experiment00_length_optimization"
    seed: int = 20260721
    strict_dataset: bool = True
    allow_incomplete: bool = False
    allow_missing_dark: bool = False


@dataclass
class DatasetCfg:
    expected_lengths_cm: list[int] = field(default_factory=lambda: [7, 9, 11, 13, 15])
    expected_fibers: list[str] = field(
        default_factory=lambda: ["F01", "F02", "F03", "F04", "F05"]
    )
    expected_rounds: list[str] = field(default_factory=lambda: ["A", "B"])
    expected_challenges: list[str] = field(
        default_factory=lambda: [f"C{i:02d}" for i in range(1, 9)]
    )
    challenge_macro_pixel: int = 2


@dataclass
class PathsCfg:
    videos_dir: str = "videos"
    dark_dir: str = "dark"
    outputs_dir: str = "outputs/experiment00/runs"
    dark_video: str | None = None
    dark_artifact: str | None = None


@dataclass
class VideoCfg:
    trim_start_s: float = 10.0
    trim_end_s: float = 10.0
    temporal_blocks: int = 3
    aggregation: str = "median"
    minimum_valid_frames_per_block: int = 3


@dataclass
class PreprocessingCfg:
    dark_correction: bool = True
    use_global_valid_mask: bool = True
    detail_sigma_px: float = 42.0
    detail_epsilon: float = 1.0
    green_common_mode: bool = True
    red_common_mode: bool = False
    common_mode_grouping: list[str] = field(
        default_factory=lambda: ["length_cm", "fiber_id", "round", "temporal_block"]
    )
    mask_abs_floor: float = 5.0
    mask_percentile: float = 10.0


@dataclass
class QualityCfg:
    saturation_fraction_threshold: float = 0.02
    acf_width_cap_px: float = 40.0
    minimum_valid_pixel_fraction: float = 0.50
    fail_on_resolution_mismatch: bool = True


@dataclass
class StatisticsCfg:
    bootstrap_iterations: int = 5000
    confidence_level: float = 0.95
    cluster_by_device: bool = True
    resample_challenges_within_device: bool = True
    bootstrap_iterations_smoke: int = 200


@dataclass
class LengthSelectionCfg:
    criterion: str = "maximin_robust_gap"
    use_red_as_guardrail: bool = True
    never_force_unique_optimum: bool = True
    prefer_shorter_length_inside_plateau: bool = True
    plateau_ci_overlap: bool = True
    selection_prob_unique_min: float = 0.60


@dataclass
class PlottingCfg:
    font_family: str = "Times New Roman"
    dpi: int = 600
    save_png: bool = True
    save_pdf: bool = True
    save_svg: bool = True
    add_panel_labels: bool = True
    add_figure_titles: bool = False
    skip_figures: bool = True


@dataclass
class WatcherCfg:
    enabled: bool = True
    poll_interval_s: float = 5.0
    file_stability_wait_s: float = 60.0
    require_complete_dataset: bool = True


@dataclass
class RuntimeCfg:
    n_workers: int = 1
    cache_templates: bool = True


@dataclass
class AcquisitionCfg:
    sequence_assignment: dict[str, str] = field(
        default_factory=lambda: {
            "F01": "P1",
            "F02": "P2",
            "F03": "P3",
            "F04": "P4",
            "F05": "P5",
        }
    )
    challenge_sequences: dict[str, list[str]] = field(
        default_factory=lambda: {
            "P1": ["C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08"],
            "P2": ["C03", "C04", "C05", "C06", "C07", "C08", "C01", "C02"],
            "P3": ["C05", "C06", "C07", "C08", "C01", "C02", "C03", "C04"],
            "P4": ["C07", "C08", "C01", "C02", "C03", "C04", "C05", "C06"],
            "P5": ["C02", "C04", "C06", "C08", "C01", "C03", "C05", "C07"],
        }
    )
    same_sequence_in_round_a_and_b: bool = True
    infer_order_from_file_timestamp: bool = False
    infer_order_from_directory_listing: bool = False


@dataclass
class Experiment00Config:
    experiment: ExperimentMeta = field(default_factory=ExperimentMeta)
    dataset: DatasetCfg = field(default_factory=DatasetCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)
    video: VideoCfg = field(default_factory=VideoCfg)
    preprocessing: PreprocessingCfg = field(default_factory=PreprocessingCfg)
    quality_control: QualityCfg = field(default_factory=QualityCfg)
    statistics: StatisticsCfg = field(default_factory=StatisticsCfg)
    length_selection: LengthSelectionCfg = field(default_factory=LengthSelectionCfg)
    plotting: PlottingCfg = field(default_factory=PlottingCfg)
    watcher: WatcherCfg = field(default_factory=WatcherCfg)
    runtime: RuntimeCfg = field(default_factory=RuntimeCfg)
    acquisition: AcquisitionCfg = field(default_factory=AcquisitionCfg)
    # resolved absolute paths
    root: Path = field(default_factory=lambda: PKG_ROOT)

    def resolve_path(self, value: str | Path | None) -> Path | None:
        """Resolve path relative to package root, then repo root."""
        if value is None:
            return None
        p = Path(value)
        if p.is_absolute():
            return p
        cand = self.root / p
        if cand.exists():
            return cand
        repo = self.root.parent / p
        if repo.exists():
            return repo
        return cand

    def videos_path(self) -> Path:
        return self.resolve_path(self.paths.videos_dir) or (self.root / self.paths.videos_dir)

    def dark_path(self) -> Path:
        return self.resolve_path(self.paths.dark_dir) or (self.root / self.paths.dark_dir)

    def outputs_path(self) -> Path:
        p = Path(self.paths.outputs_dir)
        return p if p.is_absolute() else self.root / p

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["root"] = str(self.root)
        return d


def _merge_section(cls, raw: dict | None):
    raw = raw or {}
    fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in raw.items() if k in fields})


def load_config(path: str | Path | None = None) -> Experiment00Config:
    cfg_path = Path(path) if path else PKG_ROOT / "configs" / "fiber_length_optimization.yaml"
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    cfg = Experiment00Config(
        experiment=_merge_section(ExperimentMeta, raw.get("experiment")),
        dataset=_merge_section(DatasetCfg, raw.get("dataset")),
        paths=_merge_section(PathsCfg, raw.get("paths")),
        video=_merge_section(VideoCfg, raw.get("video")),
        preprocessing=_merge_section(PreprocessingCfg, raw.get("preprocessing")),
        quality_control=_merge_section(QualityCfg, raw.get("quality_control")),
        statistics=_merge_section(StatisticsCfg, raw.get("statistics")),
        length_selection=_merge_section(LengthSelectionCfg, raw.get("length_selection")),
        plotting=_merge_section(PlottingCfg, raw.get("plotting")),
        watcher=_merge_section(WatcherCfg, raw.get("watcher")),
        runtime=_merge_section(RuntimeCfg, raw.get("runtime")),
        acquisition=_merge_section(AcquisitionCfg, raw.get("acquisition")),
        root=PKG_ROOT,
    )
    if int(cfg.dataset.challenge_macro_pixel) != 2:
        raise ValueError("challenge_macro_pixel must remain frozen at m=2")
    if cfg.acquisition.infer_order_from_file_timestamp:
        raise ValueError("infer_order_from_file_timestamp must remain false")
    if cfg.acquisition.infer_order_from_directory_listing:
        raise ValueError("infer_order_from_directory_listing must remain false")
    from experiment00.acquisition import validate_challenge_sequences

    validate_challenge_sequences(
        cfg.acquisition.challenge_sequences,
        expected_challenges=cfg.dataset.expected_challenges,
    )
    return cfg


def save_frozen_config(cfg: Experiment00Config, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False, allow_unicode=True)
