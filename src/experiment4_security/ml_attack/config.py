"""YAML configuration for registered-database green-credential security analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {p}")
    return data


@dataclass
class VideoProtocolConfig:
    nominal_duration_s: float = 8.0
    use_all_decoded_frames: bool = True
    discard_head_s: float = 0.0
    discard_tail_s: float = 0.0
    minimum_duration_s: float = 7.5
    maximum_duration_s: float = 10.5
    minimum_frames_used: int = 20


@dataclass
class QCThresholdsConfig:
    duration_warn_below_s: float = 7.5
    duration_error_below_s: float = 5.0
    duration_warn_above_s: float = 10.5
    saturation_warn_fraction: float = 0.02
    saturation_error_fraction: float = 0.10
    half_split_ncc_warn_below: float = 0.85
    half_split_ncc_error_below: float = 0.5
    relative_intensity_drift_warn: float = 0.15
    relative_intensity_drift_error: float = 0.40
    centroid_drift_warn_px: float = 5.0
    centroid_drift_error_px: float = 20.0
    min_frames_used_error_below: int = 20


@dataclass
class PreprocessingConfig:
    dark_mode: str = "none"
    envelope_sigma: float = 42.0
    envelope_epsilon: float = 1.0
    primary_representation: str = "fullres_detail_cm"
    optional_representation: str = "fullres_detail"


@dataclass
class PCAConfig:
    # Fixed PCA dimension -- NOT selected on Round B.
    dimension: int = 64


@dataclass
class ModelsConfig:
    mean_baseline: bool = True
    exact_template_replay: bool = True
    ridge: bool = True
    kernel_ridge: bool = True
    random_fourier_ridge: bool = True
    small_mlp: bool = True
    # Fixed hyperparameters (no Round-B selection).
    ridge_alpha: float = 10.0
    kernel_ridge_alpha: float = 1.0
    kernel_ridge_gamma: float = 0.01
    rff_n_components: int = 256
    rff_gamma: float = 0.01
    rff_alpha: float = 1.0
    mlp_hidden_sizes: tuple[int, ...] = (64, 32)
    mlp_max_iter: int = 150
    mlp_learning_rate: float = 1e-3
    mlp_random_seed: int = 20260721


@dataclass
class BootstrapConfig:
    n_resamples: int = 5000
    unit: str = "challenge_bank"
    bank_size: int = 8
    seed: int = 20260721


@dataclass
class EvaluationConfig:
    batch_size: int = 32
    track_b_parallel_targets: int = 4
    track_d_parallel_targets: int = 4


@dataclass
class OutputConfig:
    make_figures: bool = False
    save_predictions: bool = True
    save_models: bool = False
    save_fullres_templates: bool = True


@dataclass
class MLAttackConfig:
    project_root: Path
    videos_root: Path
    challenge_root: Path
    lifecycle_run: Path
    output_root: Path
    device_id: str = "F01"
    source_device_id: str = "F1"
    states: tuple[str, ...] = ("M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7")
    rounds: tuple[str, ...] = ("A", "B")
    challenge_id_start: int = 1
    challenge_id_end: int = 128
    video_protocol: VideoProtocolConfig = field(default_factory=VideoProtocolConfig)
    qc_thresholds: QCThresholdsConfig = field(default_factory=QCThresholdsConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    pca: PCAConfig = field(default_factory=PCAConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    # Optional path to previously validated vector cache (e.g. archived run).
    reuse_vector_cache_from: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "MLAttackConfig":
        raw = load_yaml(path)
        # Relative paths are resolved against the parent of the config directory (repository root).
        base = Path(path).resolve().parent.parent

        def _p(key: str, default: str | None = None) -> Path:
            val = raw.get(key, default)
            if val is None:
                raise ValueError(f"Missing required config key: {key}")
            p = Path(val)
            return p if p.is_absolute() else (base / p).resolve()

        vp = raw.get("video_protocol", {})
        qc = raw.get("qc_thresholds", {})
        pp = raw.get("preprocessing", {})
        pca = raw.get("pca", {})
        mdl = raw.get("models", {})
        bs = raw.get("bootstrap", {})
        ev = raw.get("evaluation", {})
        out = raw.get("output", {})
        ch = raw.get("challenge_ids", {})
        reuse = raw.get("reuse_vector_cache_from")

        return cls(
            project_root=_p("project_root"),
            videos_root=_p("videos_root"),
            challenge_root=_p("challenge_root"),
            lifecycle_run=_p("lifecycle_run"),
            output_root=_p("output_root"),
            device_id=str(raw.get("device_id", "F01")),
            source_device_id=str(raw.get("source_device_id", "F1")),
            states=tuple(raw.get("states", ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7"])),
            rounds=tuple(raw.get("rounds", ["A", "B"])),
            challenge_id_start=int(ch.get("start", 1)),
            challenge_id_end=int(ch.get("end", 128)),
            video_protocol=VideoProtocolConfig(
                nominal_duration_s=float(vp.get("nominal_duration_s", 8.0)),
                use_all_decoded_frames=bool(vp.get("use_all_decoded_frames", True)),
                discard_head_s=float(vp.get("discard_head_s", 0.0)),
                discard_tail_s=float(vp.get("discard_tail_s", 0.0)),
                minimum_duration_s=float(vp.get("minimum_duration_s", 7.5)),
                maximum_duration_s=float(vp.get("maximum_duration_s", 10.5)),
                minimum_frames_used=int(vp.get("minimum_frames_used", 20)),
            ),
            qc_thresholds=QCThresholdsConfig(
                duration_warn_below_s=float(qc.get("duration_warn_below_s", 7.5)),
                duration_error_below_s=float(qc.get("duration_error_below_s", 5.0)),
                duration_warn_above_s=float(qc.get("duration_warn_above_s", 10.5)),
                saturation_warn_fraction=float(qc.get("saturation_warn_fraction", 0.02)),
                saturation_error_fraction=float(qc.get("saturation_error_fraction", 0.10)),
                half_split_ncc_warn_below=float(qc.get("half_split_ncc_warn_below", 0.85)),
                half_split_ncc_error_below=float(qc.get("half_split_ncc_error_below", 0.5)),
                relative_intensity_drift_warn=float(qc.get("relative_intensity_drift_warn", 0.15)),
                relative_intensity_drift_error=float(qc.get("relative_intensity_drift_error", 0.40)),
                centroid_drift_warn_px=float(qc.get("centroid_drift_warn_px", 5.0)),
                centroid_drift_error_px=float(qc.get("centroid_drift_error_px", 20.0)),
                min_frames_used_error_below=int(qc.get("min_frames_used_error_below", 20)),
            ),
            preprocessing=PreprocessingConfig(
                dark_mode=str(pp.get("dark_mode", "none")),
                envelope_sigma=float(pp.get("envelope_sigma", 42.0)),
                envelope_epsilon=float(pp.get("envelope_epsilon", 1.0)),
                primary_representation=str(pp.get("primary_representation", "fullres_detail_cm_enrollment_frozen")),
                optional_representation=str(pp.get("optional_representation", "fullres_detail")),
            ),
            pca=PCAConfig(dimension=int(pca.get("dimension", pca.get("candidate_dimensions", [64])[-1] if isinstance(pca.get("candidate_dimensions"), list) else 64))),
            models=ModelsConfig(
                mean_baseline=bool(mdl.get("mean_baseline", True)),
                exact_template_replay=bool(mdl.get("exact_template_replay", mdl.get("nearest_challenge", True))),
                ridge=bool(mdl.get("ridge", True)),
                kernel_ridge=bool(mdl.get("kernel_ridge", True)),
                random_fourier_ridge=bool(mdl.get("random_fourier_ridge", True)),
                small_mlp=bool(mdl.get("small_mlp", True)),
                ridge_alpha=float(mdl.get("ridge_alpha", 10.0)),
                kernel_ridge_alpha=float(mdl.get("kernel_ridge_alpha", 1.0)),
                kernel_ridge_gamma=float(mdl.get("kernel_ridge_gamma", 0.01)),
                rff_n_components=int(mdl.get("rff_n_components", 256)),
                rff_gamma=float(mdl.get("rff_gamma", 0.01)),
                rff_alpha=float(mdl.get("rff_alpha", 1.0)),
                mlp_hidden_sizes=tuple(mdl.get("mlp_hidden_sizes", [64, 32])),
                mlp_max_iter=int(mdl.get("mlp_max_iter", 150)),
                mlp_learning_rate=float(mdl.get("mlp_learning_rate", 1e-3)),
                mlp_random_seed=int(mdl.get("mlp_random_seed", 20260721)),
            ),
            bootstrap=BootstrapConfig(
                n_resamples=int(bs.get("n_resamples", 5000)),
                unit=str(bs.get("unit", "challenge_bank")),
                bank_size=int(bs.get("bank_size", 8)),
                seed=int(bs.get("seed", 20260721)),
            ),
            evaluation=EvaluationConfig(
                batch_size=int(ev.get("batch_size", 32)),
                track_b_parallel_targets=int(ev.get("track_b_parallel_targets", 4)),
                track_d_parallel_targets=int(
                    ev.get("track_d_parallel_targets", ev.get("track_b_parallel_targets", 4))
                ),
            ),
            output=OutputConfig(
                make_figures=bool(out.get("make_figures", False)),
                save_predictions=bool(out.get("save_predictions", True)),
                save_models=bool(out.get("save_models", False)),
                save_fullres_templates=bool(out.get("save_fullres_templates", True)),
            ),
            reuse_vector_cache_from=(Path(reuse) if Path(reuse).is_absolute() else (base / reuse).resolve()) if reuse else None,
            raw=raw,
        )

    def to_resolved_dict(self) -> dict[str, Any]:
        def _path(p: Path | None) -> str | None:
            return None if p is None else str(p)

        return {
            "project_root": _path(self.project_root),
            "videos_root": _path(self.videos_root),
            "challenge_root": _path(self.challenge_root),
            "lifecycle_run": _path(self.lifecycle_run),
            "output_root": _path(self.output_root),
            "reuse_vector_cache_from": _path(self.reuse_vector_cache_from),
            "device_id": self.device_id,
            "source_device_id": self.source_device_id,
            "states": list(self.states),
            "rounds": list(self.rounds),
            "challenge_ids": {"start": self.challenge_id_start, "end": self.challenge_id_end},
            "video_protocol": vars(self.video_protocol),
            "qc_thresholds": vars(self.qc_thresholds),
            "preprocessing": vars(self.preprocessing),
            "pca": vars(self.pca),
            "models": {
                **{k: v for k, v in vars(self.models).items() if not isinstance(v, tuple)},
                **{k: list(v) for k, v in vars(self.models).items() if isinstance(v, tuple)},
            },
            "bootstrap": vars(self.bootstrap),
            "evaluation": vars(self.evaluation),
            "output": vars(self.output),
            "protocol": "registered_database_authentication_revocation_clone",
            "absolute_asr": None,
            "asr_status": "SCORE_SPACE_OR_PROTOCOL_MISMATCH",
            "make_figures": False,
        }

    def run_output_dir(self, run_id: str | None = None) -> Path:
        """Timestamped runs live under output_root/runs/<run_id>/."""
        rid = run_id or "unnamed"
        return self.output_root / "runs" / rid

    def cache_dir(self, run_id: str | None = None) -> Path:
        return self.run_output_dir(run_id) / "_cache"

    def pattern_cache_dir(self) -> Path:
        return self.output_root / "_challenge_pattern_cache"
