"""YAML config loading for Experiment 4."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from experiment4_security.common.frozen_protocol import (
    DEVELOPMENT_DEVICES,
    FROZEN_TEST_DEVICES,
)


def package_root() -> Path:
    """Repository root (the directory that contains ``src/``, ``configs/`` and ``data/``)."""
    return Path(__file__).resolve().parents[3]


def repo_root() -> Path:
    return package_root()


def load_yaml(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {p}")
    return data


def resolve_path(base: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    if p.is_absolute():
        return p
    cand = (base / p).resolve()
    if cand.exists():
        return cand
    return (repo_root() / p).resolve()


@dataclass
class ChallengeLibraryConfig:
    output_root: Path
    master_seed: int = 20260719
    macro_pixel_size: int = 2
    canvas_width_px: int = 1024
    canvas_height_px: int = 768
    active_width_px: int = 512
    active_height_px: int = 512
    active_offset_x_px: int = 256
    active_offset_y_px: int = 128
    closed_level: int = 0
    open_level: int = 255
    n_challenges: int = 128
    max_abs_input_ncc: float = 0.02
    max_abs_hd_z: float = 3.5
    overwrite: bool = False
    canonical_source_glob: str = (
        "data/challenge_patterns/mp002/mp002_C{idx:02d}.png"
    )
    version: str = "m2_128_v1"

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ChallengeLibraryConfig":
        raw = load_yaml(path)
        pkg = package_root()
        g = raw.get("geometry", {})
        v = raw.get("validation", {})
        version = str(raw.get("version", "m2_128_v1"))
        # Always create new libraries under this package (do not fall back to repo root).
        out = pkg / Path(raw.get("output_root", f"data/challenges/{version}"))
        return cls(
            output_root=out,
            master_seed=int(raw.get("master_seed", 20260719)),
            macro_pixel_size=int(g.get("macro_pixel_size", 2)),
            canvas_width_px=int(g.get("canvas_width_px", 1024)),
            canvas_height_px=int(g.get("canvas_height_px", 768)),
            active_width_px=int(g.get("active_width_px", 512)),
            active_height_px=int(g.get("active_height_px", 512)),
            active_offset_x_px=int(g.get("active_offset_x_px", 256)),
            active_offset_y_px=int(g.get("active_offset_y_px", 128)),
            closed_level=int(g.get("closed_level", 0)),
            open_level=int(g.get("open_level", 255)),
            n_challenges=int(raw.get("n_challenges", 128)),
            max_abs_input_ncc=float(v.get("max_abs_input_ncc", 0.02)),
            max_abs_hd_z=float(v.get("max_abs_hd_z", 3.5)),
            overwrite=bool(raw.get("overwrite", False)),
            canonical_source_glob=str(
                raw.get(
                    "canonical_source_glob",
                    "data/challenge_patterns/mp002/mp002_C{idx:02d}.png",
                )
            ),
            version=version,
        )


@dataclass
class LifecycleConfig:
    threshold_development_root: Path
    metadata_csv: Path
    videos_root: Path
    output_dir: Path
    development_devices: list[str] = field(
        default_factory=lambda: list(DEVELOPMENT_DEVICES)
    )
    frozen_test_devices: list[str] = field(
        default_factory=lambda: list(FROZEN_TEST_DEVICES)
    )
    envelope_sigma_px: float = 42.0
    envelope_eps: float = 1.0
    dry_run: bool = False
    random_seed: int = 42
    allow_missing_dark: bool = True

    @classmethod
    def from_yaml(cls, path: Path | str) -> "LifecycleConfig":
        raw = load_yaml(path)
        pkg = package_root()
        e3 = resolve_path(pkg, raw.get("threshold_development_root", "data/raw/threshold_development"))
        meta = resolve_path(pkg, raw.get("metadata_csv")) or (
            e3 / "data" / "experiment3_metadata.csv"
        )
        vids = resolve_path(pkg, raw.get("videos_root")) or (e3 / "videos")
        out = pkg / Path(raw.get("output_dir", "outputs/experiment4/lifecycle"))
        splits = raw.get("splits", {})
        analysis = raw.get("analysis", {})
        return cls(
            threshold_development_root=e3,
            metadata_csv=meta,
            videos_root=vids,
            output_dir=out,
            development_devices=list(splits.get("development_devices", DEVELOPMENT_DEVICES)),
            frozen_test_devices=list(splits.get("frozen_test_devices", FROZEN_TEST_DEVICES)),
            envelope_sigma_px=float(analysis.get("envelope_sigma_px", 42.0)),
            envelope_eps=float(analysis.get("envelope_eps", 1.0)),
            dry_run=bool(raw.get("dry_run", False)),
            random_seed=int(raw.get("random_seed", 42)),
            allow_missing_dark=bool(raw.get("allow_missing_dark", True)),
        )
