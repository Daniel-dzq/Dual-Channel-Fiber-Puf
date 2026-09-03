"""Challenge image validation for Experiment 2."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from experiment2.config import Experiment2Config


@dataclass
class ChallengeValidationResult:
    challenge_id: str
    path: str
    valid: bool
    width: int
    height: int
    duty_cycle: float
    unique_levels: list[int]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ChallengeValidationReport:
    valid: bool
    results: list[ChallengeValidationResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "num_challenges": len(self.results),
            "results": [
                {
                    "challenge_id": r.challenge_id,
                    "path": r.path,
                    "valid": r.valid,
                    "duty_cycle": r.duty_cycle,
                    "errors": r.errors,
                    "warnings": r.warnings,
                }
                for r in self.results
            ],
        }


def discover_challenge_paths(cfg: Experiment2Config) -> dict[str, Path]:
    source = cfg.challenges_dir
    if not source.exists():
        raise FileNotFoundError(f"Challenge source_dir does not exist: {source}")

    found: dict[str, Path] = {}
    for cid in cfg.slm.challenge_ids:
        candidates = [
            source / f"{cid}" / f"mp002_{cid}.png",
            source / f"mp002_{cid}.png",
            *sorted(source.rglob(f"mp002_{cid}.png")),
            *sorted(source.rglob(f"*{cid}*.png")),
        ]
        chosen = None
        for path in candidates:
            if path.exists() and path.is_file():
                chosen = path
                break
        if chosen is None:
            raise FileNotFoundError(
                f"Required challenge file missing for {cid} under {source}"
            )
        found[cid] = chosen.resolve()
    return found


def extract_active_challenge(
    image: np.ndarray,
    cfg: Experiment2Config,
) -> np.ndarray:
    h, w = image.shape[:2]
    target = cfg.slm.challenge_width_px
    if h == target and w == target:
        return image
    if (
        h == cfg.slm.display_height_px
        and w == cfg.slm.display_width_px
    ):
        x, y, tw, th = cfg.slm.active_region
        crop = image[y : y + th, x : x + tw]
        if crop.shape[:2] != (target, target):
            raise ValueError(
                f"Active region crop shape {crop.shape[:2]} != ({target}, {target})"
            )
        return crop
    raise ValueError(
        f"Challenge image shape {image.shape[:2]} is neither "
        f"{target}x{target} nor full SLM canvas "
        f"{cfg.slm.display_width_px}x{cfg.slm.display_height_px}"
    )


def validate_macro_blocks(binary: np.ndarray, macro_size: int) -> list[str]:
    errors: list[str] = []
    h, w = binary.shape
    if h % macro_size != 0 or w % macro_size != 0:
        errors.append(
            f"Image size {h}x{w} not divisible by macro_size {macro_size}"
        )
        return errors
    for r in range(0, h, macro_size):
        for c in range(0, w, macro_size):
            block = binary[r : r + macro_size, c : c + macro_size]
            if not np.all(block == block.flat[0]):
                errors.append(
                    f"Broken {macro_size}x{macro_size} block at row={r}, col={c}"
                )
                break
        if errors:
            break
    return errors


def validate_challenge_image(
    path: Path,
    challenge_id: str,
    cfg: Experiment2Config,
) -> ChallengeValidationResult:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        else:
            # Use luminance-free check: all channels must match for binary SLM pattern
            if not (np.all(arr[:, :, 0] == arr[:, :, 1]) and np.all(arr[:, :, 1] == arr[:, :, 2])):
                return ChallengeValidationResult(
                    challenge_id=challenge_id,
                    path=str(path),
                    valid=False,
                    width=arr.shape[1],
                    height=arr.shape[0],
                    duty_cycle=float("nan"),
                    unique_levels=[],
                    errors=["Non-grayscale color challenge image is not allowed"],
                )
            arr = arr[:, :, 0]

    try:
        active = extract_active_challenge(arr, cfg)
    except ValueError as exc:
        return ChallengeValidationResult(
            challenge_id=challenge_id,
            path=str(path),
            valid=False,
            width=arr.shape[1],
            height=arr.shape[0],
            duty_cycle=float("nan"),
            unique_levels=[],
            errors=[str(exc)],
        )

    unique = sorted(int(v) for v in np.unique(active))
    allowed = set(cfg.challenges.allowed_levels)
    errors: list[str] = []
    warnings: list[str] = []

    if len(unique) != 2:
        errors.append(
            f"Challenge must be binary with exactly two levels; got {unique}"
        )
    elif set(unique) != allowed:
        errors.append(
            f"Challenge levels {unique} do not match allowed_levels "
            f"{sorted(allowed)}"
        )

    # Detect interpolation artifacts: only allowed levels should appear
    if any(v not in allowed for v in unique):
        errors.append("Non-binary or interpolated pixel values detected")

    binary = (active >= 128).astype(np.uint8)
    duty = float(np.mean(binary))
    target = cfg.challenges.duty_cycle_target
    tol = cfg.challenges.duty_cycle_tolerance
    if abs(duty - target) > tol:
        errors.append(
            f"Duty cycle {duty:.4f} outside target {target} +/- {tol}"
        )

    errors.extend(validate_macro_blocks(active, cfg.challenges.macro_size_px))

    return ChallengeValidationResult(
        challenge_id=challenge_id,
        path=str(path),
        valid=len(errors) == 0,
        width=active.shape[1],
        height=active.shape[0],
        duty_cycle=duty,
        unique_levels=unique,
        errors=errors,
        warnings=warnings,
    )


def validate_all_challenges(cfg: Experiment2Config) -> ChallengeValidationReport:
    paths = discover_challenge_paths(cfg)
    report = ChallengeValidationReport(valid=True)
    for cid in cfg.slm.challenge_ids:
        result = validate_challenge_image(paths[cid], cid, cfg)
        report.results.append(result)
        if not result.valid:
            report.valid = False
            report.errors.extend(
                [f"{cid}: {e}" for e in result.errors]
            )
    return report


def write_challenge_validation_csv(
    report: ChallengeValidationReport,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "challenge_id",
        "path",
        "valid",
        "width",
        "height",
        "duty_cycle",
        "unique_levels",
        "errors",
        "warnings",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in report.results:
            writer.writerow(
                {
                    "challenge_id": result.challenge_id,
                    "path": result.path,
                    "valid": result.valid,
                    "width": result.width,
                    "height": result.height,
                    "duty_cycle": result.duty_cycle,
                    "unique_levels": ";".join(str(v) for v in result.unique_levels),
                    "errors": "; ".join(result.errors),
                    "warnings": "; ".join(result.warnings),
                }
            )


def write_challenge_validation_json(
    report: ChallengeValidationReport,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2)
