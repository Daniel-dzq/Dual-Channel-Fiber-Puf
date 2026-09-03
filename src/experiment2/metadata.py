"""Metadata CSV loading and validation for Experiment 2."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from experiment2.config import Experiment2Config

REQUIRED_COLUMNS = [
    "dataset",
    "experiment",
    "device_id",
    "session_id",
    "record_type",
    "channel",
    "challenge_id",
    "challenge_seed",
    "macro_size_px",
    "wavelength_nm",
    "video_path",
    "frame_rate_fps",
    "exposure_ms",
    "gain",
    "input_power",
    "capture_order",
    "camera_id",
    "frame_width_px",
    "frame_height_px",
    "bit_depth",
    "color_format",
]

VALID_RECORD_TYPES = {"red_before", "green_challenge", "red_after"}
VALID_CHANNELS = {"red", "green"}
SUPPORTED_COLOR_FORMATS = {"opencv_bgr", "bgr"}


@dataclass
class MetadataRecord:
    dataset: str
    experiment: str
    device_id: str
    session_id: str
    record_type: str
    channel: str
    challenge_id: str
    challenge_seed: str
    macro_size_px: int
    wavelength_nm: int
    video_path: str
    frame_rate_fps: float
    exposure_ms: float
    gain: float
    input_power: str
    capture_order: int
    camera_id: str
    frame_width_px: int
    frame_height_px: int
    bit_depth: int
    color_format: str

    @property
    def resolved_video_path(self) -> Path:
        path = Path(self.video_path)
        return path


@dataclass
class MetadataValidationReport:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    records: list[MetadataRecord] = field(default_factory=list)
    devices_found: list[str] = field(default_factory=list)
    record_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "devices_found": self.devices_found,
            "record_counts": self.record_counts,
            "num_records": len(self.records),
        }


def _parse_row(row: dict[str, str], cfg: Experiment2Config) -> MetadataRecord:
    missing = [c for c in REQUIRED_COLUMNS if c not in row]
    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")

    return MetadataRecord(
        dataset=row["dataset"].strip(),
        experiment=row["experiment"].strip(),
        device_id=row["device_id"].strip(),
        session_id=row["session_id"].strip(),
        record_type=row["record_type"].strip(),
        channel=row["channel"].strip(),
        challenge_id=row["challenge_id"].strip(),
        challenge_seed=row["challenge_seed"].strip(),
        macro_size_px=int(float(row["macro_size_px"])),
        wavelength_nm=int(float(row["wavelength_nm"])),
        video_path=row["video_path"].strip(),
        frame_rate_fps=float(row["frame_rate_fps"]),
        exposure_ms=float(row["exposure_ms"]),
        gain=float(row["gain"]),
        input_power=row["input_power"].strip(),
        capture_order=int(float(row["capture_order"])),
        camera_id=row["camera_id"].strip(),
        frame_width_px=int(float(row["frame_width_px"])),
        frame_height_px=int(float(row["frame_height_px"])),
        bit_depth=int(float(row["bit_depth"])),
        color_format=row["color_format"].strip().lower(),
    )


def _resolve_video_path(record: MetadataRecord, cfg: Experiment2Config) -> Path:
    path = Path(record.video_path)
    if not path.is_absolute():
        path = (cfg.root / path).resolve()
    return path


def load_metadata_csv(path: Path, cfg: Experiment2Config) -> list[MetadataRecord]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Metadata CSV has no header row")
        return [_parse_row(row, cfg) for row in reader]


def validate_metadata(
    records: list[MetadataRecord],
    cfg: Experiment2Config,
    *,
    check_files: bool = True,
) -> MetadataValidationReport:
    report = MetadataValidationReport(valid=True, records=records)
    expected_devices = set(cfg.experiment.device_ids)
    expected_challenges = set(cfg.slm.challenge_ids)

    seen_keys: set[tuple[str, str, str]] = set()
    by_device: dict[str, dict[str, MetadataRecord]] = {
        d: {} for d in expected_devices
    }

    frame_dims: set[tuple[int, int]] = set()
    exposure_gain: list[tuple[float, float]] = []

    for rec in records:
        key = (rec.device_id, rec.record_type, rec.challenge_id)
        if key in seen_keys:
            report.errors.append(
                f"Duplicate record for device={rec.device_id} "
                f"record_type={rec.record_type} challenge_id={rec.challenge_id}"
            )
            report.valid = False
        seen_keys.add(key)

        if rec.device_id not in expected_devices:
            report.errors.append(f"Unknown device_id: {rec.device_id}")
            report.valid = False
            continue

        if rec.record_type not in VALID_RECORD_TYPES:
            report.errors.append(
                f"Invalid record_type '{rec.record_type}' for device {rec.device_id}"
            )
            report.valid = False

        if rec.channel not in VALID_CHANNELS:
            report.errors.append(
                f"Invalid channel '{rec.channel}' for device {rec.device_id}"
            )
            report.valid = False

        if rec.color_format not in SUPPORTED_COLOR_FORMATS:
            report.errors.append(
                f"Unsupported color_format '{rec.color_format}' for device {rec.device_id}"
            )
            report.valid = False

        if rec.macro_size_px != cfg.slm.macro_size_px:
            report.errors.append(
                f"Invalid macro_size_px={rec.macro_size_px} for device {rec.device_id}; "
                f"expected {cfg.slm.macro_size_px}"
            )
            report.valid = False

        if rec.record_type in {"red_before", "red_after"}:
            if rec.channel != "red":
                report.errors.append(
                    f"{rec.record_type} must use red channel (device {rec.device_id})"
                )
                report.valid = False
            if rec.wavelength_nm != cfg.optics.red_wavelength_nm:
                report.errors.append(
                    f"Red record wavelength {rec.wavelength_nm} != "
                    f"{cfg.optics.red_wavelength_nm} for device {rec.device_id}"
                )
                report.valid = False
            if rec.challenge_id:
                report.warnings.append(
                    f"Red record has non-empty challenge_id for device {rec.device_id}"
                )
        elif rec.record_type == "green_challenge":
            if rec.channel != "green":
                report.errors.append(
                    f"green_challenge must use green channel (device {rec.device_id})"
                )
                report.valid = False
            if rec.wavelength_nm != cfg.optics.green_wavelength_nm:
                report.errors.append(
                    f"Green record wavelength {rec.wavelength_nm} != "
                    f"{cfg.optics.green_wavelength_nm} for device {rec.device_id}"
                )
                report.valid = False
            if rec.challenge_id not in expected_challenges:
                report.errors.append(
                    f"Unknown challenge_id '{rec.challenge_id}' for device {rec.device_id}"
                )
                report.valid = False

        frame_dims.add((rec.frame_width_px, rec.frame_height_px))
        exposure_gain.append((rec.exposure_ms, rec.gain))

        if check_files:
            video_path = _resolve_video_path(rec, cfg)
            if not video_path.exists():
                report.errors.append(f"Missing video file: {video_path}")
                report.valid = False

        by_device.setdefault(rec.device_id, {})
        store_key = rec.challenge_id if rec.record_type == "green_challenge" else rec.record_type
        if store_key in by_device[rec.device_id]:
            report.errors.append(
                f"Multiple records for device {rec.device_id} key {store_key}"
            )
            report.valid = False
        by_device[rec.device_id][store_key] = rec

    if len(frame_dims) > 1:
        report.errors.append(f"Inconsistent frame dimensions across records: {frame_dims}")
        report.valid = False

    for device_id in expected_devices:
        device_records = by_device.get(device_id, {})
        if "red_before" not in device_records:
            report.errors.append(f"Missing red_before for device {device_id}")
            report.valid = False
        if "red_after" not in device_records:
            report.errors.append(f"Missing red_after for device {device_id}")
            report.valid = False
        for cid in expected_challenges:
            if cid not in device_records:
                report.errors.append(
                    f"Missing green challenge {cid} for device {device_id}"
                )
                report.valid = False

    missing_devices = expected_devices - set(by_device.keys())
    for device_id in sorted(missing_devices):
        report.errors.append(f"Missing all records for device {device_id}")
        report.valid = False

    report.devices_found = sorted(by_device.keys())
    report.record_counts = {
        d: len(recs) for d, recs in by_device.items()
    }
    return report


def write_metadata_validation_json(
    report: MetadataValidationReport,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2)
