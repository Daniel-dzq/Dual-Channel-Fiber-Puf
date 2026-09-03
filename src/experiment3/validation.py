"""Input validation and QC flagging for Experiment 3."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from experiment3.config import Experiment3Config
from experiment3.metadata import (
    discover_from_directory,
    find_missing_and_duplicates,
    inventory,
    load_metadata_csv,
    parse_filename,
    resolve_video_paths,
)
from experiment3.video_processing import (
    compute_retained_frame_indices,
    probe_video,
    split_non_overlapping_blocks,
)


@dataclass
class ValidationResult:
    valid: bool
    metadata: pd.DataFrame
    inventory: dict[str, Any]
    missing: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    extras: list[str] = field(default_factory=list)
    qc_rows: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    is_pilot: bool = False
    development_devices: list[str] = field(default_factory=list)
    frozen_test_devices_present: list[str] = field(default_factory=list)


def _flag_row(
    *,
    device_id: str,
    state_id: str,
    channel: str,
    round_id: str,
    challenge_id: str,
    video_path: str,
    status: str,
    reason: str,
    **extra: Any,
) -> dict[str, Any]:
    row = {
        "device_id": device_id,
        "state_id": state_id,
        "channel": channel,
        "round_id": round_id,
        "challenge_id": challenge_id,
        "video_path": video_path,
        "status": status,
        "reason": reason,
    }
    row.update(extra)
    return row


def validate_metadata_rows(df: pd.DataFrame, cfg: Experiment3Config) -> list[str]:
    errors: list[str] = []
    for i, row in df.iterrows():
        ch = row["channel"]
        if ch not in {"red", "green"}:
            errors.append(f"row {i}: invalid channel {ch}")
            continue
        if ch == "red":
            if row["round_id"] or row["challenge_id"]:
                errors.append(f"row {i}: red row must not have round/challenge")
        else:
            if row["round_id"] not in cfg.experiment.round_ids:
                errors.append(f"row {i}: missing/invalid round_id for green")
            if row["challenge_id"] not in cfg.experiment.challenge_ids:
                errors.append(
                    f"row {i}: missing/invalid challenge_id {row['challenge_id']}"
                )
        try:
            parsed = parse_filename(row["video_filename"])
        except ValueError as exc:
            errors.append(f"row {i}: {exc}")
            continue
        if parsed.device_id != row["device_id"] or parsed.state_id != row["state_id"]:
            errors.append(f"row {i}: filename/device-state mismatch")
        if parsed.channel != ch:
            errors.append(f"row {i}: filename/channel mismatch")
    return errors


def validate_inputs(
    cfg: Experiment3Config,
    *,
    metadata_path: Path | None = None,
    check_files: bool = True,
    probe_videos: bool = True,
) -> ValidationResult:
    path = metadata_path or cfg.metadata_path
    errors: list[str] = []
    warnings: list[str] = []

    if path.exists():
        df = load_metadata_csv(path)
    else:
        warnings.append(f"Metadata CSV missing ({path}); discovering from videos_root")
        df = discover_from_directory(cfg.videos_root)

    if df.empty:
        errors.append("No metadata rows available")
        return ValidationResult(
            valid=False,
            metadata=df,
            inventory={"devices": [], "states": [], "rounds": [], "challenges": []},
            errors=errors,
            warnings=warnings,
        )

    df = resolve_video_paths(df, root=cfg.root, videos_root=cfg.videos_root)
    errors.extend(validate_metadata_rows(df, cfg))
    inv = inventory(df)
    miss_dup = find_missing_and_duplicates(
        df,
        devices=inv["devices"],
        states=inv["states"],
        rounds=inv["rounds"] or list(cfg.experiment.round_ids),
        challenges=inv["challenges"] or list(cfg.experiment.challenge_ids),
    )

    available = set(inv["devices"])
    configured_dev = list(cfg.splits.development_devices)
    configured_frozen = list(cfg.splits.frozen_test_devices)
    overlap = sorted(set(configured_dev) & set(configured_frozen))
    if overlap:
        errors.append(f"Configured development/frozen overlap: {overlap}")

    dev_cfg = [d for d in configured_dev if d in available]
    frozen_present = [d for d in configured_frozen if d in available]

    # Fail-closed: never silently expand development to all devices.
    if not configured_dev:
        errors.append(
            "Development device set is empty in config. "
            "Formal evaluation cannot continue."
        )
    if not cfg.splits.pilot_development_only and not configured_frozen:
        errors.append(
            "Frozen test device set is empty in config while "
            "pilot_development_only=false. Formal held-out metrics cannot be generated."
        )
    if not cfg.splits.pilot_development_only and not frozen_present and configured_frozen:
        errors.append(
            "Configured frozen-test devices are absent from the available cohort. "
            "Refusing to evaluate the full cohort as experiment4_security."
        )
    if not dev_cfg and configured_dev:
        errors.append(
            "None of the configured development devices are present in the data. "
            "Refusing all-device development fallback."
        )

    is_pilot = (
        len(available) < cfg.statistics.min_devices_for_frozen_test
        or cfg.splits.pilot_development_only
        or len(frozen_present) == 0
    )
    if is_pilot:
        warnings.append(
            "PILOT / NOT FOR FINAL CLAIMS: frozen held-out evaluation is unavailable "
            "or pilot_development_only=true. Development metrics are not independent tests."
        )

    qc_rows: list[dict[str, Any]] = []
    if check_files:
        for _, row in df.iterrows():
            vp = Path(row["video_path"])
            base = dict(
                device_id=row["device_id"],
                state_id=row["state_id"],
                channel=row["channel"],
                round_id=row["round_id"],
                challenge_id=row["challenge_id"],
                video_path=str(vp),
            )
            if not vp.exists():
                qc_rows.append(_flag_row(**base, status="fail", reason="unreadable_file"))
                errors.append(f"Missing video: {vp}")
                continue
            if not probe_videos:
                qc_rows.append(_flag_row(**base, status="ok", reason="exists_unprobed"))
                continue
            try:
                info = probe_video(vp)
            except Exception as exc:  # noqa: BLE001
                qc_rows.append(
                    _flag_row(**base, status="fail", reason=f"unreadable_file:{exc}")
                )
                continue

            reasons: list[str] = []
            if info["n_frames"] <= 0:
                reasons.append("unreadable_file")
            try:
                retained = compute_retained_frame_indices(
                    info["n_frames"],
                    info["fps"],
                    cfg.acquisition.discard_head_s,
                    cfg.acquisition.discard_tail_s,
                )
                split_non_overlapping_blocks(retained, cfg.acquisition.num_time_blocks)
            except ValueError:
                reasons.append("insufficient_frames")

            ew = cfg.acquisition.expected_frame_width_px
            eh = cfg.acquisition.expected_frame_height_px
            if ew is not None and eh is not None:
                if info["width"] != ew or info["height"] != eh:
                    reasons.append("incorrect_resolution")

            if row["channel"] == "green" and (
                not row["round_id"] or not row["challenge_id"]
            ):
                reasons.append("missing_challenge_or_round")

            status = "fail" if reasons else "ok"
            if status == "fail" and any(
                r in reasons
                for r in (
                    "insufficient_frames",
                    "incorrect_resolution",
                    "unreadable_file",
                    "missing_challenge_or_round",
                )
            ):
                errors.append(f"QC fail {vp.name}: {','.join(reasons)}")
            qc_rows.append(
                _flag_row(
                    **base,
                    status=status,
                    reason="|".join(reasons) if reasons else "ok",
                    n_frames=info["n_frames"],
                    fps=info["fps"],
                    width=info["width"],
                    height=info["height"],
                )
            )

    if miss_dup["missing"]:
        warnings.append(f"{len(miss_dup['missing'])} expected recordings missing")
    if miss_dup["duplicates"]:
        errors.append(f"Duplicated recordings: {miss_dup['duplicates']}")

    valid = len(errors) == 0
    return ValidationResult(
        valid=valid,
        metadata=df,
        inventory=inv,
        missing=miss_dup["missing"],
        duplicates=miss_dup["duplicates"],
        extras=miss_dup["extras"],
        qc_rows=qc_rows,
        errors=errors,
        warnings=warnings,
        is_pilot=is_pilot,
        development_devices=sorted(dev_cfg),
        frozen_test_devices_present=sorted(frozen_present),
    )


def apply_runtime_qc(
    *,
    saturation_fraction: float,
    mean_intensity: float,
    variance: float,
    cfg: Experiment3Config,
) -> list[str]:
    """QC flags that require decoded frames. Never score-based."""
    reasons: list[str] = []
    if saturation_fraction > cfg.analysis.max_saturation_fraction:
        reasons.append("severe_saturation")
    if mean_intensity < cfg.analysis.min_mean_intensity:
        reasons.append("nearly_absent_signal")
    if variance < cfg.analysis.min_image_variance:
        reasons.append("nearly_absent_signal")
    return reasons
