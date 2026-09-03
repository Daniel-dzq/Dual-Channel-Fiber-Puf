"""Dark-template construction and reuse (Experiment 1 numerical behavior)."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from puf_common.channels import apply_roi, split_green, split_red

logger = logging.getLogger(__name__)

DARK_PROCESSING_VERSION = "exp1_mean_v1"


@dataclass
class DarkProvenance:
    dark_source_experiment: str
    dark_source_path: str
    dark_artifact_hash: str
    dark_processing_version: str
    dark_channel: str
    dark_parameter_match: bool


@dataclass
class CameraSettings:
    frame_width_px: int | None = None
    frame_height_px: int | None = None
    bit_depth: int | None = None
    color_format: str | None = None
    exposure_ms: float | None = None
    gain: float | None = None
    frame_rate_fps: float | None = None
    camera_id: str | None = None
    roi: list[int] | None = None
    channel_convention: str = "opencv_bgr"


def _sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_dark_bgr_mean_from_video(
    path: Path,
    roi: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build mean dark red/green maps from a BGR video (Experiment 1 mean rule).

    Returns (dark_red, dark_green, n_frames).
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open dark video: {path}")
    acc_r = None
    acc_g = None
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = apply_roi(split_red(frame).astype(np.float64), roi)
        g = apply_roi(split_green(frame).astype(np.float64), roi)
        acc_r = r if acc_r is None else acc_r + r
        acc_g = g if acc_g is None else acc_g + g
        n += 1
    cap.release()
    if acc_r is None or n == 0:
        raise RuntimeError(f"No frames in dark video: {path}")
    return acc_r / n, acc_g / n, n


def load_or_build_experiment1_dark(
    *,
    artifact_path: Path | None,
    video_path: Path | None,
    cache_path: Path | None,
    roi: list[int] | None,
    allow_missing: bool,
) -> tuple[np.ndarray, np.ndarray, DarkProvenance]:
    """Load a 2048×1536 dark template (mean aggregation).

    Does not alter spatial shape. Does not recompute per Experiment 2 recording.
    """
    if artifact_path is not None and Path(artifact_path).exists():
        path = Path(artifact_path)
        data = np.load(path, allow_pickle=False)
        if isinstance(data, np.lib.npyio.NpzFile) or str(path).endswith(".npz"):
            z = np.load(path)
            if "dark_red" in z.files and "dark_green" in z.files:
                dark_r = np.asarray(z["dark_red"], dtype=np.float64)
                dark_g = np.asarray(z["dark_green"], dtype=np.float64)
            else:
                raise ValueError(
                    f"Dark NPZ must contain dark_red and dark_green arrays: {path}"
                )
            digest = _sha256_file(path)
        else:
            arr = np.asarray(data, dtype=np.float64)
            if arr.ndim == 3 and arr.shape[2] >= 3:
                # Assume BGR-ordered planes in last axis? Prefer explicit channels.
                # If saved as HxWx3 BGR:
                dark_r = arr[:, :, 2]
                dark_g = arr[:, :, 1]
            elif arr.ndim == 2:
                raise ValueError(
                    "Single-channel dark artifact cannot serve both red and green. "
                    "Provide an NPZ with dark_red and dark_green."
                )
            else:
                raise ValueError(f"Unsupported dark artifact shape {arr.shape}: {path}")
            digest = _sha256_array(arr)
        prov = DarkProvenance(
            dark_source_experiment="experiment1",
            dark_source_path=str(path.resolve()),
            dark_artifact_hash=digest,
            dark_processing_version=DARK_PROCESSING_VERSION,
            dark_channel="red_and_green",
            dark_parameter_match=True,
        )
        return dark_r, dark_g, prov

    if video_path is not None and Path(video_path).exists():
        path = Path(video_path)
        dark_r, dark_g, n = build_dark_bgr_mean_from_video(path, roi=roi)
        logger.info("Built Experiment 1 dark templates from video %s (%d frames)", path, n)
        if cache_path is not None:
            cache_path = Path(cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                dark_red=dark_r,
                dark_green=dark_g,
                source_video=str(path.resolve()),
                n_frames=n,
                processing_version=DARK_PROCESSING_VERSION,
            )
            digest = _sha256_file(cache_path)
            source_for_hash = str(cache_path.resolve())
        else:
            digest = _sha256_array(np.stack([dark_r, dark_g], axis=0))
            source_for_hash = str(path.resolve())
        prov = DarkProvenance(
            dark_source_experiment="experiment1",
            dark_source_path=source_for_hash,
            dark_artifact_hash=digest,
            dark_processing_version=DARK_PROCESSING_VERSION,
            dark_channel="red_and_green",
            dark_parameter_match=True,
        )
        return dark_r, dark_g, prov

    if allow_missing:
        raise RuntimeError("allow_missing_dark is not supported when reuse_experiment1 is required")
    raise FileNotFoundError(
        "Experiment 1 dark artifact/video not found. Set dark_frame.experiment1_dark_artifact "
        "or dark_frame.experiment1_dark_video in the Experiment 2 config."
    )


def assert_camera_settings_match(
    dark_settings: CameraSettings,
    record_settings: CameraSettings,
    *,
    strict: bool,
) -> list[str]:
    """Compare camera settings; raise if strict and critical fields differ."""
    mismatches: list[str] = []

    def _cmp(name: str, a, b, critical: bool = True) -> None:
        if a is None or b is None:
            return
        if isinstance(a, float) or isinstance(b, float):
            if abs(float(a) - float(b)) > 1e-6:
                mismatches.append(f"{name}: dark={a} record={b}")
        elif a != b:
            mismatches.append(f"{name}: dark={a} record={b}")

    _cmp("frame_width_px", dark_settings.frame_width_px, record_settings.frame_width_px)
    _cmp("frame_height_px", dark_settings.frame_height_px, record_settings.frame_height_px)
    _cmp("bit_depth", dark_settings.bit_depth, record_settings.bit_depth)
    _cmp("color_format", dark_settings.color_format, record_settings.color_format)
    _cmp("exposure_ms", dark_settings.exposure_ms, record_settings.exposure_ms)
    _cmp("gain", dark_settings.gain, record_settings.gain)
    _cmp("frame_rate_fps", dark_settings.frame_rate_fps, record_settings.frame_rate_fps)
    _cmp(
        "channel_convention",
        dark_settings.channel_convention,
        record_settings.channel_convention,
    )
    if dark_settings.roi is not None and record_settings.roi is not None:
        if list(dark_settings.roi) != list(record_settings.roi):
            mismatches.append(f"roi: dark={dark_settings.roi} record={record_settings.roi}")

    if mismatches and strict:
        raise ValueError(
            "Dark-parameter mismatch with Experiment 2 record settings: "
            + "; ".join(mismatches)
        )
    return mismatches


def provenance_to_dict(p: DarkProvenance) -> dict:
    return asdict(p)
