"""Video loading, dark correction, and three-block median templates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from experiment3.config import Experiment3Config
from puf_common.channels import apply_roi, dark_subtract, split_green, split_red
from puf_common.features import median_template

ProgressCallback = Callable[[str], None]


@dataclass
class ProcessedRecording:
    device_id: str
    state_id: str
    channel: str
    round_id: str
    challenge_id: str
    video_path: Path
    fps: float
    n_frames: int
    width: int
    height: int
    retained_indices: list[int]
    block_indices: list[list[int]]
    blocks: list[np.ndarray]
    representative: np.ndarray
    saturation_fraction: float
    mean_intensity: float
    variance: float
    qc_flags: list[str]


def probe_video(path: Path) -> dict[str, float | int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if n_frames <= 0 or width <= 0 or height <= 0:
        # decode one frame as fallback
        ok, frame = cap.read()
        if ok and frame is not None:
            height, width = frame.shape[:2]
            n_frames = max(n_frames, 1)
        else:
            cap.release()
            raise RuntimeError(f"Unreadable video: {path}")
    if fps <= 0:
        fps = 30.0
    cap.release()
    return {"fps": fps, "n_frames": n_frames, "width": width, "height": height}


def load_video_frames(
    path: Path,
    *,
    on_progress: ProgressCallback | None = None,
    progress_every: int = 48,
) -> tuple[list[np.ndarray], float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.ndim != 3 or frame.shape[2] < 3:
            cap.release()
            raise ValueError(
                f"Expected BGR color frames, got shape {getattr(frame, 'shape', None)} "
                f"in {path}. Grayscale conversion is forbidden."
            )
        frames.append(frame)
        if on_progress is not None and (
            len(frames) == 1
            or len(frames) % max(progress_every, 1) == 0
            or (n_frames > 0 and len(frames) >= n_frames)
        ):
            total = n_frames if n_frames > 0 else "?"
            on_progress(f"decode {len(frames)}/{total}")
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {path}")
    if n_frames <= 0:
        n_frames = len(frames)
    if fps <= 0:
        fps = 30.0
    return frames, fps, n_frames


def extract_channel(frame: np.ndarray, channel: str) -> np.ndarray:
    if channel == "red":
        return split_red(frame)
    if channel == "green":
        return split_green(frame)
    raise ValueError(f"Unsupported channel: {channel}")


def compute_retained_frame_indices(
    n_frames: int,
    fps: float,
    discard_head_s: float,
    discard_tail_s: float,
) -> list[int]:
    start = int(round(discard_head_s * fps))
    end = n_frames - int(round(discard_tail_s * fps))
    if end <= start:
        raise ValueError(
            f"Insufficient frames after trim: n_frames={n_frames}, fps={fps}, "
            f"discard_head_s={discard_head_s}, discard_tail_s={discard_tail_s}"
        )
    return list(range(start, end))


def split_non_overlapping_blocks(
    indices: list[int],
    num_blocks: int,
) -> list[list[int]]:
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    n = len(indices)
    if n < num_blocks:
        raise ValueError(
            f"Not enough retained frames ({n}) for {num_blocks} non-overlapping blocks"
        )
    base = n // num_blocks
    blocks: list[list[int]] = []
    offset = 0
    for b in range(num_blocks):
        size = base + (1 if b < n % num_blocks else 0)
        if size <= 0:
            raise ValueError("Block construction produced an empty block")
        blocks.append(indices[offset : offset + size])
        offset += size
    return blocks


def aggregate_frames_median(frames: list[np.ndarray]) -> np.ndarray:
    if not frames:
        raise ValueError("Cannot aggregate empty frame list")
    stack = np.stack([f.astype(np.float64, copy=False) for f in frames], axis=0)
    return np.median(stack, axis=0).astype(np.float32, copy=False)


def process_video(
    *,
    video_path: Path,
    channel: str,
    dark: np.ndarray | None,
    cfg: Experiment3Config,
    device_id: str,
    state_id: str,
    round_id: str = "",
    challenge_id: str = "",
    on_progress: ProgressCallback | None = None,
) -> ProcessedRecording:
    def _progress(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    _progress("opening")
    frames, fps, n_frames = load_video_frames(video_path, on_progress=on_progress)
    h, w = frames[0].shape[:2]
    retained = compute_retained_frame_indices(
        len(frames),
        fps,
        cfg.acquisition.discard_head_s,
        cfg.acquisition.discard_tail_s,
    )
    block_indices = split_non_overlapping_blocks(
        retained,
        cfg.acquisition.num_time_blocks,
    )

    corrected_frames: list[np.ndarray] = []
    sat_count = 0
    pix_count = 0
    n_retained = len(retained)
    roi = cfg.camera.roi
    for i, idx in enumerate(retained, start=1):
        if on_progress is not None and (i == 1 or i == n_retained or i % 32 == 0):
            _progress(f"correct {i}/{n_retained}")
        raw = extract_channel(frames[idx], channel)
        raw = apply_roi(raw, roi)
        if dark is not None:
            if dark.shape != raw.shape:
                raise ValueError(
                    f"Dark shape {dark.shape} != frame shape {raw.shape}; "
                    "dark templates must not be resized."
                )
            corr = dark_subtract(raw, dark)
        else:
            corr = raw.astype(np.float64)
        # Keep per-frame corrections in float32; aggregation still uses float64 locally.
        corrected_frames.append(np.asarray(corr, dtype=np.float32))
        sat_count += int(np.count_nonzero(raw >= cfg.analysis.saturation_threshold))
        pix_count += raw.size

    # Drop full BGR decode buffer before median aggregation.
    del frames

    _progress("aggregate")
    index_map = {frame_idx: pos for pos, frame_idx in enumerate(retained)}
    block_templates = [
        aggregate_frames_median([corrected_frames[index_map[i]] for i in block])
        for block in block_indices
    ]
    representative = np.asarray(median_template(corrected_frames), dtype=np.float32)
    del corrected_frames
    sat_frac = float(sat_count / max(pix_count, 1))
    mean_i = float(np.mean(representative))
    var_i = float(np.var(representative))
    qc_flags: list[str] = []
    if sat_frac > cfg.analysis.max_saturation_fraction:
        qc_flags.append("severe_saturation")
    if mean_i < cfg.analysis.min_mean_intensity or var_i < cfg.analysis.min_image_variance:
        qc_flags.append("nearly_absent_signal")

    return ProcessedRecording(
        device_id=device_id,
        state_id=state_id,
        channel=channel,
        round_id=round_id,
        challenge_id=challenge_id,
        video_path=Path(video_path),
        fps=fps,
        n_frames=n_frames,
        width=w,
        height=h,
        retained_indices=retained,
        block_indices=block_indices,
        blocks=block_templates,
        representative=representative,
        saturation_fraction=sat_frac,
        mean_intensity=mean_i,
        variance=var_i,
        qc_flags=qc_flags,
    )


def load_dark_channel(
    cfg: Experiment3Config,
    channel: str,
    reference_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray | None, str]:
    """Load dark template for a channel. Returns (dark_or_None, provenance_hash)."""
    art = cfg.dark_artifact_path
    if art is not None and art.exists():
        data = np.load(art)
        key = "dark_red" if channel == "red" else "dark_green"
        if key not in data:
            raise KeyError(f"{key} missing from dark artifact {art}")
        dark = np.asarray(data[key], dtype=np.float64)
        if reference_shape is not None and dark.shape != reference_shape:
            raise ValueError(
                f"Dark artifact shape {dark.shape} != reference {reference_shape}"
            )
        return dark, f"artifact:{art.name}"

    vid = cfg.dark_video_path
    if vid is not None and vid.exists():
        from puf_common.dark import build_dark_bgr_mean_from_video

        dark_r, dark_g, _n = build_dark_bgr_mean_from_video(vid, roi=cfg.camera.roi)
        dark = dark_r if channel == "red" else dark_g
        if reference_shape is not None and dark.shape != reference_shape:
            raise ValueError(
                f"Dark video shape {dark.shape} != reference {reference_shape}"
            )
        return dark, f"video:{vid.name}"

    if cfg.dark_frame.allow_missing_dark:
        return None, "none"
    raise FileNotFoundError("No valid dark video/artifact and allow_missing_dark=false")
