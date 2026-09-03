"""Video loading, dark correction, and time-block aggregation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from experiment2.config import Experiment2Config
from puf_common.channels import apply_roi, dark_subtract, split_green, split_red
from puf_common.features import median_template

ProgressCallback = Callable[[str], None]


@dataclass
class ProcessedRecording:
    device_id: str
    record_type: str
    channel: str
    challenge_id: str
    session_id: str
    video_path: Path
    fps: float
    n_frames: int
    retained_indices: list[int]
    block_indices: list[list[int]]
    blocks: list[np.ndarray]
    representative: np.ndarray
    saturation_fraction: float
    mean_intensity: float


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


def assert_dark_shape_matches(dark: np.ndarray, frame: np.ndarray) -> None:
    if dark.shape != frame.shape:
        raise ValueError(
            f"Dark template shape {dark.shape} does not match frame shape "
            f"{frame.shape}."
        )


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
            f"Trim removes all frames: n_frames={n_frames}, "
            f"discard_head_s={discard_head_s}, discard_tail_s={discard_tail_s}, fps={fps}"
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
        blocks.append(indices[offset : offset + size])
        offset += size
    return blocks


def aggregate_frames_median(frames: list[np.ndarray]) -> np.ndarray:
    if not frames:
        raise ValueError("Cannot aggregate empty frame list")
    stack = np.stack([f.astype(np.float64) for f in frames], axis=0)
    return np.median(stack, axis=0)


def process_video(
    *,
    video_path: Path,
    channel: str,
    dark: np.ndarray,
    roi: list[int] | None,
    cfg: Experiment2Config,
    device_id: str,
    record_type: str,
    challenge_id: str,
    session_id: str,
    on_progress: ProgressCallback | None = None,
) -> ProcessedRecording:
    def _progress(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    _progress("opening")
    frames, fps, n_frames = load_video_frames(video_path, on_progress=on_progress)
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
    for i, idx in enumerate(retained, start=1):
        if on_progress is not None and (i == 1 or i == n_retained or i % 32 == 0):
            _progress(f"correct {i}/{n_retained}")
        raw = extract_channel(frames[idx], channel)
        raw = apply_roi(raw, roi)
        assert_dark_shape_matches(dark, raw)
        corr = dark_subtract(raw, dark)
        corrected_frames.append(corr)
        sat_count += int(np.count_nonzero(raw >= cfg.analysis.saturation_threshold))
        pix_count += raw.size

    _progress("aggregate")
    index_map = {frame_idx: pos for pos, frame_idx in enumerate(retained)}
    block_templates = [
        aggregate_frames_median(
            [corrected_frames[index_map[i]] for i in block]
        )
        for block in block_indices
    ]
    representative = median_template(corrected_frames)

    return ProcessedRecording(
        device_id=device_id,
        record_type=record_type,
        channel=channel,
        challenge_id=challenge_id,
        session_id=session_id,
        video_path=video_path,
        fps=fps,
        n_frames=n_frames,
        retained_indices=retained,
        block_indices=block_indices,
        blocks=block_templates,
        representative=representative,
        saturation_fraction=float(sat_count / max(pix_count, 1)),
        mean_intensity=float(np.mean(representative)),
    )
