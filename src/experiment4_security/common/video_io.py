"""BGR channel video loading and three-block median templates (no grayscale)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from puf_common.channels import dark_subtract, split_green, split_red
from puf_common.features import median_template


@dataclass
class ProcessedRecording:
    device_id: str
    state_id: str
    channel: str
    round_id: str
    challenge_id: str
    video_path: Path
    blocks: list[np.ndarray]
    representative: np.ndarray
    n_frames: int
    fps: float
    width: int
    height: int
    saturation_fraction: float
    mean_intensity: float
    qc_flags: list[str]


def extract_channel(frame: np.ndarray, channel: str) -> np.ndarray:
    if channel == "green":
        return split_green(frame)
    if channel == "red":
        return split_red(frame)
    raise ValueError(f"Unknown channel {channel}")


def process_video(
    *,
    video_path: Path,
    channel: str,
    dark: np.ndarray | None,
    device_id: str,
    state_id: str,
    round_id: str = "",
    challenge_id: str = "",
    discard_head_s: float = 10.0,
    discard_tail_s: float = 10.0,
    num_blocks: int = 3,
    sat_threshold: float = 250.0,
    max_sat_frac: float = 0.02,
) -> ProcessedRecording:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 10.0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame is None or frame.ndim != 3 or frame.shape[2] < 3:
            cap.release()
            raise ValueError(f"Expected BGR color frames in {video_path}")
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames in {video_path}")
    n = len(frames)
    h, w = frames[0].shape[:2]
    head = int(round(discard_head_s * fps))
    tail = int(round(discard_tail_s * fps))
    retained = list(range(head, max(head, n - tail)))
    if len(retained) < num_blocks:
        retained = list(range(n))
    base = len(retained) // num_blocks
    block_idx: list[list[int]] = []
    offset = 0
    for b in range(num_blocks):
        size = base + (1 if b < len(retained) % num_blocks else 0)
        block_idx.append(retained[offset : offset + size])
        offset += size

    corrected: list[np.ndarray] = []
    sat = 0
    pix = 0
    for idx in retained:
        raw = extract_channel(frames[idx], channel)
        corr = dark_subtract(raw, dark) if dark is not None else raw.astype(np.float64)
        corrected.append(np.asarray(corr, dtype=np.float32))
        sat += int(np.count_nonzero(raw >= sat_threshold))
        pix += raw.size
    del frames

    index_map = {frame_idx: pos for pos, frame_idx in enumerate(retained)}
    blocks = []
    for bi in block_idx:
        stack = np.stack([corrected[index_map[i]] for i in bi], axis=0)
        blocks.append(np.median(stack, axis=0).astype(np.float32))
    rep = np.asarray(median_template(corrected), dtype=np.float32)
    del corrected
    sat_frac = float(sat / max(pix, 1))
    flags: list[str] = []
    if sat_frac > max_sat_frac:
        flags.append("severe_saturation")
    mean_i = float(np.mean(rep))
    if mean_i < 1.0:
        flags.append("nearly_absent_signal")
    return ProcessedRecording(
        device_id=device_id,
        state_id=state_id,
        channel=channel,
        round_id=round_id,
        challenge_id=challenge_id,
        video_path=Path(video_path),
        blocks=blocks,
        representative=rep,
        n_frames=n,
        fps=fps,
        width=w,
        height=h,
        saturation_fraction=sat_frac,
        mean_intensity=mean_i,
        qc_flags=flags,
    )
