"""RGB/BGR color decode + wavelength-matched channel extraction (no grayscale).

Convention (aligned with experiment_02 / puf_common):
- OpenCV BGR frames
- filename color green (532 nm) → green plane (BGR index 1)
- filename color red   (650 nm) → red plane   (BGR index 2)

Grayscale / luma decoding is intentionally rejected for science paths.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from puf_common.channels import split_green, split_red

logger = logging.getLogger(__name__)

ColorLabel = Literal["green", "red"]


@dataclass
class VideoMeta:
    path: str
    width: int
    height: int
    codec: str
    pixel_format: str
    duration_s: float
    encoded_frame_count: int | None
    decode_backend: str
    color_label: str
    wavelength_nm: int
    analysis_channel: str  # "green" | "red"
    analysis_channel_bgr_index: int


@dataclass
class DecodedVideo:
    frames: np.ndarray  # (N,H,W) float32 selected color plane
    timestamps_s: np.ndarray  # (N,)
    meta: VideoMeta
    # Optional mid-video BGR sample for leakage audit (H,W,3) uint8/float
    bgr_sample: np.ndarray | None = None


def color_label_to_channel(color_label: str) -> tuple[str, int, int]:
    """Return (channel_name, bgr_index, wavelength_nm)."""
    c = str(color_label).lower().strip()
    if c == "green":
        return "green", 1, 532
    if c == "red":
        return "red", 2, 650
    raise ValueError(f"color_label must be green|red, got {color_label!r}")


def extract_analysis_channel(bgr: np.ndarray, color_label: str) -> np.ndarray:
    """Extract wavelength-matched plane from BGR using puf_common helpers."""
    c = str(color_label).lower().strip()
    if c == "green":
        return split_green(bgr).astype(np.float32)
    if c == "red":
        return split_red(bgr).astype(np.float32)
    raise ValueError(f"color_label must be green|red, got {color_label!r}")


def _finalize_timestamps(times: list[float], fps_fallback: float) -> np.ndarray:
    ts = np.asarray(times, dtype=np.float64)
    for i in range(1, len(ts)):
        if ts[i] <= ts[i - 1]:
            ts[i] = ts[i - 1] + (1.0 / max(fps_fallback, 1e-6))
    return ts


def _try_pyav_bgr(path: Path, color_label: str) -> DecodedVideo | None:
    try:
        import av  # type: ignore
    except Exception:
        return None

    channel_name, bgr_index, wavelength_nm = color_label_to_channel(color_label)
    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    fps_fb = float(stream.average_rate or 30.0)
    codec_name = str(getattr(stream.codec_context, "name", "unknown"))
    pix_fmt = str(getattr(stream.codec_context, "pix_fmt", "unknown"))
    planes: list[np.ndarray] = []
    times: list[float] = []
    bgr_sparse: list[tuple[int, np.ndarray]] = []
    for i, frame in enumerate(container.decode(video=0)):
        rgb = frame.to_ndarray(format="rgb24")
        bgr = rgb[:, :, ::-1].copy()
        if bgr.ndim != 3 or bgr.shape[2] < 3:
            container.close()
            raise ValueError(f"Expected 3-channel frame from {path}, got {bgr.shape}")
        planes.append(extract_analysis_channel(bgr, color_label))
        if frame.pts is not None and stream.time_base is not None:
            times.append(float(frame.pts * stream.time_base))
        else:
            times.append(float(i) / fps_fb)
        if i % 5 == 0:
            bgr_sparse.append((i, bgr.astype(np.float32)))
    container.close()
    n = len(planes)
    if n == 0:
        return None
    mid = n // 2
    sample_bgr = (
        min(bgr_sparse, key=lambda t: abs(t[0] - mid))[1] if bgr_sparse else None
    )
    ts = _finalize_timestamps(times, fps_fb)
    meta = VideoMeta(
        path=str(path),
        width=int(planes[0].shape[1]),
        height=int(planes[0].shape[0]),
        codec=codec_name,
        pixel_format=pix_fmt,
        duration_s=float(ts[-1] - ts[0]) if ts.size > 1 else 0.0,
        encoded_frame_count=n,
        decode_backend="pyav_bgr_channel",
        color_label=str(color_label).lower(),
        wavelength_nm=wavelength_nm,
        analysis_channel=channel_name,
        analysis_channel_bgr_index=bgr_index,
    )
    return DecodedVideo(
        frames=np.stack(planes, axis=0),
        timestamps_s=ts,
        meta=meta,
        bgr_sample=sample_bgr,
    )


def _opencv_bgr_channel(path: Path, color_label: str) -> DecodedVideo:
    import cv2

    channel_name, bgr_index, wavelength_nm = color_label_to_channel(color_label)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-6:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    planes: list[np.ndarray] = []
    times: list[float] = []
    bgr_all_for_sample: list[np.ndarray] = []
    idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if bgr is None or bgr.ndim != 3 or bgr.shape[2] < 3:
            cap.release()
            raise ValueError(f"Expected BGR color frame from {path}, got {getattr(bgr, 'shape', None)}")
        planes.append(extract_analysis_channel(bgr, color_label))
        msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        t = msec / 1000.0 if msec > 0 else idx / fps
        times.append(t)
        # keep sparse samples for mid-frame selection without storing all
        if idx % 5 == 0:
            bgr_all_for_sample.append((idx, bgr.copy()))
        idx += 1
    cap.release()
    if not planes:
        raise RuntimeError(f"No frames decoded from {path}")
    ts = _finalize_timestamps(times, fps)
    mid = len(planes) // 2
    sample_bgr = None
    if bgr_all_for_sample:
        # nearest stored sample to mid
        sample_bgr = min(bgr_all_for_sample, key=lambda t: abs(t[0] - mid))[1].astype(np.float32)
    meta = VideoMeta(
        path=str(path),
        width=width or int(planes[0].shape[1]),
        height=height or int(planes[0].shape[0]),
        codec="opencv_unknown",
        pixel_format="bgr24",
        duration_s=float(ts[-1] - ts[0]),
        encoded_frame_count=n_reported if n_reported > 0 else len(planes),
        decode_backend="opencv_bgr_channel",
        color_label=str(color_label).lower(),
        wavelength_nm=wavelength_nm,
        analysis_channel=channel_name,
        analysis_channel_bgr_index=bgr_index,
    )
    return DecodedVideo(
        frames=np.stack(planes, axis=0),
        timestamps_s=ts,
        meta=meta,
        bgr_sample=sample_bgr,
    )


def decode_channel_video(path: str | Path, color_label: str) -> DecodedVideo:
    """Decode color video and return wavelength-matched channel stack."""
    path = Path(path)
    color_label_to_channel(color_label)  # validate early
    out = _try_pyav_bgr(path, color_label)
    if out is not None:
        return out
    logger.info(
        "PyAV unavailable/failed for %s; OpenCV BGR → %s channel",
        path.name,
        color_label,
    )
    return _opencv_bgr_channel(path, color_label)


# Back-compat alias: refuse silent gray use
def decode_luma_video(path: str | Path, color_label: str | None = None) -> DecodedVideo:
    if color_label is None:
        raise TypeError(
            "Grayscale/luma decode is disabled. Pass color_label='green'|'red' "
            "and use decode_channel_video()."
        )
    return decode_channel_video(path, color_label)


def select_analysis_frames(
    timestamps_s: np.ndarray,
    *,
    analysis_start_s: float,
    analysis_end_margin_s: float,
    max_sampled_frames: int,
    duration_s: float | None = None,
) -> np.ndarray:
    """Return indices into the timestamp array (time-based window + uniform subsample)."""
    ts = np.asarray(timestamps_s, dtype=np.float64)
    if ts.size == 0:
        return np.asarray([], dtype=int)
    t0 = float(ts[0])
    t_last = float(ts[-1])
    win_start = t0 + float(analysis_start_s)
    win_end = t_last - float(analysis_end_margin_s)
    if win_end <= win_start:
        win_start = t0 + 0.25 * (t_last - t0)
        win_end = t0 + 0.75 * (t_last - t0)
    mask = (ts >= win_start) & (ts <= win_end)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        idx = np.arange(ts.size)
    if idx.size <= max_sampled_frames:
        return idx.astype(int)
    t_sel = ts[idx]
    targets = np.linspace(t_sel[0], t_sel[-1], int(max_sampled_frames))
    chosen = []
    used = set()
    for tgt in targets:
        j = int(np.argmin(np.abs(t_sel - tgt)))
        while j in used and j + 1 < len(idx):
            j += 1
        if j in used:
            continue
        used.add(j)
        chosen.append(int(idx[j]))
    return np.asarray(sorted(chosen), dtype=int)


def split_time_blocks(
    timestamps_s: np.ndarray,
    retained_indices: np.ndarray,
    n_blocks: int,
) -> list[np.ndarray]:
    if retained_indices.size == 0:
        return [np.asarray([], dtype=int) for _ in range(n_blocks)]
    ts = timestamps_s[retained_indices]
    t0, t1 = float(ts.min()), float(ts.max())
    if t1 <= t0 or n_blocks <= 1:
        return [retained_indices.astype(int)]
    edges = np.linspace(t0, t1, int(n_blocks) + 1)
    blocks = []
    for b in range(int(n_blocks)):
        lo, hi = edges[b], edges[b + 1]
        if b < n_blocks - 1:
            m = (ts >= lo) & (ts < hi)
        else:
            m = (ts >= lo) & (ts <= hi)
        blocks.append(retained_indices[np.flatnonzero(m)].astype(int))
    return blocks


def maximum_timestamp_gap(timestamps_s: np.ndarray) -> float:
    ts = np.asarray(timestamps_s, dtype=np.float64)
    if ts.size < 2:
        return 0.0
    return float(np.max(np.diff(ts)))


def meta_to_dict(meta: VideoMeta) -> dict[str, Any]:
    return {
        "source_path": meta.path,
        "width": meta.width,
        "height": meta.height,
        "codec": meta.codec,
        "pixel_format": meta.pixel_format,
        "duration_s": meta.duration_s,
        "encoded_frame_count": meta.encoded_frame_count,
        "decode_backend": meta.decode_backend,
        "color_label": meta.color_label,
        "wavelength_nm": meta.wavelength_nm,
        "analysis_channel": meta.analysis_channel,
        "analysis_channel_bgr_index": meta.analysis_channel_bgr_index,
    }
