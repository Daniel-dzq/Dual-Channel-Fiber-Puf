"""Video I/O, dark correction, trim, and temporal-block median templates.

Reuses puf_common.channels / features. Trim/block logic mirrors Experiment 3.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from experiment00.config import Experiment00Config
from puf_common.channels import dark_subtract, saturation_fraction, split_green, split_red
from puf_common.features import median_template


@dataclass
class ProcessedVideo:
    path: Path
    length_cm: int
    fiber_id: str
    illumination: str
    round: str | None
    challenge: str | None
    red_phase: str | None
    fps: float
    n_frames: int
    width: int
    height: int
    retained_indices: list[int]
    block_indices: list[list[int]]
    blocks_raw: list[np.ndarray]
    representative_raw: np.ndarray
    saturation_fraction: float
    mean_intensity: float
    qc_flags: list[str] = field(default_factory=list)
    challenge_id: str | None = None
    sequence_id: str | None = None
    acquisition_position: int | None = None
    acquisition_position_zero_based: int | None = None
    order_source: str | None = None


def probe_video(path: Path) -> dict[str, float | int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if (n_frames <= 0 or width <= 0 or height <= 0) and cap.read()[0]:
        ok, frame = True, cap.read()[1] if False else None
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
        if ok and frame is not None:
            height, width = frame.shape[:2]
            n_frames = max(n_frames, 1)
    cap.release()
    if fps <= 0:
        fps = 30.0
    if width <= 0 or height <= 0 or n_frames <= 0:
        raise RuntimeError(f"Unreadable video metadata: {path}")
    return {"fps": fps, "n_frames": n_frames, "width": width, "height": height}


def compute_retained_frame_indices(
    n_frames: int, fps: float, discard_head_s: float, discard_tail_s: float
) -> list[int]:
    start = int(round(discard_head_s * fps))
    end = n_frames - int(round(discard_tail_s * fps))
    if end <= start:
        raise ValueError(
            f"Insufficient frames after trim: n={n_frames}, fps={fps}, "
            f"head={discard_head_s}, tail={discard_tail_s}"
        )
    return list(range(start, end))


def split_non_overlapping_blocks(indices: list[int], num_blocks: int) -> list[list[int]]:
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    n = len(indices)
    if n < num_blocks:
        raise ValueError(f"Not enough retained frames ({n}) for {num_blocks} blocks")
    base = n // num_blocks
    blocks: list[list[int]] = []
    offset = 0
    for b in range(num_blocks):
        size = base + (1 if b < n % num_blocks else 0)
        if size <= 0:
            raise ValueError("Empty temporal block")
        blocks.append(indices[offset : offset + size])
        offset += size
    return blocks


def load_channel_frames(path: Path, channel: str) -> tuple[list[np.ndarray], float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.ndim != 3 or frame.shape[2] < 3:
            cap.release()
            raise ValueError(f"Expected BGR frames in {path}, got {getattr(frame, 'shape', None)}")
        ch = split_red(frame) if channel == "red" else split_green(frame)
        frames.append(ch.astype(np.float32, copy=False))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded: {path}")
    if fps <= 0:
        fps = 30.0
    return frames, fps, len(frames)


@dataclass
class DarkFrame:
    """Per-channel mean dark for one (length, fiber) pair."""

    green: np.ndarray
    red: np.ndarray
    source: str
    n_frames: int
    length_cm: int | None = None
    fiber_id: str | None = None


# Filename patterns under dark/ (recursive). Parent folder may use ASCII or fullwidth ().
_DARK_NAMED_RE = re.compile(
    r"^(?P<length>7|9|11|13|15)cm_(?P<fiber>F\d{2})_dark\.mp4$",
    re.IGNORECASE,
)
_DARK_FIBER_ONLY_RE = re.compile(r"^dark_(?P<fiber>F\d{2})\.mp4$", re.IGNORECASE)
_DARK_FOLDER_LEN_RE = re.compile(r"(?:dark)?[\(（]\s*(?P<length>7|9|11|13|15)\s*cm\s*[\)）]", re.IGNORECASE)


@dataclass
class DarkBank:
    """Map (length_cm, fiber_id) → DarkFrame. One dark per fiber per length."""

    frames: dict[tuple[int, str], DarkFrame] = field(default_factory=dict)
    fallback: DarkFrame | None = None  # single shared dark if no per-fiber map

    def get(self, length_cm: int, fiber_id: str) -> DarkFrame | None:
        key = (int(length_cm), str(fiber_id).upper())
        if key in self.frames:
            return self.frames[key]
        return self.fallback

    def require(self, length_cm: int, fiber_id: str) -> DarkFrame:
        d = self.get(length_cm, fiber_id)
        if d is None:
            raise KeyError(f"No dark for length={length_cm}cm fiber={fiber_id}")
        return d

    @property
    def green(self) -> np.ndarray:
        """Legacy single-dark attribute (first / fallback). Prefer get()."""
        if self.fallback is not None:
            return self.fallback.green
        if not self.frames:
            raise AttributeError("Empty DarkBank")
        return next(iter(self.frames.values())).green

    @property
    def red(self) -> np.ndarray:
        if self.fallback is not None:
            return self.fallback.red
        if not self.frames:
            raise AttributeError("Empty DarkBank")
        return next(iter(self.frames.values())).red


def _parse_dark_key(path: Path, dark_root: Path) -> tuple[int | None, str | None]:
    """Infer (length_cm, fiber_id) from dark filename and/or parent folder."""
    name = path.name
    m = _DARK_NAMED_RE.match(name)
    if m:
        return int(m.group("length")), m.group("fiber").upper()
    m2 = _DARK_FIBER_ONLY_RE.match(name)
    fiber = m2.group("fiber").upper() if m2 else None
    length = None
    # Parent folders: dark(9cm) / dark（11cm）
    try:
        root = dark_root.resolve()
    except OSError:
        root = dark_root
    for parent in [path.parent, *path.parents]:
        tag = _DARK_FOLDER_LEN_RE.search(parent.name)
        if tag:
            length = int(tag.group("length"))
            break
        try:
            if parent.resolve() == root:
                break
        except OSError:
            break
    return length, fiber


def discover_dark_paths(cfg: Experiment00Config) -> list[Path]:
    """Collect dark MP4 candidates (recursive under dark_dir + optional single dark_video)."""
    candidates: list[Path] = []
    if cfg.paths.dark_video:
        rp = cfg.resolve_path(cfg.paths.dark_video)
        if rp is not None and rp.is_file():
            candidates.append(rp)
    if cfg.paths.dark_artifact:
        rp = cfg.resolve_path(cfg.paths.dark_artifact)
        if rp is not None and rp.is_file() and rp.suffix.lower() == ".mp4":
            candidates.append(rp)
    dark_dir = cfg.dark_path()
    if dark_dir.exists():
        candidates.extend(sorted(dark_dir.rglob("*.mp4")))
        candidates.extend(sorted(dark_dir.rglob("*.MP4")))
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in candidates:
        if not p.is_file() or p.stat().st_size <= 0:
            continue
        if p.suffix.lower() != ".mp4":
            continue
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def load_dark_bank(cfg: Experiment00Config) -> tuple[DarkBank | None, dict]:
    """Load per-(length, fiber) darks. Falls back to a single shared dark if needed."""
    from puf_common.dark import build_dark_bgr_mean_from_video

    provenance: dict = {
        "used": False,
        "mode": None,
        "source": None,
        "resized": False,
        "n_mapped": 0,
        "mapping": {},
        "unparsed": [],
    }
    paths = discover_dark_paths(cfg)
    if not paths:
        if cfg.experiment.allow_missing_dark:
            provenance["warning"] = "No dark field; proceeding with allow_missing_dark=true"
            return None, provenance
        raise FileNotFoundError(
            "No dark-field video found. Place per-fiber darks under dark/ "
            "(e.g. dark/dark(9cm)/dark_F01.mp4 or dark/dark（7cm）/7cm_F01_dark.mp4), "
            "or set paths.dark_video, or allow_missing_dark: true (exploratory only)."
        )

    dark_root = cfg.dark_path()
    bank = DarkBank()
    unparsed: list[str] = []
    for path in paths:
        length, fiber = _parse_dark_key(path, dark_root)
        dark_r, dark_g, n = build_dark_bgr_mean_from_video(path, roi=None)
        frame = DarkFrame(
            green=dark_g,
            red=dark_r,
            source=str(path.resolve()),
            n_frames=int(n),
            length_cm=length,
            fiber_id=fiber,
        )
        if length is not None and fiber is not None:
            key = (int(length), fiber)
            if key in bank.frames:
                provenance.setdefault("duplicates", []).append(
                    {"key": f"{length}cm|{fiber}", "kept": bank.frames[key].source, "ignored": str(path)}
                )
                continue
            bank.frames[key] = frame
            provenance["mapping"][f"{length}cm|{fiber}"] = str(path.resolve())
        else:
            unparsed.append(str(path))
            # Single unlabeled dark becomes fallback only if no mapped frames yet
            if bank.fallback is None and not bank.frames:
                bank.fallback = frame

    provenance["unparsed"] = unparsed
    provenance["n_mapped"] = len(bank.frames)

    if bank.frames:
        provenance.update(
            {
                "used": True,
                "mode": "per_length_fiber",
                "source": "dark_bank",
                "n_frames_example": next(iter(bank.frames.values())).n_frames,
                "shape_green": list(next(iter(bank.frames.values())).green.shape),
                "shape_red": list(next(iter(bank.frames.values())).red.shape),
            }
        )
        return bank, provenance

    if bank.fallback is not None:
        provenance.update(
            {
                "used": True,
                "mode": "single_shared",
                "source": bank.fallback.source,
                "n_frames": bank.fallback.n_frames,
                "shape_green": list(bank.fallback.green.shape),
                "shape_red": list(bank.fallback.red.shape),
                "warning": "Using a single shared dark for all fibers (no per-fiber map).",
            }
        )
        return bank, provenance

    if cfg.experiment.allow_missing_dark:
        provenance["warning"] = "Dark files present but could not be mapped; exploratory continue"
        return None, provenance
    raise FileNotFoundError(
        "Dark MP4s found but none could be mapped to (length, fiber). "
        f"Unparsed: {unparsed[:5]}"
    )


def load_dark_mean(cfg: Experiment00Config) -> tuple[DarkBank | None, dict]:
    """Backward-compatible alias for load_dark_bank."""
    return load_dark_bank(cfg)


def resolve_dark_frame(
    dark: DarkBank | DarkFrame | None,
    *,
    length_cm: int,
    fiber_id: str,
) -> DarkFrame | None:
    if dark is None:
        return None
    if isinstance(dark, DarkBank):
        return dark.get(length_cm, fiber_id)
    if isinstance(dark, DarkFrame) or hasattr(dark, "green"):
        return dark  # type: ignore[return-value]
    return None


def process_video_file(
    *,
    path: Path,
    length_cm: int,
    fiber_id: str,
    illumination: str,
    round_id: str | None,
    challenge: str | None,
    red_phase: str | None,
    dark,
    cfg: Experiment00Config,
    sequence_id: str | None = None,
    acquisition_position: int | None = None,
    acquisition_position_zero_based: int | None = None,
    order_source: str | None = None,
    challenge_id: str | None = None,
) -> ProcessedVideo:
    from experiment00.acquisition import ORDER_SOURCE, resolve_acquisition_meta

    ch_id = challenge_id or challenge
    if sequence_id is None and illumination == "green" and ch_id:
        meta = resolve_acquisition_meta(
            fiber_id=fiber_id,
            challenge_id=ch_id,
            illumination=illumination,
            assignment=cfg.acquisition.sequence_assignment,
            sequences=cfg.acquisition.challenge_sequences,
        )
        sequence_id = meta.sequence_id
        acquisition_position = meta.acquisition_position
        acquisition_position_zero_based = meta.acquisition_position_zero_based
        order_source = meta.order_source or ORDER_SOURCE
        ch_id = meta.challenge_id or ch_id

    channel = "red" if illumination == "red" else "green"
    frames, fps, n_frames = load_channel_frames(path, channel)
    h, w = frames[0].shape[:2]
    qc: list[str] = []
    dark_ch = None
    if dark is not None and cfg.preprocessing.dark_correction:
        dark_frame = resolve_dark_frame(dark, length_cm=length_cm, fiber_id=fiber_id)
        if dark_frame is None:
            raise KeyError(
                f"No matched dark for {length_cm}cm {fiber_id} "
                f"(video={path.name}). Provide dark/{length_cm}cm dark_{fiber_id} "
                "or {length_cm}cm_{fiber_id}_dark.mp4."
            )
        dark_ch = dark_frame.red if channel == "red" else dark_frame.green
        qc.append(f"dark_src={Path(dark_frame.source).name}")
        if dark_ch.shape != (h, w):
            # Dark from prior experiments may differ in resolution; resize explicitly.
            src_shape = tuple(int(x) for x in dark_ch.shape[:2])
            dark_ch = cv2.resize(dark_ch.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
            qc.append(f"dark_resized_from_{src_shape[0]}x{src_shape[1]}")
    try:
        retained = compute_retained_frame_indices(
            n_frames, fps, cfg.video.trim_start_s, cfg.video.trim_end_s
        )
    except ValueError as exc:
        raise ValueError(f"Invalid trim for {path.name}: {exc}") from exc
    block_indices = split_non_overlapping_blocks(retained, cfg.video.temporal_blocks)
    for bi, bidx in enumerate(block_indices):
        if len(bidx) < cfg.video.minimum_valid_frames_per_block:
            raise ValueError(
                f"Block {bi} of {path.name} has only {len(bidx)} frames "
                f"(min {cfg.video.minimum_valid_frames_per_block})"
            )

    corrected: list[np.ndarray] = []
    sat_vals: list[float] = []
    for fr in frames:
        x = fr.astype(np.float64)
        if dark_ch is not None:
            x = dark_subtract(x, dark_ch)
        corrected.append(x.astype(np.float32, copy=False))
        sat_vals.append(float(saturation_fraction(fr)))

    blocks_raw = [
        median_template([corrected[i] for i in idxs]).astype(np.float32, copy=False)
        for idxs in block_indices
    ]
    representative = median_template([corrected[i] for i in retained]).astype(np.float32, copy=False)
    sat = float(np.mean(sat_vals)) if sat_vals else 0.0
    if sat > cfg.quality_control.saturation_fraction_threshold:
        qc.append("high_saturation")
    mean_i = float(np.mean(representative))
    return ProcessedVideo(
        path=path,
        length_cm=length_cm,
        fiber_id=fiber_id,
        illumination=illumination,
        round=round_id,
        challenge=ch_id if illumination == "green" else None,
        challenge_id=ch_id if illumination == "green" else None,
        red_phase=red_phase,
        fps=fps,
        n_frames=n_frames,
        width=w,
        height=h,
        retained_indices=retained,
        block_indices=block_indices,
        blocks_raw=blocks_raw,
        representative_raw=representative,
        saturation_fraction=sat,
        mean_intensity=mean_i,
        qc_flags=qc,
        sequence_id=sequence_id if illumination == "green" else None,
        acquisition_position=acquisition_position if illumination == "green" else None,
        acquisition_position_zero_based=(
            acquisition_position_zero_based if illumination == "green" else None
        ),
        order_source=order_source if illumination == "green" else None,
    )
