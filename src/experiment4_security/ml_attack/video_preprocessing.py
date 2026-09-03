"""8-second independent-clip video preprocessing, QC, and full-resolution
detail-response extraction.

Hard rules enforced here (do not "fix" by adding trimming logic):

- The full 8 s clip's every decodable frame is used to build the official
  response template (temporal median). No head/tail discard, no
  half-only templates, no fixed-time cropping.
- First-half / second-half templates are QC-only diagnostics. They must
  never replace, or be blended into, the official full-video template.
- The valid_mask, envelope sigma/epsilon, dark-correction convention, and
  NCC definition are all reused unmodified from `puf_common` /
  `experiment4_security.lifecycle` (read-only dependency).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from experiment4_security.common.video_io import process_video
from experiment4_security.ml_attack.config import MLAttackConfig
from puf_common.channels import split_green
from puf_common.features import to_detail
from puf_common.masks import build_valid_mask_from_refs, load_mask_png, mask_hash, save_mask_png
from puf_common.ncc import zero_mean_ncc


# ---------------------------------------------------------------------------
# Frozen valid_mask: read-only reuse of the lifecycle recipe
# ---------------------------------------------------------------------------


def reconstruct_frozen_valid_mask(
    cfg: MLAttackConfig, *, cache_dir: Path
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load or, if the binary is missing on disk, deterministically rebuild
    the frozen lifecycle valid_mask.

    This function NEVER writes into `outputs/experiment4/lifecycle/`. It
    only reads:
      - `<lifecycle_run>/valid_mask.png` (used as-is if present), else
      - `<lifecycle_run>/validated_metadata.csv` + `resolved_config.yaml`
        (already-frozen, already-validated Experiment 3 video paths used by
        the lifecycle run) together with the exact same reference-selection
        filter and `puf_common.masks.build_valid_mask_from_refs(..., abs_floor=5.0,
        percentile=10.0)` call used by
        `experiment4_security.lifecycle.cli.run_lifecycle_analysis` (imported
        read-only, not reimplemented from scratch).

    The reconstructed mask is cached under this pilot's own output tree
    (never inside the lifecycle run directory).
    """
    lifecycle_run = Path(cfg.lifecycle_run)
    direct_mask_path = lifecycle_run / "valid_mask.png"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_mask_path = cache_dir / "frozen_valid_mask.png"
    provenance_path = cache_dir / "valid_mask_provenance.json"

    if direct_mask_path.exists():
        mask = load_mask_png(direct_mask_path)
        provenance = {
            "source": "lifecycle_run_direct_file",
            "lifecycle_run": str(lifecycle_run),
            "mask_path": str(direct_mask_path),
            "mask_hash": mask_hash(mask),
            "mask_shape": list(mask.shape),
            "mask_coverage_fraction": float(mask.mean()),
        }
        save_mask_png(mask, cached_mask_path)
        provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        return mask, provenance

    if cached_mask_path.exists() and provenance_path.exists():
        mask = load_mask_png(cached_mask_path)
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("source", "").startswith("reconstructed"):
            return mask, provenance

    validated_metadata_path = lifecycle_run / "validated_metadata.csv"
    if not validated_metadata_path.exists():
        raise FileNotFoundError(
            f"Cannot obtain frozen valid_mask: neither {direct_mask_path} nor "
            f"{validated_metadata_path} exist. The lifecycle run directory "
            "must contain either the mask PNG directly, or the validated "
            "metadata needed to deterministically rebuild it read-only."
        )
    meta = pd.read_csv(validated_metadata_path)
    meta["challenge_id"] = meta["challenge_id"].fillna("").astype(str)
    meta["round_id"] = meta["round_id"].fillna("").astype(str)

    ref_meta = meta[
        ((meta["channel"] == "red") & (meta["state_id"] == "S0"))
        | (
            (meta["channel"] == "green")
            & (meta["state_id"] == "S0")
            & (meta["round_id"] == "A")
            & (meta["challenge_id"].isin(["C01", "C001"]))
        )
    ]
    missing_videos = [p for p in ref_meta["video_path"].tolist() if not Path(p).exists()]
    if missing_videos:
        raise FileNotFoundError(
            "Cannot reconstruct frozen valid_mask: reference videos listed in "
            f"the lifecycle run's validated_metadata.csv are missing on disk: "
            f"{missing_videos[:5]} (and {max(0, len(missing_videos) - 5)} more)."
        )

    refs: list[np.ndarray] = []
    for row in ref_meta.itertuples(index=False):
        rec = process_video(
            video_path=Path(row.video_path),
            channel=row.channel,
            dark=None,
            device_id=row.device_id,
            state_id=row.state_id,
            round_id=getattr(row, "round_id", "") or "",
            challenge_id=getattr(row, "challenge_id", "") or "",
        )
        refs.append(rec.blocks[0])

    mask, coverage, threshold = build_valid_mask_from_refs(refs, abs_floor=5.0, percentile=10.0)
    save_mask_png(mask, cached_mask_path)
    provenance = {
        "source": "reconstructed_from_lifecycle_recipe",
        "lifecycle_run": str(lifecycle_run),
        "reference_video_count": len(refs),
        "reference_selection_filter": (
            "(channel=='red' & state_id=='S0') | "
            "(channel=='green' & state_id=='S0' & round_id=='A' & challenge_id in ['C01','C001'])"
        ),
        "build_valid_mask_from_refs_params": {"abs_floor": 5.0, "percentile": 10.0},
        "mask_hash": mask_hash(mask),
        "mask_shape": list(mask.shape),
        "mask_coverage_fraction": float(coverage),
        "mask_threshold": float(threshold),
        "note": (
            "valid_mask.png was absent from the frozen lifecycle run directory "
            "on disk in this workspace; this mask is a deterministic, "
            "byte-verifiable-recipe reconstruction using the same read-only "
            "lifecycle code path (puf_common.masks.build_valid_mask_from_refs) "
            "and the same already-frozen reference video set recorded in "
            "validated_metadata.csv. No lifecycle output file was written to."
        ),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return mask, provenance


# ---------------------------------------------------------------------------
# Per-clip processing
# ---------------------------------------------------------------------------


@dataclass
class ClipResult:
    sample_id: str
    decode_status: str
    duration_s: float | None
    fps: float | None
    frame_count_total: int | None
    frame_count_decoded: int
    frame_count_used: int
    mean_intensity: float | None
    std_intensity: float | None
    saturation_fraction: float | None
    centroid_x: float | None
    centroid_y: float | None
    first_half_vs_second_half_ncc: float | None
    mean_intensity_first_half: float | None
    mean_intensity_second_half: float | None
    relative_intensity_drift: float | None
    centroid_first_half_x: float | None
    centroid_first_half_y: float | None
    centroid_second_half_x: float | None
    centroid_second_half_y: float | None
    centroid_drift_px: float | None
    qc_status: str
    qc_reasons: list[str]


def _centroid(image: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    weights = np.clip(image, 0.0, None)
    weights = np.where(mask, weights, 0.0)
    total = float(weights.sum())
    if total <= 0.0:
        return float("nan"), float("nan")
    ys, xs = np.indices(image.shape)
    cx = float((weights * xs).sum() / total)
    cy = float((weights * ys).sum() / total)
    return cx, cy


def _decode_all_green_frames(video_path: Path) -> tuple[list[np.ndarray], float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], 0.0, 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame is None or frame.ndim != 3 or frame.shape[2] < 3:
            break
        # float32 (not float64): halves peak RAM for the ~50-frame stack per
        # clip, which matters once clips are processed in parallel worker
        # processes. Harmless numerically -- `puf_common.envelope.gaussian_blur`
        # (used by `to_detail` below) always upcasts to float64 internally
        # regardless of the input dtype, so this has zero effect on the
        # detail-representation math.
        frames.append(split_green(frame).astype(np.float32))
    reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return frames, fps, reported


def process_clip(
    video_path: Path,
    *,
    sample_id: str,
    valid_mask: np.ndarray,
    cfg: MLAttackConfig,
) -> tuple[np.ndarray | None, np.ndarray | None, ClipResult]:
    """Process one independent 8 s clip.

    Returns (raw_full_res_template, full_res_detail, qc) where the first two
    are `None` if decoding failed. `dark_mode` is read from
    `cfg.preprocessing.dark_mode`; only `"none"` is currently supported
    (matches the current lifecycle convention of `dark=None`).
    """
    if cfg.preprocessing.dark_mode != "none":
        raise NotImplementedError(
            f"dark_mode={cfg.preprocessing.dark_mode!r} not supported; only "
            "'none' is implemented, matching the current lifecycle convention."
        )

    frames, fps, reported = _decode_all_green_frames(video_path)
    n_decoded = len(frames)
    qc_reasons: list[str] = []
    th = cfg.qc_thresholds

    if n_decoded == 0:
        qc = ClipResult(
            sample_id=sample_id,
            decode_status="ERROR_CANNOT_DECODE",
            duration_s=None,
            fps=fps if fps > 0 else None,
            frame_count_total=reported or None,
            frame_count_decoded=0,
            frame_count_used=0,
            mean_intensity=None,
            std_intensity=None,
            saturation_fraction=None,
            centroid_x=None,
            centroid_y=None,
            first_half_vs_second_half_ncc=None,
            mean_intensity_first_half=None,
            mean_intensity_second_half=None,
            relative_intensity_drift=None,
            centroid_first_half_x=None,
            centroid_first_half_y=None,
            centroid_second_half_x=None,
            centroid_second_half_y=None,
            centroid_drift_px=None,
            qc_status="ERROR",
            qc_reasons=["cannot_decode_any_frame"],
        )
        return None, None, qc

    duration_s = n_decoded / fps if fps > 0 else None
    stack = np.stack(frames, axis=0)
    n_half = n_decoded // 2
    first_half = stack[:n_half] if n_half > 0 else stack
    second_half = stack[n_half:] if n_half > 0 else stack

    raw_template = np.median(stack, axis=0).astype(np.float32)
    first_template = np.median(first_half, axis=0).astype(np.float32)
    second_template = np.median(second_half, axis=0).astype(np.float32)

    if valid_mask.shape != raw_template.shape:
        mask_resized = cv2.resize(
            valid_mask.astype(np.uint8), (raw_template.shape[1], raw_template.shape[0]), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    else:
        mask_resized = valid_mask

    mean_intensity = float(raw_template[mask_resized].mean())
    std_intensity = float(raw_template[mask_resized].std())
    mean_i1 = float(first_template[mask_resized].mean())
    mean_i2 = float(second_template[mask_resized].mean())
    rel_drift = float((mean_i2 - mean_i1) / mean_i1) if mean_i1 != 0 else float("nan")

    sat_threshold = 250.0
    sat_count = int(np.count_nonzero(stack[:, mask_resized] >= sat_threshold))
    sat_total = int(stack[:, mask_resized].size)
    saturation_fraction = sat_count / max(sat_total, 1)

    cx, cy = _centroid(raw_template, mask_resized)
    cx1, cy1 = _centroid(first_template, mask_resized)
    cx2, cy2 = _centroid(second_template, mask_resized)
    centroid_drift_px = float(np.hypot(cx2 - cx1, cy2 - cy1)) if np.isfinite(cx1) and np.isfinite(cx2) else float("nan")

    half_ncc = float(zero_mean_ncc(first_template, second_template, mask=mask_resized))

    # --- QC decisions (all thresholds sourced from config) ---
    qc_status = "PASS"

    def _escalate(new_status: str, reason: str) -> None:
        nonlocal qc_status
        qc_reasons.append(reason)
        order = {"PASS": 0, "WARN": 1, "ERROR": 2}
        if order[new_status] > order[qc_status]:
            qc_status = new_status

    if duration_s is not None and duration_s < th.duration_error_below_s:
        _escalate("ERROR", f"duration_s={duration_s:.3f} < error_threshold={th.duration_error_below_s}")
    elif duration_s is not None and duration_s < th.duration_warn_below_s:
        _escalate("WARN", f"duration_s={duration_s:.3f} < warn_threshold={th.duration_warn_below_s}")
    if duration_s is not None and duration_s > th.duration_warn_above_s:
        _escalate("WARN", f"duration_s={duration_s:.3f} > warn_threshold_above={th.duration_warn_above_s}")

    if n_decoded < th.min_frames_used_error_below:
        _escalate("ERROR", f"frame_count_used={n_decoded} < error_threshold={th.min_frames_used_error_below}")

    if saturation_fraction >= th.saturation_error_fraction:
        _escalate("ERROR", f"saturation_fraction={saturation_fraction:.4f} >= error_threshold={th.saturation_error_fraction}")
    elif saturation_fraction >= th.saturation_warn_fraction:
        _escalate("WARN", f"saturation_fraction={saturation_fraction:.4f} >= warn_threshold={th.saturation_warn_fraction}")

    if half_ncc < th.half_split_ncc_error_below:
        _escalate("ERROR", f"first_half_vs_second_half_ncc={half_ncc:.4f} < error_threshold={th.half_split_ncc_error_below}")
    elif half_ncc < th.half_split_ncc_warn_below:
        _escalate("WARN", f"first_half_vs_second_half_ncc={half_ncc:.4f} < warn_threshold={th.half_split_ncc_warn_below}")

    if np.isfinite(rel_drift) and abs(rel_drift) >= th.relative_intensity_drift_error:
        _escalate("ERROR", f"relative_intensity_drift={rel_drift:.4f} beyond error_threshold={th.relative_intensity_drift_error}")
    elif np.isfinite(rel_drift) and abs(rel_drift) >= th.relative_intensity_drift_warn:
        _escalate("WARN", f"relative_intensity_drift={rel_drift:.4f} beyond warn_threshold={th.relative_intensity_drift_warn}")

    if np.isfinite(centroid_drift_px) and centroid_drift_px >= th.centroid_drift_error_px:
        _escalate("ERROR", f"centroid_drift_px={centroid_drift_px:.3f} >= error_threshold={th.centroid_drift_error_px}")
    elif np.isfinite(centroid_drift_px) and centroid_drift_px >= th.centroid_drift_warn_px:
        _escalate("WARN", f"centroid_drift_px={centroid_drift_px:.3f} >= warn_threshold={th.centroid_drift_warn_px}")

    if not qc_reasons:
        qc_reasons.append("ok")

    detail = to_detail(raw_template, sigma=cfg.preprocessing.envelope_sigma, eps=cfg.preprocessing.envelope_epsilon)
    detail = detail.astype(np.float32)

    qc = ClipResult(
        sample_id=sample_id,
        decode_status="OK",
        duration_s=duration_s,
        fps=fps,
        frame_count_total=reported or n_decoded,
        frame_count_decoded=n_decoded,
        frame_count_used=n_decoded,
        mean_intensity=mean_intensity,
        std_intensity=std_intensity,
        saturation_fraction=saturation_fraction,
        centroid_x=cx,
        centroid_y=cy,
        first_half_vs_second_half_ncc=half_ncc,
        mean_intensity_first_half=mean_i1,
        mean_intensity_second_half=mean_i2,
        relative_intensity_drift=rel_drift,
        centroid_first_half_x=cx1,
        centroid_first_half_y=cy1,
        centroid_second_half_x=cx2,
        centroid_second_half_y=cy2,
        centroid_drift_px=centroid_drift_px,
        qc_status=qc_status,
        qc_reasons=qc_reasons,
    )
    return raw_template, detail, qc


def clip_result_to_row(qc: ClipResult) -> dict[str, Any]:
    row = asdict(qc)
    row["qc_reasons"] = "|".join(qc.qc_reasons)
    return row


def mask_vector(full_res_array: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Flatten a full-resolution 2D array to its valid_mask-selected pixels.

    This is the "full-resolution response representation" used for modeling
    and for the final NCC evaluation: it uses every valid pixel at native
    resolution, it is simply not padded with pixels the project's own
    valid_mask convention already excludes as non-signal-bearing (the same
    convention `green_credential.green_score` already applies at NCC time).
    """
    if full_res_array.shape != valid_mask.shape:
        raise ValueError(f"shape mismatch {full_res_array.shape} vs mask {valid_mask.shape}")
    return full_res_array[valid_mask].astype(np.float32)


def array_sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Disk-backed vector cache (avoids holding all ~2000 full-res vectors in RAM)
# ---------------------------------------------------------------------------


def vector_cache_path(cache_dir: Path, sample_id: str) -> Path:
    return Path(cache_dir) / f"{sample_id}.npy"


def save_vector_cache(cache_dir: Path, sample_id: str, vector: np.ndarray) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = vector_cache_path(cache_dir, sample_id)
    np.save(p, vector.astype(np.float32))
    return p


def load_vector_cache(cache_dir: Path, sample_id: str) -> np.ndarray:
    p = vector_cache_path(cache_dir, sample_id)
    if not p.exists():
        raise KeyError(f"No cached vector for sample_id={sample_id} at {p}")
    return np.load(p)


# ---------------------------------------------------------------------------
# Multi-process Phase 4/5 driver (same math as `process_clip` above; this
# only changes *how many clips run concurrently*, never what is computed).
#
# Each worker process saves its own vector straight to the shared disk cache
# and only returns the small QC dict to the parent -- the full-resolution
# raw/detail arrays never have to be pickled back over IPC.
# ---------------------------------------------------------------------------

_WORKER_CTX: dict[str, Any] = {}


def init_pool_worker(cfg: MLAttackConfig, valid_mask: np.ndarray, vector_cache_dir: Path) -> None:
    """`ProcessPoolExecutor(initializer=..., initargs=...)` entry point.

    Runs once per worker process (not once per clip), so `cfg` and
    `valid_mask` are pickled across the process boundary O(n_workers) times
    total, not O(n_clips) times.
    """
    _WORKER_CTX["cfg"] = cfg
    _WORKER_CTX["valid_mask"] = valid_mask
    _WORKER_CTX["vector_cache_dir"] = Path(vector_cache_dir)


def _error_qc_row(sample_id: str, message: str) -> dict[str, Any]:
    qc = ClipResult(
        sample_id=sample_id,
        decode_status="ERROR_WORKER_EXCEPTION",
        duration_s=None,
        fps=None,
        frame_count_total=None,
        frame_count_decoded=0,
        frame_count_used=0,
        mean_intensity=None,
        std_intensity=None,
        saturation_fraction=None,
        centroid_x=None,
        centroid_y=None,
        first_half_vs_second_half_ncc=None,
        mean_intensity_first_half=None,
        mean_intensity_second_half=None,
        relative_intensity_drift=None,
        centroid_first_half_x=None,
        centroid_first_half_y=None,
        centroid_second_half_x=None,
        centroid_second_half_y=None,
        centroid_drift_px=None,
        qc_status="ERROR",
        qc_reasons=[f"worker_exception:{message}"],
    )
    return clip_result_to_row(qc)


def process_and_cache_clip_worker(video_path_str: str, sample_id: str) -> dict[str, Any]:
    """Runs inside a `ProcessPoolExecutor` worker (after `init_pool_worker`).

    A single unexpectedly-failing clip must never take down the whole pool
    or silently drop a row -- any exception is converted into an explicit
    ERROR QC row instead of propagating.
    """
    cfg = _WORKER_CTX["cfg"]
    valid_mask = _WORKER_CTX["valid_mask"]
    vector_cache_dir = _WORKER_CTX["vector_cache_dir"]
    try:
        raw_template, detail, qc = process_clip(Path(video_path_str), sample_id=sample_id, valid_mask=valid_mask, cfg=cfg)
        if detail is not None:
            vector = mask_vector(detail, valid_mask)
            save_vector_cache(vector_cache_dir, sample_id, vector)
        return clip_result_to_row(qc)
    except Exception as exc:  # pragma: no cover - defensive, exercised only on real corrupt data
        return _error_qc_row(sample_id, repr(exc))


def process_and_cache_clip(
    video_path: Path,
    *,
    sample_id: str,
    valid_mask: np.ndarray,
    cfg: MLAttackConfig,
    vector_cache_dir: Path,
) -> dict[str, Any]:
    """Single-process convenience wrapper (used when `n_workers<=1`): process
    one clip and immediately persist its masked detail vector to disk,
    returning the QC row. Identical outcome to the parallel path, just
    in-process."""
    raw_template, detail, qc = process_clip(video_path, sample_id=sample_id, valid_mask=valid_mask, cfg=cfg)
    if detail is not None:
        vector = mask_vector(detail, valid_mask)
        save_vector_cache(vector_cache_dir, sample_id, vector)
    return clip_result_to_row(qc)
