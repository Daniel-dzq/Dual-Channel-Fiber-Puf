"""Green-only macro-pixel screening analysis (no plots).

Pipeline:
  color BGR -> green plane only
  dark subtraction
  fixed ROI + global valid_mask
  three fixed time-window means per challenge video
  detail_cm = local_ratio detail - shared common template
  intra (24) and inter-template (28) NCC metrics
  rank macros by robust_gap_detail_cm

Red channel is ignored (green-only screening mode).
Only detail_cm is retained for ranking and reporting.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from e01.analysis.envelope import (
    estimate_speckle_width,
    local_ratio_detail,
)
from e01.analysis.metrics_basic import auc_roc, d_prime, equal_error_rate
from e01.analysis.ncc import zero_mean_ncc
from e01.config import AppConfig
from e01.naming import PathLayout, challenge_grid_size, mp_tag
from e01.utils.io_utils import assert_writable, ensure_dir
from puf_common.tqdm_progress import mute_console_logging, stage_tqdm

logger = logging.getLogger(__name__)


@dataclass
class ChallengeWindows:
    macro_pixel: int
    challenge_id: str
    path: Path
    fps: float
    n_frames: int
    duration_s: float
    g_windows: list[np.ndarray]  # length 3, float64
    mean_intensity: float
    saturation_fraction: float


def parse_macro_from_dirname(name: str) -> int | None:
    if not name.lower().startswith("mp"):
        return None
    digits = name[2:]
    if not digits.isdigit():
        return None
    return int(digits)


def apply_roi(image: np.ndarray, roi: list[int] | None) -> np.ndarray:
    if roi is None:
        return image
    x, y, w, h = [int(v) for v in roi]
    return image[y : y + h, x : x + w]


def split_green(frame: np.ndarray) -> np.ndarray:
    """Extract green plane from OpenCV BGR. Reject grayscale."""
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError(
            f"Expected color BGR frame, got shape {getattr(frame, 'shape', None)}. "
            "Grayscale video cannot be used for this screening pipeline."
        )
    return frame[:, :, 1]


def resolve_video(folder: Path, challenge_id: str) -> Path:
    """Resolve challenge video path.

    Accepts both C01 and plain numeric names such as 1.mp4.
    """
    stems: list[str] = [challenge_id]
    cid = challenge_id.strip()
    if cid.upper().startswith("C") and cid[1:].isdigit():
        n = int(cid[1:])
        stems.extend([str(n), f"{n:02d}", f"C{n}", f"C{n:02d}"])
    elif cid.isdigit():
        n = int(cid)
        stems.extend([str(n), f"{n:02d}", f"C{n}", f"C{n:02d}"])

    seen: set[str] = set()
    ordered_stems: list[str] = []
    for s in stems:
        if s not in seen:
            seen.add(s)
            ordered_stems.append(s)

    suffixes = [".mp4", ".avi", ".mov", ".mkv"]
    for stem in ordered_stems:
        for suf in suffixes:
            path = folder / f"{stem}{suf}"
            if path.exists():
                return path
        seq = folder / f"{stem}_frames"
        if seq.is_dir():
            return seq
    raise FileNotFoundError(f"Missing video for {challenge_id} under {folder}")


_EXPECTED_DARK_SHAPE = (1536, 2048)


def require_dark_shape(
    dark: np.ndarray | float, frame_g: np.ndarray
) -> np.ndarray | float:
    """Require the dark map to match the green frame; do not resample."""
    if not isinstance(dark, np.ndarray):
        return dark
    if dark.shape != frame_g.shape:
        raise ValueError(
            f"Dark template shape {dark.shape} does not match frame shape "
            f"{frame_g.shape}."
        )
    return dark


def prepare_dark_for_videos(
    dark: np.ndarray | float,
    sample_path: Path,
    cfg: AppConfig,
) -> np.ndarray | float:
    """Confirm dark shape against the first challenge frame before the progress bar."""
    if not isinstance(dark, np.ndarray):
        return dark
    if sample_path.is_dir():
        files = sorted(
            p
            for p in sample_path.iterdir()
            if p.suffix.lower() in {".png", ".tif", ".tiff", ".bmp"}
        )
        if not files:
            return dark
        arr = np.array(Image.open(files[0]))
        g = apply_roi(arr[:, :, 1].astype(np.float64), cfg.camera.roi)
        return require_dark_shape(dark, g)

    cap = cv2.VideoCapture(str(sample_path))
    if not cap.isOpened():
        return dark
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return dark
    g = apply_roi(split_green(frame).astype(np.float64), cfg.camera.roi)
    return require_dark_shape(dark, g)


def discover_screening_videos(
    videos_root: Path,
    macro_sizes: list[int],
    challenge_ids: list[str],
    allow_partial: bool,
) -> dict[int, dict[str, Path]]:
    """Discover videos/mpXXX/CYY.* for requested macros."""
    by_macro: dict[int, Path] = {}
    if not videos_root.exists():
        raise FileNotFoundError(f"Videos root not found: {videos_root}")
    for d in videos_root.iterdir():
        if not d.is_dir() or d.name.startswith("."):
            continue
        if d.name.lower() in {"dark", "control", "calibration"}:
            continue
        m = parse_macro_from_dirname(d.name)
        if m is None:
            continue
        by_macro[m] = d

    found: dict[int, dict[str, Path]] = {}
    missing_macros: list[int] = []
    for m in macro_sizes:
        folder = by_macro.get(m)
        if folder is None:
            # also try canonical tag
            folder = videos_root / mp_tag(m)
        if folder is None or not folder.exists():
            missing_macros.append(m)
            continue
        pair: dict[str, Path] = {}
        incomplete = False
        for cid in challenge_ids:
            try:
                pair[cid] = resolve_video(folder, cid)
            except FileNotFoundError:
                incomplete = True
                logger.warning("Missing %s for macro=%d", cid, m)
        if incomplete:
            missing_macros.append(m)
            if allow_partial and pair:
                logger.warning(
                    "Macro %d incomplete (%d/%d challenges); skipped for ranking",
                    m,
                    len(pair),
                    len(challenge_ids),
                )
            continue
        found[m] = pair
        logger.info("Macro %d complete: %d challenges in %s", m, len(pair), folder)

    if not found:
        raise FileNotFoundError(
            f"No complete macro folders with {challenge_ids} under {videos_root}. "
            f"Missing/incomplete: {missing_macros}"
        )
    if missing_macros:
        logger.warning("Incomplete or missing macros (skipped): %s", missing_macros)
    return found


def load_dark_green(cfg: AppConfig) -> tuple[np.ndarray | float, bool]:
    """Load the 2048×1536 dark-green map, or a scalar if no artifact is configured."""
    path_str = cfg.analysis.dark_artifact_path
    if not path_str:
        logger.warning(
            "No dark artifact configured; using dark_green_scalar=%s",
            cfg.analysis.dark_green_scalar,
        )
        return float(cfg.analysis.dark_green_scalar), False

    path = Path(path_str)
    if not path.is_absolute():
        path = cfg.root / path
    if not path.exists():
        logger.warning("Dark artifact not found (%s); using scalar 0", path)
        return float(cfg.analysis.dark_green_scalar), False

    data = np.load(path)
    if "dark_green" not in data.files:
        raise ValueError(f"Dark NPZ must contain dark_green: {path}")
    dark = apply_roi(np.asarray(data["dark_green"], dtype=np.float64), cfg.camera.roi)
    if dark.shape != _EXPECTED_DARK_SHAPE:
        raise ValueError(
            f"Dark template shape {dark.shape} does not match recordings "
            f"{_EXPECTED_DARK_SHAPE}."
        )
    logger.info("Loaded dark green from %s shape=%s", path, dark.shape)
    return dark, True


def window_frame_groups(
    n_frames: int,
    fps: float,
    edges: list[float],
    frames_per_window: int,
) -> list[list[int]]:
    """Return frame index lists for each time window."""
    if fps <= 0 or not np.isfinite(fps):
        raise ValueError(f"Invalid FPS: {fps}")
    groups: list[list[int]] = []
    for i in range(len(edges) - 1):
        t0, t1 = float(edges[i]), float(edges[i + 1])
        f0 = int(np.floor(t0 * fps))
        f1 = int(np.ceil(t1 * fps))
        f0 = max(0, min(f0, n_frames - 1))
        f1 = max(f0 + 1, min(f1, n_frames))
        candidates = list(range(f0, f1))
        if not candidates:
            candidates = [f0]
        if len(candidates) <= frames_per_window:
            chosen = candidates
        else:
            picks = np.linspace(0, len(candidates) - 1, frames_per_window)
            chosen = []
            seen = set()
            for p in picks:
                idx = candidates[int(round(p))]
                if idx not in seen:
                    seen.add(idx)
                    chosen.append(idx)
        groups.append(chosen)
    return groups


def read_challenge_windows(
    path: Path,
    macro_pixel: int,
    challenge_id: str,
    cfg: AppConfig,
    dark: np.ndarray | float,
) -> ChallengeWindows:
    """Read one challenge video and build 3 green window-mean images."""
    if path.is_dir():
        files = sorted(
            p for p in path.iterdir() if p.suffix.lower() in {".png", ".tif", ".tiff", ".bmp"}
        )
        if not files:
            raise FileNotFoundError(f"No frames in {path}")
        frames = []
        for f in files:
            arr = np.array(Image.open(f))
            if arr.ndim == 2:
                raise ValueError(f"Grayscale frame not allowed: {f}")
            # PIL RGB -> use G channel index 1 same as RGB G
            g = arr[:, :, 1].astype(np.float64)
            frames.append(apply_roi(g, cfg.camera.roi))
        fps = float(cfg.camera.requested_fps or 30.0)
        n_frames = len(frames)
        duration_s = n_frames / fps
        groups = window_frame_groups(
            n_frames, fps, cfg.analysis.window_edges_seconds, cfg.analysis.frames_per_window
        )
        g_windows = []
        sat = 0
        pix = 0
        means = []
        dark_aligned = require_dark_shape(dark, frames[0])
        for group in groups:
            acc = None
            for idx in group:
                g = np.clip(frames[idx] - dark_aligned, 0.0, None)
                acc = g if acc is None else acc + g
                sat += int(np.count_nonzero(g >= cfg.analysis.saturation_threshold))
                pix += int(g.size)
                means.append(float(np.mean(g)))
            g_windows.append(acc / len(group))
        return ChallengeWindows(
            macro_pixel=macro_pixel,
            challenge_id=challenge_id,
            path=path,
            fps=fps,
            n_frames=n_frames,
            duration_s=duration_s,
            g_windows=g_windows,
            mean_intensity=float(np.mean(means)),
            saturation_fraction=sat / max(pix, 1),
        )

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0:
        fps = float(cfg.camera.requested_fps or 30.0)
        logger.warning("Invalid FPS in %s; using fallback %.3f", path, fps)
    if n_frames <= 0:
        # count
        n_frames = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            n_frames += 1
        cap.release()
        cap = cv2.VideoCapture(str(path))

    duration_s = n_frames / fps
    groups = window_frame_groups(
        n_frames, fps, cfg.analysis.window_edges_seconds, cfg.analysis.frames_per_window
    )
    needed = sorted({i for g in groups for i in g})
    needed_set = set(needed)
    index_to_windows: dict[int, list[int]] = {}
    for wi, group in enumerate(groups):
        for idx in group:
            index_to_windows.setdefault(idx, []).append(wi)

    sums: list[np.ndarray | None] = [None] * (len(groups))
    counts = [0] * len(groups)
    sat = 0
    pix = 0
    means: list[float] = []
    dark_aligned: np.ndarray | float | None = None

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in needed_set:
            g = apply_roi(split_green(frame).astype(np.float64), cfg.camera.roi)
            if dark_aligned is None:
                dark_aligned = require_dark_shape(dark, g)
            g = np.clip(g - dark_aligned, 0.0, None)
            for wi in index_to_windows[frame_idx]:
                if sums[wi] is None:
                    sums[wi] = np.zeros_like(g)
                sums[wi] += g
                counts[wi] += 1
            sat += int(np.count_nonzero(g >= cfg.analysis.saturation_threshold))
            pix += int(g.size)
            means.append(float(np.mean(g)))
        frame_idx += 1
    cap.release()

    g_windows = []
    for wi in range(len(groups)):
        if sums[wi] is None or counts[wi] == 0:
            raise RuntimeError(f"Window {wi} empty in {path}")
        g_windows.append(sums[wi] / counts[wi])

    return ChallengeWindows(
        macro_pixel=macro_pixel,
        challenge_id=challenge_id,
        path=path,
        fps=fps,
        n_frames=n_frames,
        duration_s=duration_s,
        g_windows=g_windows,
        mean_intensity=float(np.mean(means)) if means else float("nan"),
        saturation_fraction=sat / max(pix, 1),
    )


def build_valid_mask(
    green_refs: list[np.ndarray],
    cfg: AppConfig,
) -> tuple[np.ndarray, float]:
    """Build one global mask from green reference means (locked for all macros)."""
    layout = PathLayout(cfg.root)
    if cfg.analysis.valid_mask_path:
        p = Path(cfg.analysis.valid_mask_path)
        if not p.is_absolute():
            p = cfg.root / p
        if p.exists():
            arr = np.array(Image.open(p).convert("L"))
            arr = apply_roi(arr, cfg.camera.roi)
            mask = arr > 0
            return mask, float(mask.mean())

    stack = np.stack(green_refs, axis=0)
    ref = np.mean(stack, axis=0)
    thr = max(
        float(cfg.analysis.valid_mask_abs_floor),
        float(np.percentile(ref, cfg.analysis.valid_mask_percentile)),
    )
    mask = ref > thr
    frac = float(mask.mean())
    if int(np.count_nonzero(mask)) < 64:
        raise RuntimeError(f"valid_mask too small ({int(np.count_nonzero(mask))} pixels)")
    # save for reproducibility
    out = layout.screening / "valid_mask.png"
    ensure_dir(out.parent)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(out)
    logger.info("Built valid_mask thr=%.3f fraction=%.4f -> %s", thr, frac, out)
    return mask, frac


def compute_detail_cm_metrics(
    challenges: dict[str, ChallengeWindows],
    mask: np.ndarray,
    sigma: float,
    eps: float,
) -> dict[str, float]:
    """Only retained screening branch: local_ratio detail + one shared common.

    Model: G ~= E * (1 + S_c). Local-ratio removes slow envelope E. Remaining
    challenge-invariant fiber/imaging modes are removed once per macro:

        common = mean_c template_c(detail)
        residual = detail - common

    Intra and inter both use this same residual (unified representation).
    """
    ids = sorted(challenges.keys())
    detail_windows: dict[str, list[np.ndarray]] = {}
    detail_templates: dict[str, np.ndarray] = {}
    for cid in ids:
        ws = [
            local_ratio_detail(g, sigma=sigma, eps_env=eps)
            for g in challenges[cid].g_windows
        ]
        detail_windows[cid] = ws
        detail_templates[cid] = np.mean(np.stack(ws, axis=0), axis=0)

    common = np.mean(np.stack([detail_templates[cid] for cid in ids], axis=0), axis=0)

    windows: dict[str, list[np.ndarray]] = {}
    templates: dict[str, np.ndarray] = {}
    contrasts: list[float] = []
    for cid in ids:
        ws = [w - common for w in detail_windows[cid]]
        windows[cid] = ws
        templates[cid] = detail_templates[cid] - common
        sel = templates[cid][mask]
        mu = float(np.mean(np.abs(sel)))
        sd = float(np.std(sel))
        contrasts.append(sd / mu if mu > 1e-12 else float("nan"))

    intra: list[float] = []
    for cid in ids:
        ws = windows[cid]
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                intra.append(zero_mean_ncc(ws[i], ws[j], mask=mask))

    inter: list[float] = []
    for i, ci in enumerate(ids):
        for cj in ids[i + 1 :]:
            inter.append(zero_mean_ncc(templates[ci], templates[cj], mask=mask))

    return _pack_ncc_metrics("detail_cm", intra, inter, contrasts)


def _pack_ncc_metrics(
    branch: str,
    intra: list[float],
    inter: list[float],
    contrasts: list[float],
) -> dict[str, float]:
    intra_a = np.asarray(intra, dtype=np.float64)
    inter_a = np.asarray(inter, dtype=np.float64)
    median_intra = float(np.median(intra_a))
    median_inter = float(np.median(inter_a))
    robust = float(np.percentile(intra_a, 5) - np.percentile(inter_a, 95))
    return {
        f"median_intra_ncc_{branch}": median_intra,
        f"median_inter_ncc_{branch}": median_inter,
        f"delta_median_{branch}": median_intra - median_inter,
        f"robust_gap_{branch}": robust,
        f"d_prime_{branch}": d_prime(intra_a, inter_a),
        f"eer_{branch}": equal_error_rate(intra_a, inter_a),
        f"auc_{branch}": auc_roc(intra_a, inter_a),
        f"speckle_contrast_{branch}": float(np.nanmean(contrasts)),
        f"n_intra_{branch}": int(intra_a.size),
        f"n_inter_{branch}": int(inter_a.size),
        f"intra_p05_{branch}": float(np.percentile(intra_a, 5)),
        f"inter_p95_{branch}": float(np.percentile(inter_a, 95)),
    }


def rank_macros(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Rank by robust_gap_detail_cm; prefer smaller m on near-ties."""

    def metric_key(r: dict[str, object]) -> tuple:
        return (
            -int(r.get("quality_pass", 0)),
            -float(r["robust_gap_detail_cm"]),
            float(r["median_inter_ncc_detail_cm"]),
            -float(r["median_intra_ncc_detail_cm"]),
            -float(r["auc_detail_cm"]),
            float(r["eer_detail_cm"]),
            -float(r["d_prime_detail_cm"]),
        )

    if not rows:
        return rows

    ordered = sorted(rows, key=metric_key)
    top = [r for r in ordered if int(r.get("quality_pass", 0)) == 1]
    pool = top if top else ordered
    best_gap = float(pool[0]["robust_gap_detail_cm"])
    near = [
        r
        for r in pool
        if float(r["robust_gap_detail_cm"]) >= 0.95 * best_gap
        or (best_gap <= 0 and float(r["robust_gap_detail_cm"]) >= best_gap - 1e-12)
    ]
    preferred = min(near, key=lambda r: int(r["macro_pixel"]))

    rest = [r for r in ordered if int(r["macro_pixel"]) != int(preferred["macro_pixel"])]
    final = [preferred] + rest
    for i, r in enumerate(final, start=1):
        r["rank"] = i
        r["recommended"] = int(int(r["macro_pixel"]) == int(preferred["macro_pixel"]))
    return final


def _write_csv(path: Path, rows: list[dict[str, object]], force: bool) -> None:
    ensure_dir(path.parent)
    assert_writable(path, force=force)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_green_screening(cfg: AppConfig, force: bool = False) -> dict[str, Path]:
    """Run green-only macro-pixel screening and write CSV/log (no figures)."""
    layout = PathLayout(cfg.root)
    out_dir = layout.screening
    ensure_dir(out_dir)

    challenge_ids = list(cfg.experiment.challenge_ids)
    dark, dark_ok = load_dark_green(cfg)
    discovered = discover_screening_videos(
        layout.videos,
        cfg.experiment.macro_pixel_sizes,
        challenge_ids,
        allow_partial=cfg.analysis.allow_partial_macros,
    )

    # Load all complete macros
    loaded: dict[int, dict[str, ChallengeWindows]] = {}
    video_rows: list[dict[str, object]] = []
    green_refs: list[np.ndarray] = []

    read_jobs = [
        (m, cid, path)
        for m, paths in sorted(discovered.items())
        for cid, path in sorted(paths.items())
    ]
    if not read_jobs:
        raise FileNotFoundError(
            f"No screening videos found under {layout.videos} "
            f"for macros={cfg.experiment.macro_pixel_sizes} "
            f"challenges={challenge_ids}"
        )

    dark = prepare_dark_for_videos(dark, read_jobs[0][2], cfg)

    with mute_console_logging():
        for m, cid, path in stage_tqdm(
            read_jobs, desc="Reading videos", unit="video", total=len(read_jobs)
        ):
            loaded.setdefault(m, {})
            cw = read_challenge_windows(path, m, cid, cfg, dark)
            loaded[m][cid] = cw
            green_refs.append(np.mean(np.stack(cw.g_windows, axis=0), axis=0))
            video_rows.append(
                {
                    "macro_pixel": m,
                    "challenge_id": cid,
                    "path": str(path.relative_to(cfg.root))
                    if path.is_relative_to(cfg.root)
                    else str(path),
                    "fps": cw.fps,
                    "n_frames": cw.n_frames,
                    "duration_s": cw.duration_s,
                    "mean_green_intensity": cw.mean_intensity,
                    "green_saturation_fraction": cw.saturation_fraction,
                    "n_windows": len(cw.g_windows),
                    "dark_correction_available": int(dark_ok),
                }
            )

    mask, mask_frac = build_valid_mask(green_refs, cfg)

    # Envelope sigma from development estimate or auto from first available refs
    if cfg.analysis.envelope_sigma_px is not None:
        sigma = float(cfg.analysis.envelope_sigma_px)
        ws = sigma / float(cfg.analysis.envelope_sigma_factor)
    else:
        ws = float(np.median([estimate_speckle_width(g, mask) for g in green_refs[:8]]))
        ws = float(np.clip(ws, 2.0, 40.0))
        sigma = float(cfg.analysis.envelope_sigma_factor) * ws
    # Hard cap: huge sigma makes GaussianBlur dominate runtime on 2k frames.
    if sigma > 160.0:
        logger.warning("Clamping envelope_sigma_px from %.3f to 160.0", sigma)
        sigma = 160.0
    logger.info("Speckle width estimate ws=%.3f px; envelope sigma=%.3f px", ws, sigma)

    summary_rows: list[dict[str, object]] = []
    # detail_cm + sigma sensitivity (still on detail_cm only).
    step_jobs: list[tuple[int, str, float | None]] = []
    for m in sorted(loaded):
        step_jobs.append((m, "detail_cm", None))
        for fac in cfg.analysis.sensitivity_sigma_factors:
            step_jobs.append((m, f"sens{int(fac)}", float(fac)))

    by_macro_rows: dict[int, dict[str, object]] = {}
    sens_acc: dict[int, list[float]] = {m: [] for m in loaded}

    with mute_console_logging():
        for m, label, fac in stage_tqdm(
            step_jobs, desc="Computing metrics", unit="step", total=len(step_jobs)
        ):
            challenges = loaded[m]
            if m not in by_macro_rows:
                by_macro_rows[m] = {
                    "macro_pixel": m,
                    "challenge_grid_size": challenge_grid_size(cfg.slm.active_width, m),
                    "n_challenges": len(challenges),
                    "envelope_sigma_px": sigma,
                    "speckle_width_px": ws,
                    "valid_mask_fraction": mask_frac,
                    "dark_correction_available": int(dark_ok),
                    "mean_green_intensity": float(
                        np.mean([c.mean_intensity for c in challenges.values()])
                    ),
                    "green_saturation_fraction": float(
                        np.max([c.saturation_fraction for c in challenges.values()])
                    ),
                }
            row = by_macro_rows[m]

            if fac is None:
                row.update(
                    compute_detail_cm_metrics(
                        challenges, mask, sigma, cfg.analysis.envelope_eps
                    )
                )
            else:
                sig_s = float(min(float(fac) * ws, 160.0))
                m_s = compute_detail_cm_metrics(
                    challenges, mask, sig_s, cfg.analysis.envelope_eps
                )
                sens_acc[m].append(m_s["robust_gap_detail_cm"])

            if fac is not None and len(sens_acc[m]) == len(
                cfg.analysis.sensitivity_sigma_factors
            ):
                sens = sens_acc[m]
                row["robust_gap_detail_cm_sens_min"] = float(min(sens))
                row["robust_gap_detail_cm_sens_max"] = float(max(sens))
                row["robust_gap_detail_cm_sens_range"] = float(max(sens) - min(sens))

                warnings: list[str] = []
                if not dark_ok:
                    warnings.append("dark_field_missing")
                if float(row["green_saturation_fraction"]) > cfg.analysis.max_green_saturation:
                    warnings.append("green_saturation_high")
                if (
                    float(row["median_intra_ncc_detail_cm"])
                    < cfg.analysis.min_intra_median_detail
                ):
                    warnings.append("detail_cm_intra_low")
                if float(row["robust_gap_detail_cm"]) <= 0:
                    warnings.append("detail_cm_distributions_overlap")

                quality_pass = (
                    dark_ok
                    and float(row["green_saturation_fraction"])
                    <= cfg.analysis.max_green_saturation
                    and float(row["median_intra_ncc_detail_cm"])
                    >= cfg.analysis.min_intra_median_detail
                    and float(row["robust_gap_detail_cm"]) > 0
                    and len(challenges) == len(challenge_ids)
                )
                row["quality_pass"] = int(quality_pass)
                row["quality_warning"] = ";".join(warnings)
                summary_rows.append(row)

    ranked = rank_macros(summary_rows)
    # CSV sorted by macro_pixel
    by_macro = {int(r["macro_pixel"]): r for r in ranked}
    summary_csv_rows = [by_macro[m] for m in sorted(by_macro)]

    info_path = out_dir / "video_information.csv"
    summary_path = out_dir / "macro_pixel_summary.csv"
    log_path = out_dir / "analysis_log.txt"
    params_path = out_dir / "locked_analysis_params.json"

    _write_csv(info_path, video_rows, force=force)
    _write_csv(summary_path, summary_csv_rows, force=force)

    params = {
        "mode": "green_only",
        "window_edges_seconds": cfg.analysis.window_edges_seconds,
        "frames_per_window": cfg.analysis.frames_per_window,
        "roi": cfg.camera.roi,
        "envelope_sigma_px": sigma,
        "speckle_width_px": ws,
        "envelope_eps": cfg.analysis.envelope_eps,
        "valid_mask_fraction": mask_frac,
        "dark_correction_available": dark_ok,
        "challenge_ids": challenge_ids,
        "ranking_branch": "detail_cm",
        "common_mode": "mean_challenge_template_after_detail",
        "note": (
            "Only detail_cm is used: local_ratio then ONE shared common template; "
            "intra/inter share the same residual space."
        ),
    }
    assert_writable(params_path, force=force)
    params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")

    recommended = next((r for r in ranked if int(r["recommended"]) == 1), None)
    lines = [
        "Green-only macro-pixel screening",
        "=" * 60,
        "Only branch: detail_cm = local_ratio + shared common subtraction.",
        "Red channel is NOT used. raw/detail/highpass are not reported.",
        f"dark_correction_available={dark_ok}",
        f"envelope_sigma_px={sigma:.4f} (ws={ws:.4f})",
        f"valid_mask_fraction={mask_frac:.6f}",
        "",
        f"{'macro':>5} {'GinCM':>7} {'GinterCM':>8} {'robCM':>7} "
        f"{'aucCM':>7} {'dprm':>7} {'pass':>4} {'rank':>4}",
    ]
    print(lines[-1])
    for r in summary_csv_rows:
        line = (
            f"{int(r['macro_pixel']):5d} "
            f"{float(r['median_intra_ncc_detail_cm']):7.3f} "
            f"{float(r['median_inter_ncc_detail_cm']):8.3f} "
            f"{float(r['robust_gap_detail_cm']):7.3f} "
            f"{float(r['auc_detail_cm']):7.3f} "
            f"{float(r['d_prime_detail_cm']):7.3f} "
            f"{int(r['quality_pass']):4d} "
            f"{int(r['rank']):4d}"
        )
        lines.append(line)
        print(line)

    lines.append("")
    if recommended is not None:
        lines.append(
            f"Preliminary recommended macro_pixel_size: {recommended['macro_pixel']} "
            f"(grid={recommended['challenge_grid_size']})"
        )
        lines.append(
            f"  robust_gap_detail_cm={float(recommended['robust_gap_detail_cm']):.4f} "
            f"median_inter_detail_cm={float(recommended['median_inter_ncc_detail_cm']):.4f} "
            f"median_intra_detail_cm={float(recommended['median_intra_ncc_detail_cm']):.4f}"
        )
    else:
        lines.append("No recommendation available.")
    lines.append(
        "This is a green-only screening study with 8 challenges/size; "
        "not a formal 256-challenge PUF performance claim."
    )
    if not dark_ok:
        lines.append("WARNING: dark field missing; results may be biased.")

    assert_writable(log_path, force=force)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines[-10:]:
        logger.info("%s", line)

    return {
        "video_information": info_path,
        "summary": summary_path,
        "log": log_path,
        "params": params_path,
    }
