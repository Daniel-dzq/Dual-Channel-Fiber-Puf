"""Nature-style Fig. 3 panels (b–e) for Experiment 1 macro-pixel screening.

No in-figure titles. Arial. Large type. Legends outside data.
Score classes: Genuine / Challenge-mismatch only (single device; no Device mismatch).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from PIL import Image

from e01.analysis.envelope import local_ratio_detail
from e01.analysis.green_screening import (
    build_valid_mask,
    discover_screening_videos,
    load_dark_green,
    prepare_dark_for_videos,
    read_challenge_windows,
)
from e01.analysis.ncc import zero_mean_ncc
from e01.config import AppConfig, load_config
from e01.naming import PathLayout, mp_tag
from e01.utils.io_utils import ensure_dir
from puf_common.tqdm_progress import mute_console_logging, stage_tqdm

logger = logging.getLogger(__name__)

# Colorblind-friendly Nature-like palette (NOT red/green channels).
INTRA_COLOR = "#3C5488"   # navy blue
INTER_COLOR = "#E69F00"   # amber / orange
ACCENT_M2 = "#4DBBD5"     # light teal highlight for selected m=2
GAP_COLOR = "#00A087"     # NPG teal-green for the robust gap G(m)
SELECT_BAND = "#E7EEF3"   # soft blue-gray band marking the selected m
# Near-tie window used by rank_macros() when preferring the smallest m.
TIE_TOLERANCE = 0.95

ALL_MACROS = [1, 2, 4, 8, 16, 32, 64]
FIG_MACROS_B = ALL_MACROS
MATRIX_MACRO = 2

# Nature / Light publication palette (fig3c dual heatmaps)
C_TEXT = "#2F2F2F"       # near-black for titles / ticks
C_ANNOT_DARK = "#3A3A3A"  # dark gray on light cells
C_SPINE = "#BDBDB8"       # thin light-gray frame
C_GRID = "#FFFFFF"        # fine white cell separators
C_FACE = "#FFFFFF"


def _raw_sequential_cmap() -> LinearSegmentedColormap:
    """Muted warm sequential map for high-NCC Raw matrix (ivory → wine)."""
    stops = [
        (0.00, "#F7F4F1"),
        (0.30, "#E8D5CE"),
        (0.55, "#D9B8B0"),
        (0.75, "#C07A78"),
        (0.90, "#A04E55"),
        (1.00, "#8E3B46"),
    ]
    return LinearSegmentedColormap.from_list(
        "nature_raw_warm_seq",
        [(p, c) for p, c in stops],
        N=256,
    )


def _g_diverging_cmap() -> LinearSegmentedColormap:
    """Muted diverging map for signed NCC (slate blue ↔ neutral ↔ terracotta)."""
    stops = [
        (0.00, "#3F6F93"),  # desaturated slate blue — negative
        (0.22, "#6F93B0"),
        (0.40, "#C5D2DC"),
        (0.50, "#F7F7F5"),  # near-white at NCC = 0
        (0.60, "#E8D0C8"),
        (0.78, "#D08A78"),
        (1.00, "#B65C4A"),  # muted salmon / soft brick — positive
    ]
    return LinearSegmentedColormap.from_list(
        "nature_g_blue_terracotta",
        [(p, c) for p, c in stops],
        N=256,
    )


RAW_CMAP = _raw_sequential_cmap()
G_CMAP = _g_diverging_cmap()


def _annotation_text_color(rgba: tuple[float, ...]) -> str:
    """White on dark cells, deep gray on light cells."""
    r, g, b = rgba[:3]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return C_ANNOT_DARK if lum > 0.58 else "white"


def _pad_lim(lo: float, hi: float, *, step: float, pad: float) -> tuple[float, float]:
    """Round display limits outward to a clean step with a small pad."""
    vmin = np.floor((lo - pad) / step) * step
    vmax = np.ceil((hi + pad) / step) * step
    if abs(vmax - vmin) < step:
        vmax = vmin + step
    return float(vmin), float(vmax)


def _apply_nature_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "axes.unicode_minus": False,
            "font.size": 18,
            "axes.titlesize": 18,
            "axes.labelsize": 20,
            "xtick.labelsize": 15,
            "ytick.labelsize": 16,
            "legend.fontsize": 15,
            "axes.linewidth": 1.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "figure.dpi": 120,
            "savefig.dpi": 400,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    ensure_dir(stem.parent)
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _display_normalize(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float64)
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _read_mid_frame_bgr(video_path: Path, *, time_frac: float = 0.5) -> np.ndarray:
    """Grab one real camera frame near mid-clip (OpenCV BGR)."""
    import cv2

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n_frames <= 0:
            raise RuntimeError(f"Empty video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        # Exp1 windows sit in the middle of ~30 s clips; ~15 s is a safe pick.
        if fps > 1e-6 and n_frames / fps > 20.0:
            target = int(round(15.0 * fps))
        else:
            target = int(round((n_frames - 1) * float(np.clip(time_frac, 0.05, 0.95))))
        target = int(np.clip(target, 0, n_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame = None
            for _ in range(target + 1):
                ok, frame = cap.read()
                if not ok:
                    break
            if frame is None:
                raise RuntimeError(f"Failed to read frame from {video_path}")
        return frame
    finally:
        cap.release()


def _boost_display_rgb(rgb: np.ndarray, p_hi: float = 99.5) -> np.ndarray:
    arr = rgb.astype(np.float64)
    peak = float(np.percentile(arr, p_hi))
    if peak <= 1.0:
        return rgb.astype(np.uint8)
    scale = min(255.0 / peak, 4.0)
    return np.clip(arr * scale, 0, 255).astype(np.uint8)


def _bright_core_center(bgr: np.ndarray) -> tuple[int, int, int]:
    """Return (cy, cx, extent) for the illuminated core."""
    import cv2

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    thr = max(8.0, float(np.percentile(gray, 92)) * 0.25)
    ys, xs = np.where(gray >= thr)
    if ys.size < 50:
        ys, xs = np.where(gray >= max(5.0, float(np.percentile(gray, 99)) * 0.15))
    if ys.size < 20:
        return bgr.shape[0] // 2, bgr.shape[1] // 2, min(bgr.shape[0], bgr.shape[1]) // 2
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return (y0 + y1) // 2, (x0 + x1) // 2, int(max(y1 - y0 + 1, x1 - x0 + 1))


def _square_crop_bgr(bgr: np.ndarray, *, cy: int, cx: int, side: int) -> np.ndarray:
    side = int(max(32, side))
    half = side // 2
    y0 = cy - half
    x0 = cx - half
    out = np.zeros((side, side, 3), dtype=bgr.dtype)
    src_y0 = max(0, y0)
    src_x0 = max(0, x0)
    src_y1 = min(bgr.shape[0], y0 + side)
    src_x1 = min(bgr.shape[1], x0 + side)
    dst_y0 = src_y0 - y0
    dst_x0 = src_x0 - x0
    out[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)] = bgr[
        src_y0:src_y1, src_x0:src_x1
    ]
    return out


def _crop_bright_spot_rgb(
    bgr: np.ndarray,
    *,
    pad_frac: float = 0.18,
    min_side: int = 256,
    side: int | None = None,
    out_side: int = 512,
) -> np.ndarray:
    """Crop illuminated core to a fixed square; return true-color RGB for display."""
    import cv2

    cy, cx, extent = _bright_core_center(bgr)
    if side is None:
        side = int(max(extent, min_side))
        side = int(side + 2 * round(side * pad_frac))
    crop = _square_crop_bgr(bgr, cy=cy, cx=cx, side=side)
    if crop.shape[0] != out_side or crop.shape[1] != out_side:
        crop = cv2.resize(crop, (out_side, out_side), interpolation=cv2.INTER_AREA)
    return _boost_display_rgb(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def _challenge_labels(ids: list[str]) -> list[str]:
    """Axis labels with subscripted challenge index, e.g. C₀₁."""
    out = []
    for cid in ids:
        if cid.upper().startswith("C"):
            num = cid[1:]
            if num.isdigit():
                num = f"{int(num):02d}"
        else:
            num = f"{int(cid):02d}"
        out.append(rf"$C_{{{num}}}$")
    return out


def _find_pattern_png(patterns_root: Path, macro: int, challenge_id: str) -> Path:
    tag = mp_tag(macro)
    folder = patterns_root / tag
    labels = []
    if challenge_id.upper().startswith("C"):
        labels.append(challenge_id.upper())
        labels.append(str(int(challenge_id[1:])))
    else:
        labels.append(f"C{int(challenge_id):02d}")
        labels.append(str(int(challenge_id)))
    candidates: list[Path] = []
    for lab in labels:
        candidates.extend(
            [
                folder / lab / f"{tag}_{lab}.png",
                folder / f"{tag}_{lab}.png",
                folder / lab / "expanded_512.png",
                folder / lab / "binary.png",
            ]
        )
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Pattern image not found for m={macro} challenge={challenge_id}")


def _ncc_matrix(templates: dict[str, np.ndarray], ids: list[str], mask: np.ndarray) -> np.ndarray:
    n = len(ids)
    mat = np.eye(n, dtype=np.float64)
    for i, ci in enumerate(ids):
        for j in range(i + 1, n):
            cj = ids[j]
            v = zero_mean_ncc(templates[ci], templates[cj], mask=mask)
            mat[i, j] = v
            mat[j, i] = v
    return mat


def _branch_scores(
    challenges: dict[str, Any],
    mask: np.ndarray,
    *,
    branch: str,
    sigma: float,
    eps: float,
) -> tuple[list[float], list[float], dict[str, np.ndarray]]:
    ids = sorted(challenges.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
    if branch == "raw":
        windows = {cid: list(challenges[cid].g_windows) for cid in ids}
        templates = {
            cid: np.mean(np.stack(windows[cid], axis=0), axis=0) for cid in ids
        }
    elif branch == "detail_cm":
        detail_windows = {
            cid: [
                local_ratio_detail(g, sigma=sigma, eps_env=eps)
                for g in challenges[cid].g_windows
            ]
            for cid in ids
        }
        detail_templates = {
            cid: np.mean(np.stack(detail_windows[cid], axis=0), axis=0) for cid in ids
        }
        common = np.mean(np.stack([detail_templates[cid] for cid in ids], axis=0), axis=0)
        windows = {cid: [w - common for w in detail_windows[cid]] for cid in ids}
        templates = {cid: detail_templates[cid] - common for cid in ids}
    else:
        raise ValueError(f"Unsupported branch: {branch}")

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
    return intra, inter, templates


def build_or_load_fig_cache(
    cfg: AppConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    layout = PathLayout(cfg.root)
    cache_dir = layout.screening / "fig3_cache"
    cache_meta = cache_dir / "cache_meta.json"
    ensure_dir(cache_dir)

    if cache_meta.exists() and not force:
        logger.info("Loading Fig.3 cache from %s", cache_dir)
        meta = json.loads(cache_meta.read_text(encoding="utf-8"))
        data: dict[str, Any] = {"meta": meta, "by_macro": {}}
        for m in meta["macros"]:
            z = np.load(cache_dir / f"m{int(m):03d}.npz", allow_pickle=False)
            data["by_macro"][int(m)] = {
                "intra_detail_cm": z["intra_detail_cm"],
                "inter_detail_cm": z["inter_detail_cm"],
                "intra_p05": float(z["intra_p05"]),
                "inter_p95": float(z["inter_p95"]),
                "raw_matrix": z["raw_matrix"] if "raw_matrix" in z.files else None,
                "detail_cm_matrix": z["detail_cm_matrix"] if "detail_cm_matrix" in z.files else None,
                "rep_raw_c01": z["rep_raw_c01"] if "rep_raw_c01" in z.files else None,
            }
        mask = np.array(Image.open(layout.screening / "valid_mask.png").convert("L")) > 0
        data["mask"] = mask
        data["challenge_ids"] = meta["challenge_ids"]
        data["sigma"] = float(meta["sigma"])
        return data

    challenge_ids = list(cfg.experiment.challenge_ids)
    dark, _ = load_dark_green(cfg)
    discovered = discover_screening_videos(
        layout.videos,
        cfg.experiment.macro_pixel_sizes,
        challenge_ids,
        allow_partial=False,
    )
    read_jobs = [
        (m, cid, path)
        for m, paths in sorted(discovered.items())
        for cid, path in sorted(paths.items())
    ]
    dark = prepare_dark_for_videos(dark, read_jobs[0][2], cfg)

    loaded: dict[int, dict[str, Any]] = {}
    green_refs: list[np.ndarray] = []
    with mute_console_logging():
        for m, cid, path in stage_tqdm(
            read_jobs, desc="Reading videos", unit="video", total=len(read_jobs)
        ):
            loaded.setdefault(m, {})
            cw = read_challenge_windows(path, m, cid, cfg, dark)
            loaded[m][cid] = cw
            green_refs.append(np.mean(np.stack(cw.g_windows, axis=0), axis=0))

    mask, _ = build_valid_mask(green_refs, cfg)
    if cfg.analysis.envelope_sigma_px is not None:
        sigma = float(cfg.analysis.envelope_sigma_px)
    else:
        sigma = 42.0
    eps = float(cfg.analysis.envelope_eps)

    by_macro: dict[int, dict[str, Any]] = {}
    macros = sorted(loaded)
    with mute_console_logging():
        for m in stage_tqdm(macros, desc="Fig.3 metrics", unit="macro", total=len(macros)):
            challenges = loaded[m]
            ids = sorted(challenges.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
            intra_cm, inter_cm, templates_cm = _branch_scores(
                challenges, mask, branch="detail_cm", sigma=sigma, eps=eps
            )
            intra_raw, inter_raw, templates_raw = _branch_scores(
                challenges, mask, branch="raw", sigma=sigma, eps=eps
            )
            rep_c01 = np.mean(np.stack(challenges[ids[0]].g_windows, axis=0), axis=0)
            entry = {
                "intra_detail_cm": np.asarray(intra_cm, dtype=np.float64),
                "inter_detail_cm": np.asarray(inter_cm, dtype=np.float64),
                "intra_p05": float(np.percentile(intra_cm, 5)),
                "inter_p95": float(np.percentile(inter_cm, 95)),
                "raw_matrix": _ncc_matrix(templates_raw, ids, mask),
                "detail_cm_matrix": _ncc_matrix(templates_cm, ids, mask),
                "rep_raw_c01": rep_c01.astype(np.float32),
            }
            by_macro[m] = entry
            np.savez_compressed(
                cache_dir / f"m{m:03d}.npz",
                intra_detail_cm=entry["intra_detail_cm"],
                inter_detail_cm=entry["inter_detail_cm"],
                intra_p05=entry["intra_p05"],
                inter_p95=entry["inter_p95"],
                raw_matrix=entry["raw_matrix"],
                detail_cm_matrix=entry["detail_cm_matrix"],
                rep_raw_c01=entry["rep_raw_c01"],
            )

    meta = {
        "macros": macros,
        "challenge_ids": challenge_ids,
        "sigma": sigma,
        "eps": eps,
        "matrix_macro": MATRIX_MACRO,
    }
    cache_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {
        "meta": meta,
        "by_macro": by_macro,
        "mask": mask,
        "challenge_ids": challenge_ids,
        "sigma": sigma,
    }


def plot_fig3b(cfg: AppConfig, cache: dict[str, Any], output_dir: Path) -> None:
    """SLM inputs (top) + real mid-clip green camera frames (bottom, true RGB)."""
    _apply_nature_style()
    macros = FIG_MACROS_B
    n = len(macros)
    fig, axes = plt.subplots(2, n, figsize=(1.72 * n, 4.55))
    cid0 = cache["challenge_ids"][0]
    layout = PathLayout(cfg.root)
    patterns_root = layout.patterns
    discovered = discover_screening_videos(
        layout.videos,
        macros,
        [cid0],
        allow_partial=False,
    )

    frames = [_read_mid_frame_bgr(discovered[m][cid0]) for m in macros]
    extents = [_bright_core_center(bgr)[2] for bgr in frames]
    shared_side = int(max(max(extents), 256))
    shared_side = int(shared_side + 2 * round(shared_side * 0.18))

    for col, m in enumerate(macros):
        ax_in = axes[0, col]
        ax_out = axes[1, col]
        png = _find_pattern_png(patterns_root, m, cid0)
        pat = np.array(Image.open(png).convert("L"))
        # Prefer active-region crop if full canvas.
        if pat.shape[0] >= 640 and pat.shape[1] >= 768:
            pat = pat[
                cfg.slm.y_start : cfg.slm.y_end,
                cfg.slm.x_start : cfg.slm.x_end,
            ]
        ax_in.imshow(pat, cmap="gray", vmin=0, vmax=255, aspect="equal")
        ax_in.set_xticks([])
        ax_in.set_yticks([])
        ax_in.set_box_aspect(1)
        for spine in ax_in.spines.values():
            spine.set_visible(True)
            spine.set_color("0.35")
            spine.set_linewidth(0.8)
        ax_in.set_xlabel(f"m = {m}", fontsize=13.5, labelpad=5, fontname="Arial")
        if col == 0:
            ax_in.set_ylabel("SLM input", fontsize=16, fontname="Arial")

        rgb = _crop_bright_spot_rgb(frames[col], side=shared_side, out_side=512)
        ax_out.imshow(rgb, aspect="equal", interpolation="nearest")
        ax_out.set_xticks([])
        ax_out.set_yticks([])
        ax_out.set_box_aspect(1)
        for spine in ax_out.spines.values():
            spine.set_visible(True)
            spine.set_color("0.35")
            spine.set_linewidth(0.8)
        if col == 0:
            ax_out.set_ylabel("Green output", fontsize=16, fontname="Arial")

    for ax in axes.ravel():
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname("Arial")
        if ax.xaxis.label is not None:
            ax.xaxis.label.set_fontname("Arial")
        if ax.yaxis.label is not None:
            ax.yaxis.label.set_fontname("Arial")

    fig.subplots_adjust(wspace=0.06, hspace=0.14)
    _save_figure(fig, output_dir / "fig3b_input_output")


def plot_fig3c(cache: dict[str, Any], output_dir: Path) -> None:
    """Raw vs G (≡ detail_cm) challenge NCC matrices for m=2 (Nature / Light).

    Left: warm sequential colormap on the data range (high-NCC resolution).
    Right: muted blue–terracotta diverging map with TwoSlopeNorm centered at 0.
    """
    _apply_nature_style()
    m = MATRIX_MACRO
    ids = cache["challenge_ids"]
    labels = _challenge_labels(ids)
    raw = cache["by_macro"][m]["raw_matrix"]
    cm = cache["by_macro"][m]["detail_cm_matrix"]
    if raw is None or cm is None:
        raise RuntimeError("Missing NCC matrices in cache; rebuild with --force")
    raw = np.asarray(raw, dtype=float)
    cm = np.asarray(cm, dtype=float)

    # Display ranges from actual data (padded to clean ticks).
    raw_vmin, raw_vmax = _pad_lim(float(np.nanmin(raw)), float(np.nanmax(raw)), step=0.02, pad=0.005)
    raw_vmax = min(raw_vmax, 1.0)  # NCC cannot exceed 1
    if raw_vmax <= raw_vmin:
        raw_vmin = max(0.0, raw_vmax - 0.02)
    g_vmin, g_vmax = _pad_lim(float(np.nanmin(cm)), float(np.nanmax(cm)), step=0.1, pad=0.02)
    # Keep a readable positive end for G even if max is exactly 1.
    g_vmax = max(g_vmax, 1.0)
    g_vmin = min(g_vmin, -0.1)  # ensure negative side exists for TwoSlopeNorm

    raw_norm = Normalize(vmin=raw_vmin, vmax=raw_vmax)
    g_norm = TwoSlopeNorm(vcenter=0.0, vmin=g_vmin, vmax=g_vmax)

    fig = plt.figure(figsize=(15.0, 6.8), facecolor=C_FACE)
    # left matrix | slim cbar | gap | right matrix | slim cbar
    gs = fig.add_gridspec(
        1,
        5,
        width_ratios=[1.0, 0.045, 0.20, 1.0, 0.045],
        wspace=0.32,
        left=0.08,
        right=0.96,
        top=0.82,
        bottom=0.18,
    )
    ax_raw = fig.add_subplot(gs[0, 0])
    cax_raw = fig.add_subplot(gs[0, 1])
    ax_g = fig.add_subplot(gs[0, 3])
    cax_g = fig.add_subplot(gs[0, 4])

    # Paper notation: detail_cm ≡ \tilde{G}_{i,c,b} after common subtraction.
    panels = [
        (ax_raw, cax_raw, raw, RAW_CMAP, raw_norm, rf"Raw  ($m = {m}$)"),
        (ax_g, cax_g, cm, G_CMAP, g_norm, rf"$\tilde{{G}}_{{i,c}}$  ($m = {m}$)"),
    ]

    n = len(labels)
    for ax, cax, mat, cmap, norm, name in panels:
        ax.set_facecolor(C_FACE)
        im = ax.imshow(mat, cmap=cmap, norm=norm, interpolation="nearest")

        # Fine white/light separators between cells.
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color=C_GRID, linewidth=1.2)
        ax.tick_params(which="minor", length=0)

        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(
            labels,
            rotation=45,
            ha="right",
            fontsize=22,
            fontname="Arial",
            color=C_TEXT,
        )
        ax.set_yticklabels(
            labels,
            fontsize=22,
            fontname="Arial",
            color=C_TEXT,
        )
        ax.tick_params(length=0, colors=C_TEXT, pad=5)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.7)
            spine.set_color(C_SPINE)

        ax.text(
            0.5,
            1.06,
            name,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=30,
            fontweight="normal",
            fontname="Arial",
            color=C_TEXT,
        )

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = float(mat[i, j])
                rgba = cmap(norm(val))
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=15,
                    fontname="Arial",
                    color=_annotation_text_color(rgba),
                )

        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(
            "NCC",
            fontsize=20,
            labelpad=14,
            color=C_TEXT,
            fontname="Arial",
        )
        cbar.ax.tick_params(labelsize=18, pad=5, colors=C_TEXT, length=3.0, width=0.7)
        cbar.outline.set_edgecolor(C_SPINE)
        cbar.outline.set_linewidth(0.7)
        for t in cbar.ax.get_yticklabels():
            t.set_color(C_TEXT)
            t.set_fontname("Arial")
            t.set_fontsize(18)

    # Right panel: hide redundant y tick labels to reduce clutter.
    ax_g.set_yticklabels([])

    stem = output_dir / "fig3c_ncc_matrices"
    ensure_dir(stem.parent)
    for ext in (".png", ".pdf", ".svg"):
        kwargs = dict(bbox_inches="tight", pad_inches=0.18, facecolor=C_FACE)
        if ext == ".png":
            kwargs["dpi"] = 400
        fig.savefig(stem.with_suffix(ext), **kwargs)
    plt.close(fig)


def plot_fig3d(cache: dict[str, Any], output_dir: Path) -> None:
    """Intra vs inter detail_cm NCC distributions across macro sizes."""
    _apply_nature_style()
    macros = ALL_MACROS
    fig, ax = plt.subplots(figsize=(6.2, 6.2))

    # Highlight selected m=2 band.
    m2_idx = macros.index(2)
    ax.axvspan(m2_idx - 0.45, m2_idx + 0.45, color=ACCENT_M2, alpha=0.18, zorder=0)

    positions_intra = []
    positions_inter = []
    data_intra = []
    data_inter = []
    for i, m in enumerate(macros):
        positions_intra.append(i - 0.18)
        positions_inter.append(i + 0.18)
        data_intra.append(cache["by_macro"][m]["intra_detail_cm"])
        data_inter.append(cache["by_macro"][m]["inter_detail_cm"])

    vp_intra = ax.violinplot(
        data_intra, positions=positions_intra, widths=0.32,
        showmeans=False, showmedians=False, showextrema=False,
    )
    vp_inter = ax.violinplot(
        data_inter, positions=positions_inter, widths=0.32,
        showmeans=False, showmedians=False, showextrema=False,
    )
    for body in vp_intra["bodies"]:
        body.set_facecolor(INTRA_COLOR)
        body.set_alpha(0.35)
        body.set_edgecolor(INTRA_COLOR)
    for body in vp_inter["bodies"]:
        body.set_facecolor(INTER_COLOR)
        body.set_alpha(0.35)
        body.set_edgecolor(INTER_COLOR)

    bp_intra = ax.boxplot(
        data_intra, positions=positions_intra, widths=0.16,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=1.6),
        whiskerprops=dict(color="0.25", linewidth=1.1),
        capprops=dict(color="0.25", linewidth=1.1),
        boxprops=dict(linewidth=1.1),
        zorder=3,
    )
    bp_inter = ax.boxplot(
        data_inter, positions=positions_inter, widths=0.16,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=1.6),
        whiskerprops=dict(color="0.25", linewidth=1.1),
        capprops=dict(color="0.25", linewidth=1.1),
        boxprops=dict(linewidth=1.1),
        zorder=3,
    )
    for patch in bp_intra["boxes"]:
        patch.set_facecolor(INTRA_COLOR)
        patch.set_alpha(0.55)
    for patch in bp_inter["boxes"]:
        patch.set_facecolor(INTER_COLOR)
        patch.set_alpha(0.55)

    rng = np.random.default_rng(42)
    # Scatter: darker than violin fills + thin white edge for contrast on fill.
    scatter_intra = "#1B263B"
    scatter_inter = "#7A4E00"
    for pos, vals in zip(positions_intra, data_intra):
        jitter = rng.uniform(-0.06, 0.06, size=len(vals))
        ax.scatter(
            np.full(len(vals), pos) + jitter, vals,
            s=12, color=scatter_intra, alpha=0.55,
            edgecolors="white", linewidths=0.35, zorder=2,
        )
    for pos, vals in zip(positions_inter, data_inter):
        jitter = rng.uniform(-0.06, 0.06, size=len(vals))
        ax.scatter(
            np.full(len(vals), pos) + jitter, vals,
            s=12, color=scatter_inter, alpha=0.55,
            edgecolors="white", linewidths=0.35, zorder=2,
        )

    ax.axhline(0.0, color="0.7", linewidth=1.0, linestyle="--", zorder=1)
    ax.set_xticks(np.arange(len(macros)))
    ax.set_xticklabels([str(m) for m in macros], fontname="Arial")
    ax.set_xlabel("Macro-pixel size m", fontname="Arial")
    ax.set_ylabel("NCC", fontname="Arial")
    ax.set_ylim(-0.55, 1.05)
    ax.set_box_aspect(1)
    ax.legend(
        handles=[
            mpl.patches.Patch(facecolor=INTRA_COLOR, alpha=0.7, label="Genuine score"),
            mpl.patches.Patch(facecolor=INTER_COLOR, alpha=0.7, label="Challenge-mismatch score"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        columnspacing=1.4,
        handlelength=1.2,
        prop={"family": "Arial", "size": 13},
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save_figure(fig, output_dir / "fig3d_genuine_challenge_distributions")


def plot_fig3e(cache: dict[str, Any], output_dir: Path) -> None:
    """Robust quantile intervals (top) with the derived gap curve G(m) (bottom).

    G(m) = Q05(Genuine) - Q95(Challenge-mismatch) is exactly the length of each
    interval above, so both panels share the gap colour to tie them together.
    """
    _apply_nature_style()
    macros = ALL_MACROS
    q05 = np.array([cache["by_macro"][m]["intra_p05"] for m in macros], dtype=float)
    q95 = np.array([cache["by_macro"][m]["inter_p95"] for m in macros], dtype=float)
    gap = q05 - q95
    x = np.arange(len(macros), dtype=float)
    m2_idx = macros.index(2)
    gap_max = float(np.max(gap))
    tie_level = TIE_TOLERANCE * gap_max

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(6.6, 7.6),
        sharex=True,
        gridspec_kw={"height_ratios": [2.15, 1.0], "hspace": 0.12},
    )

    for ax in (ax_top, ax_bot):
        ax.axvspan(m2_idx - 0.42, m2_idx + 0.42, color=SELECT_BAND, zorder=0)

    for i, (lo, hi) in enumerate(zip(q95, q05)):
        ax_top.plot([i, i], [lo, hi], color=GAP_COLOR, linewidth=8.0, alpha=0.22,
                    solid_capstyle="round", zorder=1)
        ax_top.plot([i, i], [lo, hi], color=GAP_COLOR, linewidth=1.6, alpha=0.85, zorder=2)

    ax_top.scatter(
        x, q05, s=88, color=INTRA_COLOR, edgecolors="white", linewidths=1.2,
        zorder=4, label=r"$Q_{5\%}(\mathrm{Genuine})$",
    )
    ax_top.scatter(
        x, q95, s=88, color=INTER_COLOR, edgecolors="white", linewidths=1.2,
        zorder=4, label=r"$Q_{95\%}(\mathrm{Challenge\text{-}mismatch})$",
    )
    ax_top.axhline(0.0, color="0.72", linewidth=1.0, linestyle=(0, (4, 4)), zorder=1)
    ax_top.set_ylabel("NCC", fontname="Arial")
    ax_top.set_ylim(-0.08, 0.86)
    ax_top.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax_top.tick_params(axis="x", length=0)
    handles, labels = ax_top.get_legend_handles_labels()
    handles.append(
        mpl.lines.Line2D([], [], color=GAP_COLOR, linewidth=6.0, alpha=0.45,
                         solid_capstyle="butt")
    )
    labels.append(r"$G(m)=Q_{5\%}-Q_{95\%}$")
    ax_top.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        columnspacing=1.2,
        handlelength=1.2,
        handletextpad=0.6,
        labelspacing=0.55,
        prop={"family": "Arial", "size": 12.5},
    )

    ax_bot.fill_between(x, 0.0, gap, color=GAP_COLOR, alpha=0.16, zorder=1)
    ax_bot.plot(x, gap, color=GAP_COLOR, linewidth=2.2, zorder=3)
    ax_bot.scatter(
        x, gap, s=62, color=GAP_COLOR, edgecolors="white", linewidths=1.2, zorder=4
    )
    ax_bot.axhline(
        tie_level, color="0.55", linewidth=1.1, linestyle=(0, (5, 4)), zorder=2
    )
    ax_bot.text(
        len(macros) - 0.55,
        tie_level + 0.022,
        rf"${TIE_TOLERANCE:.2f}\,G_{{\mathrm{{max}}}}$",
        ha="right", va="bottom", fontsize=12.5, color="0.40", fontname="Arial",
    )
    # Ring the operating point: smallest m still inside the tie band.
    ax_bot.scatter(
        [m2_idx], [gap[m2_idx]], s=215, facecolors="none",
        edgecolors=C_ANNOT_DARK, linewidths=1.6, zorder=5,
    )
    ax_bot.annotate(
        "$m = 2$ selected",
        xy=(m2_idx, gap[m2_idx]),
        xytext=(m2_idx + 0.30, gap[m2_idx] + 0.150),
        ha="left", va="bottom", fontsize=13.5, color=C_ANNOT_DARK, fontname="Arial",
        arrowprops=dict(arrowstyle="-", color=C_ANNOT_DARK, linewidth=1.0,
                        shrinkA=2, shrinkB=9),
    )
    ax_bot.set_ylabel(r"$G(m)$", fontname="Arial")
    ax_bot.set_ylim(0.0, 0.80)
    ax_bot.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax_bot.set_xlabel("Macro-pixel size m", fontname="Arial")

    ax_bot.set_xlim(-0.55, len(macros) - 0.45)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels([str(m) for m in macros], fontname="Arial")

    fig.align_ylabels([ax_top, ax_bot])
    fig.subplots_adjust(left=0.16, right=0.98, top=0.90, bottom=0.09, hspace=0.12)
    _save_figure(fig, output_dir / "fig3e_robust_gap_intervals")


def generate_fig3_bcde(
    config_path: Path,
    *,
    force_cache: bool = False,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    layout = PathLayout(cfg.root)
    out_dir = layout.screening / "figures"
    ensure_dir(out_dir)

    cache = build_or_load_fig_cache(cfg, force=force_cache)
    plot_fig3b(cfg, cache, out_dir)
    plot_fig3c(cache, out_dir)
    plot_fig3d(cache, out_dir)
    plot_fig3e(cache, out_dir)
    return {
        "output_dir": str(out_dir),
        "generated": [
            "fig3b_input_output",
            "fig3c_ncc_matrices",
            "fig3d_genuine_challenge_distributions",
            "fig3e_robust_gap_intervals",
        ],
    }
