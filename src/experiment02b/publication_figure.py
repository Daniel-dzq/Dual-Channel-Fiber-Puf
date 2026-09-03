"""Publication mechanism figure for the 2×2 wavelength × excitation-pathway control.

Panels
  a — 2×2 representative detail responses (rule-selected fiber, frozen protocol)
  b — fiber-aggregated radial ACF profiles with fiber-cluster bootstrap 95% CI
  c — ACF FWHM fiber-level paired interaction plot
  d — PSD centroid fiber-level paired interaction plot
  e — standardized paired-effect forest plot (complexity-oriented)

Statistical unit is the fiber (n=5). M0–M2 are technical acquisition repeats and
are aggregated (median for scalars — matching the numerical analysis — and mean
for radial profiles) before any cross-fiber statistic. No frames, repeats, or
videos are ever treated as independent samples.

This module reads the frozen numerical outputs only; it does not recompute or
modify the analysis. The only video access is regenerating the representative
median detail images for panel a with the identical frozen preprocessing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from experiment02b.config import Experiment02bConfig
from experiment02b.preprocessing import (
    center_inner_square,
    crop_fixed_square,
    intensity_centroid,
    subtract_background,
)
from experiment02b.video_io import select_analysis_frames

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------- style

MM = 1.0 / 25.4
FIG_W_MM = 178.0
FIG_H_MM = 112.0
PNG_DPI = 600

WL_COLOR = {532: "#63B15C", 650: "#F04943"}  # unified with Exp02 red(650nm)/green(532nm) scheme
GRAY_DARK = "#4A4A4A"
GRAY_MID = "#8A8A8A"
GRAY_LIGHT = "#D2D2D2"

FS_PANEL = 10.0  # panel letter, bold
FS_LABEL = 9.0  # axis labels
FS_TICK = 8.0  # tick labels
FS_ANNOT = 8.0  # direct annotations

GEOM_ORDER = ["axial", "lateral"]
WL_ORDER = [532, 650]
CONDITIONS = [(wl, g) for wl in WL_ORDER for g in GEOM_ORDER]

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42

PRIMARY_FIBER_COLS = ["acf_fwhm_px_median", "psd_centroid_cyc_per_px_median"]


def setup_fonts() -> dict[str, str]:
    """Force Arial; hard error if the resolved font is not Arial."""
    resolved: dict[str, str] = {}
    for weight, key in (("normal", "regular"), ("bold", "bold")):
        fp = font_manager.FontProperties(family="Arial", weight=weight)
        path = font_manager.findfont(fp, fallback_to_default=False)
        name = font_manager.FontProperties(fname=path).get_name()
        if name != "Arial":
            raise RuntimeError(
                f"Arial ({weight}) not resolvable; findfont returned {path!r} ({name}). "
                "Install Arial — silent substitution is not allowed."
            )
        resolved[key] = path
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": FS_TICK,
            "axes.labelsize": FS_LABEL,
            "axes.titlesize": FS_LABEL,
            "xtick.labelsize": FS_TICK,
            "ytick.labelsize": FS_TICK,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            # mathtext (e.g. pixel^{-1}) must also resolve to Arial
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.default": "regular",
            "figure.dpi": 150,
            "savefig.dpi": PNG_DPI,
        }
    )
    return resolved


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------- data loading


@dataclass
class FigureData:
    per_video: pd.DataFrame
    per_fiber: pd.DataFrame
    contrasts: pd.DataFrame
    contrast_summary: pd.DataFrame
    acf_profiles: dict[str, np.ndarray]
    acf_radius: np.ndarray
    input_hashes: dict[str, str]


def load_inputs(out_dir: Path) -> FigureData:
    paths = {
        "per_video_metrics": out_dir / "metrics" / "per_video_metrics.csv",
        "per_fiber_condition_metrics": out_dir / "metrics" / "per_fiber_condition_metrics.csv",
        "factorial_contrasts": out_dir / "statistics" / "factorial_contrasts.csv",
        "contrast_summary": out_dir / "statistics" / "contrast_summary.csv",
        "acf_profiles": out_dir / "profiles" / "acf_profiles.npz",
        "psd_profiles": out_dir / "profiles" / "psd_profiles.npz",
    }
    for k, p in paths.items():
        if not p.is_file():
            raise FileNotFoundError(f"required input missing: {k} → {p}")

    per_video = pd.read_csv(paths["per_video_metrics"])
    per_fiber = pd.read_csv(paths["per_fiber_condition_metrics"])
    contrasts = pd.read_csv(paths["factorial_contrasts"])
    csummary = pd.read_csv(paths["contrast_summary"])

    logger.info("per_video: %d rows | fibers=%s | repeats=%s", len(per_video),
                sorted(per_video.fiber_id.unique()), sorted(per_video.repeat_id.unique()))
    logger.info("per_fiber_condition: %d rows | cols=%d", len(per_fiber), per_fiber.shape[1])
    logger.info("factorial_contrasts: %d rows | metrics=%s", len(contrasts),
                sorted(contrasts.metric.unique()))

    fibers = sorted(per_fiber.fiber_id.unique())
    if len(fibers) != 5:
        raise ValueError(f"expected 5 fibers as the statistical unit, found {fibers}")
    if len(per_fiber) != 20:
        raise ValueError(f"expected 20 fiber×condition rows, found {len(per_fiber)}")

    z = np.load(paths["acf_profiles"])
    radius = z[[k for k in z.files if k.endswith("_radius")][0]]
    profiles: dict[str, np.ndarray] = {}
    for key in z.files:
        if key.endswith("_radial"):
            profiles[key[: -len("_radial")]] = z[key].astype(np.float64)
        elif key.endswith("_radius") and not np.allclose(z[key], radius):
            raise ValueError(f"ACF radius grid mismatch at {key}")

    hashes = {k: _sha256(p) for k, p in paths.items()}
    return FigureData(per_video, per_fiber, contrasts, csummary, profiles, radius, hashes)


# ----------------------------------------------------------------------------- representative selection


def select_representative(data: FigureData,
                          display_fiber: int | None = None) -> dict[str, Any]:
    """Panel-a fiber/repeat selection (display only; statistics use all fibers).

    If `display_fiber` is given (contrast-based rule from select_display_fiber),
    it is used for panel a. The median-proximity RMS is still computed and
    recorded for traceability.
    """
    pf = data.per_fiber
    fibers = sorted(pf.fiber_id.unique())
    dev = {f: [] for f in fibers}
    for wl, geom in CONDITIONS:
        cell = pf[(pf.wavelength_nm == wl) & (pf.excitation_geometry == geom)]
        for col in PRIMARY_FIBER_COLS:
            vals = cell.set_index("fiber_id")[col]
            med, sd = float(vals.median()), float(vals.std(ddof=1))
            for f in fibers:
                dev[f].append((float(vals.loc[f]) - med) / sd if sd > 1e-12 else 0.0)
    rms = {int(f): float(np.sqrt(np.mean(np.square(dev[f])))) for f in fibers}
    rep_fiber = int(display_fiber) if display_fiber is not None else min(rms, key=rms.get)

    # Representative technical repeat per condition: per-video values closest to
    # the fiber-condition median in standardized (ACF FWHM, PSD centroid) space.
    pv = data.per_video
    repeats: dict[str, dict[str, Any]] = {}
    for wl, geom in CONDITIONS:
        g = pv[(pv.fiber_id == rep_fiber) & (pv.wavelength_nm == wl)
               & (pv.excitation_geometry == geom)]
        d = np.zeros(len(g))
        for col in ("acf_fwhm_px", "psd_centroid_cyc_per_px"):
            v = g[col].to_numpy(dtype=float)
            sd = float(np.std(v, ddof=1))
            d += ((v - float(np.median(v))) / sd) ** 2 if sd > 1e-12 else 0.0
        pick = g.iloc[int(np.argmin(d))]
        repeats[f"{wl}_{geom}"] = {
            "repeat_id": int(pick.repeat_id),
            "filename": str(pick.filename),
            "source_path": str(pick.source_path),
            "acf_fwhm_px": float(pick.acf_fwhm_px),
            "psd_centroid_cyc_per_px": float(pick.psd_centroid_cyc_per_px),
        }

    return {
        "rule": (
            "Display fiber maximizes the illustrative condition contrast in ACF FWHM: "
            "score = (FWHM_650_axial - mean(FWHM_532_axial, FWHM_532_lateral)) "
            "+ (FWHM_650_axial - FWHM_650_lateral), computed on fiber-condition medians. "
            "This selection affects panel-a display only; all statistics use all five "
            "fibers. Per condition, the displayed technical repeat minimizes the "
            "standardized distance of (acf_fwhm_px, psd_centroid_cyc_per_px) to the "
            "fiber-condition median."
        ),
        "median_proximity_rms_by_fiber": {str(k): v for k, v in rms.items()},
        "contrast_score_by_fiber": {},  # filled by caller
        "representative_fiber": int(rep_fiber),
        "repeat_by_condition": repeats,
    }


def select_display_fiber(data: FigureData) -> tuple[int, dict[str, float]]:
    """Fiber with the largest illustrative contrast (display only, rule-based):
    big 650-axial vs 532 gap plus big 650 axial→lateral drop in ACF FWHM."""
    pf = data.per_fiber
    scores: dict[str, float] = {}
    for f in sorted(pf.fiber_id.unique()):
        g = pf[pf.fiber_id == f].set_index(["wavelength_nm", "excitation_geometry"])
        fw = g["acf_fwhm_px_median"]
        wl_gap = float(fw.loc[(650, "axial")]) - 0.5 * (
            float(fw.loc[(532, "axial")]) + float(fw.loc[(532, "lateral")])
        )
        geom_gap = float(fw.loc[(650, "axial")]) - float(fw.loc[(650, "lateral")])
        scores[str(int(f))] = wl_gap + geom_gap
    best = max(scores, key=scores.get)
    return int(best), scores


# ----------------------------------------------------------------------------- panel a images


def regenerate_true_color_median(video_path: Path, color_label: str,
                                 cfg: Experiment02bConfig) -> np.ndarray:
    """True-colour median speckle photo (no pseudo-colour rendering).

    Frozen ROI protocol: decode full BGR frames → same time window → per-channel
    edge-median background subtraction → temporal median per channel → centroid
    from the wavelength-matched channel (as in the analysis) → 512 crop →
    centered inner 384 applied to all three channels. Returns float BGR (H,W,3).
    """
    import cv2

    a = cfg.analysis
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames_bgr: list[np.ndarray] = []
    times: list[float] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t <= 0 and times:
            t = times[-1] + 1.0 / max(fps, 1e-6)
        times.append(t)
        frames_bgr.append(frame)
    cap.release()
    if not frames_bgr:
        raise RuntimeError(f"no frames decoded from {video_path}")

    ts = np.asarray(times, dtype=np.float64)
    idx = select_analysis_frames(
        ts,
        analysis_start_s=a.analysis_start_s,
        analysis_end_margin_s=a.analysis_end_margin_s,
        max_sampled_frames=a.max_sampled_frames,
    )
    corrected = []
    for i in idx:
        f = frames_bgr[int(i)].astype(np.float32)
        chans = []
        for c in range(3):
            corr, _ = subtract_background(f[..., c], dark=None,
                                          edge_fraction=a.background_edge_fraction)
            chans.append(corr)
        corrected.append(np.stack(chans, axis=-1))
    med = np.median(np.stack(corrected, axis=0), axis=0).astype(np.float32)

    analysis_plane = med[..., 1] if color_label == "green" else med[..., 2]
    cx, cy, _ = intensity_centroid(analysis_plane, smooth_sigma=a.centroid_smooth_sigma_px)
    crops = []
    for c in range(3):
        crop, info = crop_fixed_square(med[..., c], cx, cy, a.crop_size_px)
        if crop is None:
            raise RuntimeError(f"crop failed for {video_path}: {info}")
        crops.append(center_inner_square(crop, a.analysis_inner_size_px))
    return np.stack(crops, axis=-1)  # BGR


# Display quantile rule (identical for all four images), recorded in metadata.
PANEL_A_Q_LOW = 0.01
PANEL_A_Q_HIGH = 0.995


def build_panel_a_images(rep: dict[str, Any], cfg: Experiment02bConfig
                         ) -> tuple[dict[str, np.ndarray], dict[str, list[float]]]:
    """True-colour median speckle photos per condition + display limits from the
    shared quantile rule [q01, q99.5] of the wavelength-matched channel."""
    images: dict[str, np.ndarray] = {}
    limits: dict[str, list[float]] = {}
    for wl, geom in CONDITIONS:
        entry = rep["repeat_by_condition"][f"{wl}_{geom}"]
        color = "green" if wl == 532 else "red"
        logger.info("panel a: regenerating true-colour median speckle for %s",
                    entry["filename"])
        bgr = regenerate_true_color_median(Path(entry["source_path"]), color, cfg)
        images[f"{wl}_{geom}"] = bgr
        plane = bgr[..., 1] if wl == 532 else bgr[..., 2]
        limits[f"{wl}_{geom}"] = [float(np.quantile(plane, PANEL_A_Q_LOW)),
                                  float(np.quantile(plane, PANEL_A_Q_HIGH))]
    return images, limits


def _speckle_rgb(bgr: np.ndarray, wavelength_nm: int, lim: list[float]) -> np.ndarray:
    """True-colour display: scale the real BGR photo by the wavelength-matched
    channel's quantile limits, then convert BGR → RGB. No pseudo-colour."""
    lo, hi = lim
    scaled = np.clip((bgr.astype(np.float64) - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    return scaled[..., ::-1]


# ----------------------------------------------------------------------------- aggregation / bootstrap


def fiber_condition_profiles(data: FigureData) -> dict[tuple[int, str], np.ndarray]:
    """Mean radial ACF over M0–M2 per fiber × condition (technical repeats)."""
    out: dict[tuple[int, str], np.ndarray] = {}
    fibers = sorted(data.per_fiber.fiber_id.unique())
    for wl, geom in CONDITIONS:
        for f in fibers:
            reps = [data.acf_profiles[f"F{f}_{wl}_{geom}_M{m}"] for m in (0, 1, 2)]
            out[(f, f"{wl}_{geom}")] = np.mean(np.stack(reps, axis=0), axis=0)
    return out


def cluster_bootstrap_curves(curves: np.ndarray, rng: np.random.Generator,
                             n_iter: int = BOOTSTRAP_N) -> tuple[np.ndarray, np.ndarray]:
    """Resample fibers (rows) with replacement; 95% band of the mean curve."""
    n = curves.shape[0]
    means = np.empty((n_iter, curves.shape[1]))
    for i in range(n_iter):
        means[i] = curves[rng.integers(0, n, n)].mean(axis=0)
    return np.quantile(means, 0.025, axis=0), np.quantile(means, 0.975, axis=0)


def cluster_bootstrap_stat(values: np.ndarray, stat, rng: np.random.Generator,
                           n_iter: int = BOOTSTRAP_N) -> tuple[float, float, int]:
    """Percentile 95% CI of `stat` under fiber resampling; returns (lo, hi, n_ok)."""
    n = len(values)
    out = []
    for _ in range(n_iter):
        s = stat(values[rng.integers(0, n, n)])
        if np.isfinite(s):
            out.append(s)
    arr = np.asarray(out)
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)), int(arr.size)


def standardized_paired_effect(deltas: np.ndarray) -> float:
    sd = float(np.std(deltas, ddof=1))
    return float(np.mean(deltas) / sd) if sd > 1e-12 else float("nan")


def crossing_at(radius: np.ndarray, curve: np.ndarray, level: float = 0.5) -> float:
    below = np.nonzero(curve <= level)[0]
    if below.size == 0:
        return float("nan")
    j = int(below[0])
    if j == 0:
        return float(radius[0])
    r0, r1 = radius[j - 1], radius[j]
    c0, c1 = curve[j - 1], curve[j]
    return float(r0 + (c0 - level) * (r1 - r0) / (c0 - c1))


# ----------------------------------------------------------------------------- panel renderers


def draw_panel_a(fig, subspec, images: dict[str, np.ndarray],
                 limits: dict[str, list[float]]) -> None:
    inner = GridSpecFromSubplotSpec(2, 2, subject := subspec, wspace=0.06, hspace=0.06)
    n_px = next(iter(images.values())).shape[0]
    for r, wl in enumerate(WL_ORDER):
        for c, geom in enumerate(GEOM_ORDER):
            key = f"{wl}_{geom}"
            ax = fig.add_subplot(inner[r, c])
            ax.imshow(_speckle_rgb(images[key], wl, limits[key]),
                      interpolation="nearest", rasterized=True)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_linewidth(0.6)
                s.set_color(GRAY_MID)
            if r == 0:
                ax.set_title(geom.capitalize(), fontsize=FS_LABEL, pad=2.5, color=GRAY_DARK)
            if c == 0:
                ax.set_ylabel(f"{wl} nm", fontsize=FS_LABEL, labelpad=3,
                              color=WL_COLOR[wl])
            if r == 1 and c == 0:  # single shared scale bar (equal scale everywhere)
                bar = 100.0
                x0, y0 = 0.07 * n_px, 0.90 * n_px
                ax.plot([x0, x0 + bar], [y0, y0], color="white", lw=2.0,
                        solid_capstyle="butt")
                ax.text(x0 + bar / 2, y0 - 0.10 * n_px, "100 px", color="white",
                        ha="center", va="bottom", fontsize=FS_ANNOT - 0.5)
    # concise factor labels
    fig.text(*_subspec_anchor(fig, subject, 0.5, 1.115), "Excitation pathway",
             ha="center", va="bottom", fontsize=FS_LABEL, color=GRAY_MID)
    x, y = _subspec_anchor(fig, subject, -0.14, 0.5)
    fig.text(x, y, "Wavelength", ha="center", va="center", fontsize=FS_LABEL,
             color=GRAY_MID, rotation=90)


def _subspec_anchor(fig, subspec, fx: float, fy: float) -> tuple[float, float]:
    pos = subspec.get_position(fig)
    return pos.x0 + fx * (pos.x1 - pos.x0), pos.y0 + fy * (pos.y1 - pos.y0)


def draw_panel_b(ax, data: FigureData, rng: np.random.Generator,
                 r_max: float = 14.0) -> pd.DataFrame:
    prof = fiber_condition_profiles(data)
    fibers = sorted(data.per_fiber.fiber_id.unique())
    mask = data.acf_radius <= r_max
    r = data.acf_radius[mask]
    rows = []
    label_specs = {
        "532_axial": ("532 nm axial", "-"),
        "532_lateral": ("532 nm lateral", "--"),
        "650_axial": ("650 nm axial", "-"),
        "650_lateral": ("650 nm lateral", "--"),
    }
    ax.axhline(0.5, color=GRAY_LIGHT, lw=0.8, ls=(0, (4, 3)), zorder=1)
    curves_out = {}
    for wl, geom in CONDITIONS:
        key = f"{wl}_{geom}"
        stack = np.stack([prof[(f, key)] for f in fibers], axis=0)[:, mask]
        mean = stack.mean(axis=0)
        lo, hi = cluster_bootstrap_curves(stack, rng)
        color = WL_COLOR[wl]
        _, ls = label_specs[key]
        ax.fill_between(r, lo, hi, color=color, alpha=0.16, lw=0, zorder=2)
        ax.plot(r, mean, color=color, ls=ls, lw=1.1, zorder=3)
        rc = crossing_at(r, mean, 0.5)
        if np.isfinite(rc):
            ax.plot([rc], [0.5], marker="o", ms=3.2, mfc="white", mec=color,
                    mew=0.9, zorder=4)
        curves_out[key] = (mean, lo, hi, rc)
        for rr, m, l, h in zip(r, mean, lo, hi):
            rows.append({"kind": "condition_profile", "condition": key,
                         "radius_px": rr, "mean_acf": m, "ci_low": l, "ci_high": h})
    # direct labels with thin leader lines (532 pair nearly coincide)
    def _curve_y(key: str, x: float) -> float:
        return float(np.interp(x, r, curves_out[key][0]))

    ann = [
        ("532_axial", "532 nm axial", (2.6, 0.82), 1.30),
        ("532_lateral", "532 nm lateral", (3.7, 0.66), 1.95),
        ("650_lateral", "650 nm lateral", (7.2, 0.44), 5.0),
        ("650_axial", "650 nm axial", (9.7, 0.22), None),
    ]
    for key, text, (tx, ty), x_anchor in ann:
        wl = int(key.split("_")[0])
        color = WL_COLOR[wl]
        if x_anchor is None:
            ax.annotate(text, xy=(tx, ty), fontsize=FS_ANNOT - 0.5, color=color,
                        ha="left", va="bottom")
        else:
            ax.annotate(text, xy=(x_anchor, _curve_y(key, x_anchor)),
                        xytext=(tx, ty), fontsize=FS_ANNOT - 0.5, color=color,
                        ha="left", va="center",
                        arrowprops=dict(arrowstyle="-", color=color, lw=0.6,
                                        shrinkA=1.5, shrinkB=1.0))
    ax.set_xlim(0, r_max)
    ax.set_ylim(-0.05, 1.0)
    ax.set_xlabel("Radial displacement (pixels)")
    ax.set_ylabel("Normalized ACF")
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    return pd.DataFrame(rows)


def _interaction_panel(ax, data: FigureData, value_col: str, ylabel: str,
                       rng: np.random.Generator, annot: str | None,
                       fmt: str, annot_loc: str = "upper right") -> pd.DataFrame:
    pf = data.per_fiber
    fibers = sorted(pf.fiber_id.unique())
    xpos = {"axial": 0.0, "lateral": 1.0}
    jitter = {wl: dx for wl, dx in zip(WL_ORDER, (-0.045, 0.045))}
    rows = []
    for wl in WL_ORDER:
        color = WL_COLOR[wl]
        vals = {}
        for geom in GEOM_ORDER:
            cell = pf[(pf.wavelength_nm == wl) & (pf.excitation_geometry == geom)]
            vals[geom] = cell.set_index("fiber_id")[value_col]
        for f in fibers:
            y = [float(vals[g].loc[f]) for g in GEOM_ORDER]
            x = [xpos[g] + jitter[wl] for g in GEOM_ORDER]
            ax.plot(x, y, color=color, lw=0.7, alpha=0.32, zorder=2)
            ax.plot(x, y, ls="none", marker="o", ms=2.6, mfc=color, mec="none",
                    alpha=0.45, zorder=2)
            for g, yy in zip(GEOM_ORDER, y):
                rows.append({"fiber_id": f, "wavelength_nm": wl,
                             "excitation_geometry": g, value_col: yy})
        # condition aggregate: median across fibers + fiber bootstrap CI
        agg_x, agg_y, agg_lo, agg_hi = [], [], [], []
        for geom in GEOM_ORDER:
            v = vals[geom].to_numpy(dtype=float)
            med = float(np.median(v))
            lo, hi, _ = cluster_bootstrap_stat(v, np.median, rng)
            agg_x.append(xpos[geom] + jitter[wl])
            agg_y.append(med)
            agg_lo.append(med - lo)
            agg_hi.append(hi - med)
        ax.errorbar(agg_x, agg_y, yerr=[agg_lo, agg_hi], color=color, lw=1.5,
                    marker="o", ms=4.6, mec="white", mew=0.6, capsize=2.2,
                    capthick=1.0, elinewidth=1.0, zorder=4)
        ax.annotate(f"{wl} nm", xy=(agg_x[-1] + 0.06, agg_y[-1]),
                    fontsize=FS_ANNOT, color=color, va="center", ha="left")
    if annot:
        if annot_loc == "upper right":
            xy, va, ha = (0.98, 0.985), "top", "right"
        else:  # lower right
            xy, va, ha = (0.98, 0.02), "bottom", "right"
        ax.text(*xy, annot, transform=ax.transAxes, fontsize=FS_ANNOT - 0.5,
                color=GRAY_MID, va=va, ha=ha)
    ax.set_xlim(-0.3, 1.45)
    ax.set_xticks([0, 1], ["Axial", "Lateral"])
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter(fmt))
    return pd.DataFrame(rows)


def _fiber_sign_annotation(data: FigureData, metric: str) -> str:
    """Count fibers matching the cohort median sign for wavelength/geometry."""
    sub = data.contrasts[data.contrasts.metric == metric]
    n = len(sub)
    parts = []
    for effect, label in (("Delta_wavelength", "Wavelength"),
                          ("Delta_geometry", "Geometry")):
        d = sub[effect].to_numpy(dtype=float)
        ref = float(np.median(d))
        n_match = int(((d > 0) == (ref > 0)).sum()) if ref != 0 else n
        parts.append(f"{label}: {n_match}/{n} fibers")
    return "\n".join(parts)


def draw_panel_c(ax, data: FigureData, rng: np.random.Generator) -> pd.DataFrame:
    return _interaction_panel(
        ax, data, "acf_fwhm_px_median", "Horizontal ACF 1/e width (pixels)", rng,
        annot=_fiber_sign_annotation(data, "acf_fwhm_px_median"), fmt="%.0f")


def draw_panel_d(ax, data: FigureData, rng: np.random.Generator) -> pd.DataFrame:
    return _interaction_panel(
        ax, data, "psd_centroid_cyc_per_px_median",
        "PSD centroid (cycles pixel$^{-1}$)", rng,
        annot=_fiber_sign_annotation(data, "psd_centroid_cyc_per_px_median"),
        fmt="%.2f", annot_loc="lower right")


EFFECT_LABELS = {
    "Delta_wavelength": "Wavelength",
    "Delta_geometry": "Excitation pathway",
    "Delta_interaction": "Interaction",
}
FAMILY_METRIC = {
    "ACF-derived spatial complexity": "acf_complexity_oriented_median",
    "PSD spectral centroid": "psd_centroid_cyc_per_px_median",
}


def compute_effects(data: FigureData, rng: np.random.Generator) -> pd.DataFrame:
    """Standardized paired effects, oriented so that rightward = higher spatial
    complexity (ACF uses -log(acf_fwhm_px); PSD uses the centroid directly)."""
    rows = []
    for family, metric in FAMILY_METRIC.items():
        sub = data.contrasts[data.contrasts.metric == metric]
        if len(sub) != 5:
            raise ValueError(f"expected 5 fiber rows for {metric}, got {len(sub)}")
        for effect in ("Delta_wavelength", "Delta_geometry", "Delta_interaction"):
            d = sub[effect].to_numpy(dtype=float)
            est = standardized_paired_effect(d)
            lo, hi, n_ok = cluster_bootstrap_stat(d, standardized_paired_effect, rng)
            rows.append({
                "family": family, "metric": metric, "effect": effect,
                "effect_label": EFFECT_LABELS[effect],
                "per_fiber_deltas": ";".join(f"{x:.6g}" for x in d),
                "n_positive": int((d > 0).sum()), "n_negative": int((d < 0).sum()),
                "standardized_paired_effect": est,
                "ci_low": lo, "ci_high": hi,
                "bootstrap_valid_draws": n_ok,
            })
    return pd.DataFrame(rows)


FAMILY_SHORT = {
    "ACF-derived spatial complexity": "ACF complexity",
    "PSD spectral centroid": "PSD centroid",
}
EFFECT_TICK = {
    "Delta_wavelength": "Wavelength",
    "Delta_geometry": "Excitation\npathway",
    "Delta_interaction": "Interaction",
}


def draw_panel_e(ax, effects: pd.DataFrame, ratios: dict[str, float],
                 x_lim: tuple[float, float] | None = None) -> None:
    families = list(FAMILY_METRIC)
    order = ["Delta_wavelength", "Delta_geometry", "Delta_interaction"]
    # Keep CI display caps separate from point estimates: expand the axis so
    # every standardized estimate stays inside the frame after F3 remeasure.
    if x_lim is None:
        ests = effects["standardized_paired_effect"].astype(float)
        x_lim = (min(-5.5, float(ests.min()) - 0.6), max(8.0, float(ests.max()) + 0.8))
    y, ylabels = [], []
    yc = 0.0
    header_y = []
    for fam in families:
        header_y.append(yc)
        yc -= 0.80
        for eff in order:
            # extra clearance around the two-line "Excitation pathway" label
            if eff == "Delta_geometry":
                yc -= 0.30
            y.append(yc)
            ylabels.append(EFFECT_TICK[eff])
            yc -= 0.74
            if eff == "Delta_geometry":
                yc -= 0.30
        yc -= 0.30
    y_min = min(y) - 0.55
    y_max = max(header_y) + 1.85
    # zero line stops below the ratio-note headroom row
    zero_top = (max(header_y) + 0.42 - y_min) / (y_max - y_min)
    ax.axvline(0.0, ymax=zero_top, color=GRAY_LIGHT, lw=0.8, ls=(0, (4, 3)), zorder=1)
    cap = x_lim[1] - 0.15
    k = 0
    for fam in families:
        sub = effects[effects.family == fam].set_index("effect")
        for eff in order:
            r = sub.loc[eff]
            est = float(r.standardized_paired_effect)
            hi = float(r.ci_high)
            hi_draw = min(hi, cap)
            xerr_left = max(0.0, est - float(r.ci_low))
            xerr_right = max(0.0, hi_draw - est)
            ax.errorbar([est], [y[k]],
                        xerr=[[xerr_left], [xerr_right]],
                        color=GRAY_DARK, marker="o", ms=4.4, mfc=GRAY_DARK,
                        mec="white", mew=0.5, elinewidth=1.0,
                        capsize=0.0 if hi > cap else 2.0, capthick=1.0, zorder=3)
            if hi > cap:  # CI extends beyond the plotted range; state the bound
                ax.annotate("", xy=(x_lim[1], y[k]), xytext=(cap - 0.05, y[k]),
                            arrowprops=dict(arrowstyle="-|>", color=GRAY_DARK,
                                            lw=1.0, shrinkA=0, shrinkB=0))
                ax.text(cap - 0.35, y[k] + 0.16, f"{hi:.0f}", fontsize=FS_ANNOT - 1.0,
                        color=GRAY_MID, ha="right", va="bottom")
            k += 1
    for fy, fam in zip(header_y, families):
        ax.text(0.02, fy, FAMILY_SHORT[fam], transform=ax.get_yaxis_transform(),
                fontsize=FS_ANNOT, color=GRAY_DARK, ha="left", va="center",
                fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, linespacing=0.88)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(*x_lim)
    ax.set_xlabel("Standardized paired effect\n(\u2192 higher spatial complexity)")
    # ratio note in the reserved headroom row above the first family header
    ax.text(0.98, max(header_y) + 1.20,
            "|geometry| / |wavelength|\n"
            f"ACF \u2248 {ratios['acf']:.2f},  PSD \u2248 {ratios['psd']:.2f}",
            transform=ax.get_yaxis_transform(), fontsize=FS_ANNOT - 1.0,
            color=GRAY_MID, ha="right", va="center", linespacing=1.15)
    ax.spines["left"].set_visible(False)


# ----------------------------------------------------------------------------- assembly


def _panel_letter(fig, x: float, y: float, letter: str) -> None:
    fig.text(x, y, letter, fontsize=FS_PANEL, fontweight="bold", ha="left",
             va="top", family="Arial")


def render_main_figure(data: FigureData, images: dict[str, np.ndarray],
                       limits: dict[str, list[float]],
                       effects: pd.DataFrame, ratios: dict[str, float],
                       rng_profiles: np.random.Generator,
                       rng_cd: np.random.Generator,
                       fig_dir: Path) -> dict[str, pd.DataFrame]:
    fig = plt.figure(figsize=(FIG_W_MM * MM, FIG_H_MM * MM))
    gs = GridSpec(2, 6, figure=fig,
                  height_ratios=[1.02, 1.0],
                  left=0.075, right=0.985, top=0.925, bottom=0.115,
                  wspace=0.62, hspace=0.56)
    # row 1: a (0.42) | b (0.58) using 6 columns → a: 0-2 (~0.42), b: 2-6
    spec_a = gs[0, 0:2]
    ax_b = fig.add_subplot(gs[0, 2:6])
    ax_c = fig.add_subplot(gs[1, 0:2])
    # spacer sub-grids keep d/e y labels clear of the panels to their left
    spec_d = GridSpecFromSubplotSpec(1, 2, gs[1, 2:4], width_ratios=[0.14, 0.86])
    ax_d = fig.add_subplot(spec_d[0, 1])
    spec_e = GridSpecFromSubplotSpec(1, 2, gs[1, 4:6], width_ratios=[0.18, 0.82])
    ax_e = fig.add_subplot(spec_e[0, 1])

    draw_panel_a(fig, spec_a, images, limits)
    acf_df = draw_panel_b(ax_b, data, rng_profiles)
    c_df = draw_panel_c(ax_c, data, rng_cd)
    d_df = draw_panel_d(ax_d, data, rng_cd)
    draw_panel_e(ax_e, effects, ratios)

    pos_a = spec_a.get_position(fig)
    letter_x_b = ax_b.get_position().x0 - 0.055
    _panel_letter(fig, pos_a.x0 - 0.055, 0.985, "a")
    _panel_letter(fig, letter_x_b, 0.985, "b")
    _panel_letter(fig, ax_c.get_position().x0 - 0.055,
                  ax_c.get_position().y1 + 0.055, "c")
    # letter d shares the exact x of letter b (column-aligned panel labels)
    _panel_letter(fig, letter_x_b, ax_d.get_position().y1 + 0.055, "d")
    _panel_letter(fig, gs[1, 4:6].get_position(fig).x0 - 0.012,
                  ax_e.get_position().y1 + 0.055, "e")

    for ext in ("pdf", "svg", "png"):
        fig.savefig(fig_dir / f"Figure_orthogonal_control_main.{ext}",
                    dpi=PNG_DPI if ext == "png" else None)
    plt.close(fig)
    return {"acf": acf_df, "c": c_df, "d": d_df}


def render_individual_panels(data: FigureData, images: dict[str, np.ndarray],
                             limits: dict[str, list[float]], effects: pd.DataFrame,
                             ratios: dict[str, float], panels_dir: Path) -> None:
    panels_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(78 * MM, 78 * MM))
    gs = GridSpec(1, 1, figure=fig, left=0.16, right=0.96, top=0.86, bottom=0.05)
    draw_panel_a(fig, gs[0, 0], images, limits)
    _save_panel(fig, panels_dir / "panel_a_factorial_design")

    fig, ax = plt.subplots(figsize=(96 * MM, 62 * MM),
                           gridspec_kw=dict(left=0.13, right=0.97, top=0.95, bottom=0.19))
    draw_panel_b(ax, data, np.random.default_rng(BOOTSTRAP_SEED))
    _save_panel(fig, panels_dir / "panel_b_acf_profiles")

    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    fig, ax = plt.subplots(figsize=(62 * MM, 62 * MM),
                           gridspec_kw=dict(left=0.20, right=0.94, top=0.95, bottom=0.15))
    draw_panel_c(ax, data, rng)
    _save_panel(fig, panels_dir / "panel_c_acf_interaction")

    fig, ax = plt.subplots(figsize=(62 * MM, 62 * MM),
                           gridspec_kw=dict(left=0.22, right=0.94, top=0.95, bottom=0.15))
    draw_panel_d(ax, data, rng)
    _save_panel(fig, panels_dir / "panel_d_psd_interaction")

    fig, ax = plt.subplots(figsize=(72 * MM, 64 * MM),
                           gridspec_kw=dict(left=0.26, right=0.96, top=0.94, bottom=0.22))
    draw_panel_e(ax, effects, ratios)
    _save_panel(fig, panels_dir / "panel_e_effect_decomposition")


def _save_panel(fig, stem: Path) -> None:
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=PNG_DPI if ext == "png" else None)
    plt.close(fig)


# ----------------------------------------------------------------------------- caption / metadata


CAPTION = """# Figure — Orthogonal wavelength × excitation-pathway control (draft caption)

**Fully paired 2×2 optical control experiment on five fixed-package fibers.**
**a**, Median-template output speckle photographs of one example fiber (selected
by a fixed rule for maximal illustrative contrast between conditions; display
only — all statistics use all five fibers) under the four paired conditions:
532 nm and 650 nm illumination, each with axial and lateral excitation pathways.
Images are true-colour camera photographs (background-corrected temporal-median
intensity, no pseudo-colour). All four images share identical
region-of-interest size (384 × 384 pixels, no resizing), identical frozen
preprocessing, and an identical quantile-based display scaling; scale bar,
100 pixels.
**b**, Radial autocorrelation (ACF) profiles of the detail response for the four
conditions. Curves show the across-fiber mean of per-fiber profiles (each fiber
first averaged over its three technical acquisition repeats); shaded bands are
95% confidence intervals from a fiber-level cluster bootstrap (10,000 resamples).
Open circles mark the 0.5 crossings; the ACF full width at half maximum is
defined as acf_fwhm_px = 2 × acf_r50_px. **c**, Fiber-level paired interaction
plot of ACF FWHM. Thin lines connect the two excitation pathways for each
individual fiber (median over technical repeats); thick lines and filled symbols
show condition medians with fiber-level bootstrap 95% confidence intervals.
**d**, As in c, for the power-spectral-density (PSD) spatial-frequency centroid
(power-weighted mean radial spatial frequency, DC excluded). **e**, Standardized
paired effects (mean paired difference divided by its standard deviation across
fibers) for the wavelength contrast (532 − 650 nm), the excitation-pathway
contrast (lateral − axial), and their interaction, oriented so that positive
values indicate higher spatial complexity (−log ACF FWHM for the ACF family; PSD
centroid directly). Points show estimates; horizontal bars are fiber-level
cluster bootstrap 95% confidence intervals (10,000 resamples).

For all statistics the fiber is the independent unit (n = 5); the three
acquisitions per condition (M0–M2) are technical repeats and are aggregated
within fiber before any cross-fiber statistic. The data indicate a dominant
wavelength effect on the spatial scale and spatial-frequency content of the
output, with a smaller, wavelength-dependent modulation by the excitation
pathway (largest at 650 nm).
"""


def write_outputs(fig_dir: Path, data: FigureData, rep: dict[str, Any],
                  effects: pd.DataFrame, ratios: dict[str, float],
                  frames: dict[str, pd.DataFrame], fonts: dict[str, str],
                  cfg: Experiment02bConfig, limits: dict[str, list[float]]) -> None:
    (fig_dir / "orthogonal_control_caption.md").write_text(CAPTION)

    acf_prof = frames["acf"]
    acf_fiber = frames["c"].assign(kind="fiber_condition")
    pd.concat([acf_prof, acf_fiber], ignore_index=True).to_csv(
        fig_dir / "plotting_data_acf.csv", index=False)
    frames["d"].assign(kind="fiber_condition").to_csv(
        fig_dir / "plotting_data_psd.csv", index=False)
    effects.to_csv(fig_dir / "plotting_data_effects.csv", index=False)

    (fig_dir / "representative_selection.json").write_text(
        json.dumps(rep, indent=2) + "\n")

    import matplotlib as mpl
    meta = {
        "figure": "Figure_orthogonal_control_main",
        "panels": ["a_factorial_design", "b_acf_profiles", "c_acf_interaction",
                   "d_psd_interaction", "e_effect_decomposition"],
        "input_file_sha256": data.input_hashes,
        "data_columns_used": {
            "per_fiber_condition_metrics": PRIMARY_FIBER_COLS,
            "factorial_contrasts": list(FAMILY_METRIC.values())
                                   + ["Delta_wavelength", "Delta_geometry", "Delta_interaction"],
            "acf_profiles": "F{fiber}_{wl}_{geom}_M{repeat}_radial (mean over repeats per fiber)",
        },
        "n_fibers": 5,
        "statistical_unit": "fiber",
        "technical_repeat_aggregation": {
            "scalars": "median over M0-M2 (from frozen per_fiber_condition_metrics)",
            "acf_profiles": "mean over M0-M2 within fiber before cross-fiber statistics",
        },
        "bootstrap": {"n_iterations": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED,
                      "resampling_unit": "fiber", "ci": "percentile 95%"},
        "representative_selection_rule": rep["rule"],
        "representative_fiber": rep["representative_fiber"],
        "panel_a_display": {
            "content": "background-corrected temporal-median true-colour (BGR) speckle photo",
            "rendering": "true camera colour (BGR -> RGB), no pseudo-colour; scaled by wavelength-matched channel quantiles",
            "limit_rule": f"wavelength-matched channel quantiles [{PANEL_A_Q_LOW}, {PANEL_A_Q_HIGH}] (identical rule for all four images)",
            "display_limits_by_condition": limits,
            "roi_px": cfg.analysis.analysis_inner_size_px,
            "no_resize": True,
            "scale_bar_px": 100,
        },
        "effect_orientation": "rightward = higher spatial complexity; ACF family uses -log(acf_fwhm_px)",
        "geometry_to_wavelength_ratio": ratios,
        "colors": {"532_nm": WL_COLOR[532], "650_nm": WL_COLOR[650],
                   "grays": [GRAY_DARK, GRAY_MID, GRAY_LIGHT]},
        "fonts": {"family": "Arial", "resolved_paths": fonts,
                  "panel_letter_pt": FS_PANEL, "axis_label_pt": FS_LABEL,
                  "tick_pt": FS_TICK, "annotation_pt": FS_ANNOT},
        "figure_size_mm": [FIG_W_MM, FIG_H_MM],
        "png_dpi": PNG_DPI,
        "vector_text": "editable (pdf.fonttype=42, svg.fonttype=none)",
        "software": {
            "python": sys.version.split()[0],
            "matplotlib": mpl.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
    }
    (fig_dir / "figure_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


# ----------------------------------------------------------------------------- entry


def make_publication_figure(cfg: Experiment02bConfig) -> dict[str, Any]:
    fonts = setup_fonts()
    out_dir = Path(cfg.paths.output_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = load_inputs(out_dir)
    display_fiber, contrast_scores = select_display_fiber(data)
    rep = select_representative(data, display_fiber=display_fiber)
    rep["contrast_score_by_fiber"] = contrast_scores
    logger.info("panel-a display fiber: F%d (contrast rule, scores=%s)",
                display_fiber, contrast_scores)

    images, limits = build_panel_a_images(rep, cfg)

    rng_effects = np.random.default_rng(BOOTSTRAP_SEED + 2)
    effects = compute_effects(data, rng_effects)

    ratios = {}
    for fam_key, metric in (("acf", "acf_fwhm_px"), ("psd", "psd_centroid_cyc_per_px")):
        row = data.contrast_summary[
            (data.contrast_summary.metric == metric)
            & (data.contrast_summary.effect == "geometry_to_wavelength_effect_ratio")
        ]
        ratios[fam_key] = abs(float(row.iloc[0].geometry_to_wavelength_effect_ratio))

    frames = render_main_figure(
        data, images, limits, effects, ratios,
        rng_profiles=np.random.default_rng(BOOTSTRAP_SEED),
        rng_cd=np.random.default_rng(BOOTSTRAP_SEED + 1),
        fig_dir=fig_dir,
    )
    render_individual_panels(data, images, limits, effects, ratios,
                             fig_dir / "panels")
    write_outputs(fig_dir, data, rep, effects, ratios, frames, fonts, cfg, limits)

    return {
        "figure_dir": str(fig_dir),
        "representative_fiber": rep["representative_fiber"],
        "fonts": fonts,
        "ratios": ratios,
    }
