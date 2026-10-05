#!/usr/bin/env python3
"""Recalculate Fig. 5(d) ACF widths with Supplementary Note 4.2 radial FWHM.

Does NOT overwrite frozen Exp02b metrics or the original panel_c image.
Saves all new tables/figures under outputs/figures/s42_radial_acf_fwhm/.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1]

from experiment02b.config import load_config
from experiment02b.preprocessing import (
    crop_fixed_square,
    intensity_centroid,
    subtract_background,
    center_inner_square,
)
from experiment02b.video_io import decode_channel_video, select_analysis_frames
from puf_common.acf_features import (
    _fwhm_1d,
    extract_acf_features,
    normalized_acf_2d,
    radial_acf_profile,
)
from puf_common.features import median_template
from puf_common.masks import build_valid_mask_from_refs

LOGGER = logging.getLogger("s42_acf")

OUT_SUB = "figures/s42_radial_acf_fwhm"
MAX_LAG = 64  # Experiment 0 / S4.2
MASK_ABS_FLOOR = 5.0  # Experiment 1 / S4.1
MASK_PERCENTILE = 10.0


def radial_acf_fwhm_s42(img: np.ndarray, mask: np.ndarray, *, max_lag: int = MAX_LAG) -> dict[str, Any]:
    """S4.2 radial ACF FWHM: masked intensity, 2× zero-pad, no Hann window.

    Radial profile C(r) is even-extended so the same ``_fwhm_1d`` interpolator
    used for horizontal/vertical ACF (and described in S4.2 as the two
    half-maximum crossings) is applied to a centered profile.
    """
    acf = normalized_acf_2d(img, mask, max_lag=max_lag)
    rad = radial_acf_profile(acf)
    # One-sided radial C(r), r = 0,1,2,... Peak is C(0). S4.2 FWHM is the
    # distance between the two half-maximum crossings of the even extension,
    # which equals 2 × r50 with linear interpolation on adjacent samples.
    # Do not pass the one-sided profile to _fwhm_1d: that helper assumes a
    # centered line (used for horizontal/vertical ACF) and its left/right
    # index convention yields identically odd integers on binned radial C(r).
    r50 = float("nan")
    peak = float(rad[0]) if rad.size else float("nan")
    half = 0.5 * peak if np.isfinite(peak) and peak > 0 else 0.5
    for i in range(1, len(rad)):
        if rad[i] <= half:
            y0, y1 = float(rad[i - 1]), float(rad[i])
            if abs(y1 - y0) < 1e-18:
                r50 = float(i)
            else:
                t = (half - y0) / (y1 - y0)
                r50 = float((i - 1) + t)
            break
    fwhm = float(2.0 * r50) if np.isfinite(r50) else float("nan")
    naive = extract_acf_features(img, mask, max_lag=max_lag)
    return {
        "radial_acf_fwhm_px": fwhm,
        "radial_acf_fwhm_qc": "ok" if np.isfinite(fwhm) else "half_max_not_found",
        "r50_px": r50,
        "two_times_r50_px": fwhm,
        "exp0_extract_acf_fwhm_radial": float(naive["acf_fwhm_radial"]),
        "acf_fwhm_x": float(naive["acf_fwhm_x"]),
        "acf_fwhm_y": float(naive["acf_fwhm_y"]),
        "mask_coverage": float(np.mean(mask)),
        "n_valid_pixels": int(np.count_nonzero(mask)),
    }


def intensity_representative(row: pd.Series, cfg) -> tuple[np.ndarray | None, dict[str, Any]]:
    a = cfg.analysis
    color = str(row["color_label"]).lower()
    dec = decode_channel_video(row["source_path"], color_label=color)
    idx = select_analysis_frames(
        dec.timestamps_s,
        analysis_start_s=a.analysis_start_s,
        analysis_end_margin_s=a.analysis_end_margin_s,
        max_sampled_frames=a.max_sampled_frames,
    )
    bg_frames = []
    bg_mode = None
    for i in idx:
        corr, info = subtract_background(
            dec.frames[int(i)], dark=None, edge_fraction=a.background_edge_fraction
        )
        bg_mode = info["background_mode"]
        bg_frames.append(corr)
    global_median = median_template(bg_frames).astype(np.float32)
    cx, cy, cinfo = intensity_centroid(global_median, smooth_sigma=a.centroid_smooth_sigma_px)
    crop, cinfo2 = crop_fixed_square(global_median, cx, cy, a.crop_size_px)
    meta = {
        "background_mode": bg_mode,
        "centroid_ok": bool(cinfo.get("centroid_ok")),
        "crop_ok": bool(cinfo2.get("ok")),
        "crop_fail_reason": cinfo2.get("reason", ""),
        "n_frames": int(idx.size),
    }
    if crop is None:
        return None, meta
    inner = center_inner_square(crop, a.analysis_inner_size_px)
    return inner, meta


def summarize_condition(vals: np.ndarray) -> dict[str, float]:
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    return {
        "n": int(v.size),
        "median": float(np.median(v)),
        "Q25": float(np.quantile(v, 0.25)),
        "Q75": float(np.quantile(v, 0.75)),
        "mean": float(np.mean(v)),
        "std": float(np.std(v, ddof=1)) if v.size > 1 else float("nan"),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }


def draw_fig5d(device_df: pd.DataFrame, out_png: Path, out_pdf: Path, out_svg: Path | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiment02b.publication_figure import (
        FS_ANNOT,
        FS_LABEL,
        FS_TICK,
        GRAY_MID,
        WL_COLOR,
        setup_fonts,
        cluster_bootstrap_stat,
        GEOM_ORDER,
        WL_ORDER,
        MM,
    )

    setup_fonts()
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(58 * MM, 52 * MM), dpi=600)
    xpos = {"axial": 0.0, "lateral": 1.0}
    jitter = {532: -0.045, 650: 0.045}
    fibers = sorted(device_df.fiber_id.unique())
    for wl in WL_ORDER:
        color = WL_COLOR[wl]
        vals = {}
        for geom in GEOM_ORDER:
            cell = device_df[(device_df.wavelength_nm == wl) & (device_df.excitation_geometry == geom)]
            vals[geom] = cell.set_index("fiber_id")["radial_acf_fwhm_px"]
        for f in fibers:
            y = [float(vals[g].loc[f]) for g in GEOM_ORDER]
            x = [xpos[g] + jitter[wl] for g in GEOM_ORDER]
            ax.plot(x, y, color=color, lw=0.7, alpha=0.32, zorder=2)
            ax.plot(x, y, ls="none", marker="o", ms=2.6, mfc=color, mec="none", alpha=0.45, zorder=2)
        agg_x, agg_y, agg_lo, agg_hi = [], [], [], []
        for geom in GEOM_ORDER:
            v = vals[geom].to_numpy(dtype=float)
            med = float(np.median(v))
            lo, hi, _ = cluster_bootstrap_stat(v, np.median, rng)
            agg_x.append(xpos[geom] + jitter[wl])
            agg_y.append(med)
            agg_lo.append(med - lo)
            agg_hi.append(hi - med)
        ax.errorbar(
            agg_x, agg_y, yerr=[agg_lo, agg_hi], color=color, lw=1.5,
            marker="o", ms=4.6, mec="white", mew=0.6, capsize=2.2,
            capthick=1.0, elinewidth=1.0, zorder=4,
        )
        ax.annotate(
            f"{wl} nm", xy=(agg_x[-1] + 0.06, agg_y[-1]),
            fontsize=FS_ANNOT, color=color, va="center", ha="left",
        )
    ax.set_xlim(-0.3, 1.45)
    ax.set_xticks([0, 1], ["Axial", "Lateral"])
    ax.set_ylabel("Radial ACF FWHM (pixels)", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
    from matplotlib.lines import Line2D
    individual = Line2D([], [], color="#B4B4B4", marker="o", markersize=2.6, linewidth=0.7)
    median_ci = ax.errorbar([], [], yerr=[], color="#777777", marker="o", markersize=3,
                            capsize=2.2, linewidth=1)
    ax.legend([individual, median_ci], ["Individual fibers", "Median (95% bootstrap CI)"],
              loc="upper right", frameon=False, fontsize=4.8,
              handlelength=0.9, handletextpad=0.35, borderaxespad=0.1)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    fig.tight_layout()
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    if out_svg is not None:
        fig.savefig(out_svg)
    plt.close(fig)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg.paths.output_dir) / OUT_SUB
    out.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(Path(cfg.paths.output_dir) / "manifest" / "manifest.csv")
    prev = pd.read_csv(Path(cfg.paths.output_dir) / "metrics" / "per_fiber_condition_metrics.csv")

    video_rows = []
    for i, row in man.iterrows():
        LOGGER.info("video %d/%d %s", i + 1, len(man), row["filename"])
        inner, meta = intensity_representative(row, cfg)
        rec = {
            "filename": row["filename"],
            "fiber_id": int(row["fiber_id"]),
            "repeat_id": int(row["repeat_id"]),
            "raw_repeat_label": row["raw_repeat_label"],
            "wavelength_nm": int(row["wavelength_nm"]),
            "excitation_geometry": row["excitation_geometry"],
            **meta,
        }
        if inner is None:
            rec["radial_acf_fwhm_px"] = float("nan")
            video_rows.append(rec)
            continue
        mask, cov, thr = build_valid_mask_from_refs(
            [inner], abs_floor=MASK_ABS_FLOOR, percentile=MASK_PERCENTILE
        )
        mets = radial_acf_fwhm_s42(inner, mask)
        rec.update(mets)
        rec["mask_threshold"] = float(thr)
        rec["mask_coverage_build"] = float(cov)
        video_rows.append(rec)

    per_video = pd.DataFrame(video_rows)
    per_video.to_csv(out / "per_video_radial_acf_fwhm.csv", index=False)

    device_rows = []
    for (fid, wl, geom), g in per_video.groupby(
        ["fiber_id", "wavelength_nm", "excitation_geometry"], sort=True
    ):
        v = g["radial_acf_fwhm_px"].to_numpy(dtype=float)
        device_rows.append(
            {
                "fiber_id": int(fid),
                "wavelength_nm": int(wl),
                "excitation_geometry": geom,
                "n_repeats": int(np.isfinite(v).sum()),
                "radial_acf_fwhm_px": float(np.nanmedian(v)),
                "radial_acf_fwhm_mean_over_repeats": float(np.nanmean(v)),
                "radial_acf_fwhm_std_over_repeats": float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else float("nan"),
            }
        )
    device_df = pd.DataFrame(device_rows)
    # attach previous Fig. 5(d) values for comparison
    prev_s = prev.rename(columns={"acf_fwhm_px_median": "previous_detail_hann_acf_fwhm_px"})
    device_df = device_df.merge(
        prev_s[["fiber_id", "wavelength_nm", "excitation_geometry", "previous_detail_hann_acf_fwhm_px"]],
        on=["fiber_id", "wavelength_nm", "excitation_geometry"],
        how="left",
    )
    device_df.to_csv(out / "fig5d_device_radial_acf_fwhm.csv", index=False)

    cond_rows = []
    for (wl, geom), g in device_df.groupby(["wavelength_nm", "excitation_geometry"], sort=True):
        s = summarize_condition(g["radial_acf_fwhm_px"].to_numpy())
        s.update({"wavelength_nm": int(wl), "excitation_geometry": geom})
        cond_rows.append(s)
    cond_df = pd.DataFrame(cond_rows)
    cond_df.to_csv(out / "fig5d_condition_summary.csv", index=False)

    # qualitative comparison vs previous
    def _med(df, wl, geom, col):
        return float(df[(df.wavelength_nm == wl) & (df.excitation_geometry == geom)][col].median())

    new_532_ax = _med(device_df, 532, "axial", "radial_acf_fwhm_px")
    new_532_lat = _med(device_df, 532, "lateral", "radial_acf_fwhm_px")
    new_650_ax = _med(device_df, 650, "axial", "radial_acf_fwhm_px")
    new_650_lat = _med(device_df, 650, "lateral", "radial_acf_fwhm_px")
    old_532_ax = _med(device_df, 532, "axial", "previous_detail_hann_acf_fwhm_px")
    old_532_lat = _med(device_df, 532, "lateral", "previous_detail_hann_acf_fwhm_px")
    old_650_ax = _med(device_df, 650, "axial", "previous_detail_hann_acf_fwhm_px")
    old_650_lat = _med(device_df, 650, "lateral", "previous_detail_hann_acf_fwhm_px")

    finer_new = (new_532_ax < new_650_ax) and (new_532_lat < new_650_lat)
    lat_drops_650_new = new_650_lat < new_650_ax
    finer_old = (old_532_ax < old_650_ax) and (old_532_lat < old_650_lat)
    lat_drops_650_old = old_650_lat < old_650_ax

    summary = {
        "protocol": {
            "note": "Supplementary Note 4.2 radial ACF FWHM",
            "input": "dark-corrected (edge-median background) intensity representative; NOT detail/envelope",
            "roi": "same Exp02b centroid 512 crop, inner 384 analysis window",
            "valid_mask": f"Exp1/S4.1: mean intensity > max({MASK_ABS_FLOOR}, P{MASK_PERCENTILE})",
            "acf": "puf_common.normalized_acf_2d: zero outside mask, subtract valid-region mean, /sd (cancels after C00 norm), FFT zero-pad to 2H×2W, no Hann window",
            "fwhm": "even-extend radial C(r); _fwhm_1d interpolates the two 0.5 crossings (same interpolator as S4.2)",
            "device_aggregation": "median over M0–M2 technical repeats, matching original Fig. 5(d) unit",
            "max_lag_px": MAX_LAG,
        },
        "previous_fig5d": {
            "method": "Hann-windowed local-ratio DETAIL inner-384; acf_fwhm_px = 2 × radial r50; same-size FFT (no 2× pad)",
            "median_532_axial": old_532_ax,
            "median_532_lateral": old_532_lat,
            "median_650_axial": old_650_ax,
            "median_650_lateral": old_650_lat,
            "532_finer_than_650": finer_old,
            "lateral_decreases_650": lat_drops_650_old,
        },
        "s42_recalc": {
            "median_532_axial": new_532_ax,
            "median_532_lateral": new_532_lat,
            "median_650_axial": new_650_ax,
            "median_650_lateral": new_650_lat,
            "532_finer_than_650": finer_new,
            "lateral_decreases_650": lat_drops_650_new,
        },
        "qualitative_conclusion_unchanged": bool(finer_new == finer_old and lat_drops_650_new == lat_drops_650_old),
        "original_panel_not_overwritten": str(
            Path(cfg.paths.output_dir) / "figures/panels/panel_c_acf_interaction.png"
        ),
        "new_panel": str(out / "panel_d_radial_acf_fwhm.png"),
    }
    (out / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    draw_fig5d(
        device_df,
        out / "panel_d_radial_acf_fwhm.png",
        out / "panel_d_radial_acf_fwhm.pdf",
        out / "panel_d_radial_acf_fwhm.svg",
    )
    LOGGER.info("wrote %s", out)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
