"""Stage runners: manifest → qc/metrics → statistics → report."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiment02b.config import Experiment02bConfig
from experiment02b.factorial_statistics import (
    aggregate_fiber_condition,
    build_factorial_tables,
    complexity_oriented_acf,
)
from experiment02b.manifest import scan_videos, write_manifest
from experiment02b.preprocessing import (
    beam_envelope_qc,
    center_inner_square,
    crop_fixed_square,
    intensity_centroid,
    make_detail,
    prepare_analysis_patch,
    roi_coverage,
    subtract_background,
)
from experiment02b.qc import evaluate_qc
from experiment02b.spatial_acf import extract_acf_metrics
from experiment02b.spatial_psd import extract_psd_metrics
from experiment02b.temporal_stability import (
    block_to_block_ncc,
    frame_to_template_ncc,
    repeat_cv,
    repeat_to_repeat_ncc,
)
from experiment02b.channel_leakage import audit_bgr_sample
from experiment02b.video_io import (
    decode_channel_video,
    maximum_timestamp_gap,
    meta_to_dict,
    select_analysis_frames,
    split_time_blocks,
)
from puf_common.features import median_template

logger = logging.getLogger(__name__)

PRIMARY_METRICS = ["acf_fwhm_px", "psd_centroid_cyc_per_px"]
METRIC_COLS = [
    "acf_r50_px",
    "acf_fwhm_px",
    "acf_r1e_px",
    "acf_diameter_1e_px",
    "effective_correlation_area",
    "effective_speckle_degrees_of_freedom",
    "psd_centroid_cyc_per_px",
    "psd_rms_bandwidth_cyc_per_px",
    "psd_entropy_normalized",
    "psd_f50_cyc_per_px",
    "psd_f90_cyc_per_px",
    "high_frequency_fraction_above_0p1",
    "spatial_spectral_participation_ratio",
    "frame_to_template_ncc_median",
    "block_to_block_ncc_median",
]


def _out(cfg: Experiment02bConfig) -> Path:
    d = Path(cfg.paths.output_dir)
    d.mkdir(parents=True, exist_ok=True)
    for sub in ("manifest", "qc", "metrics", "profiles", "statistics", "reports"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def stage_manifest(cfg: Experiment02bConfig) -> dict[str, Any]:
    out = _out(cfg)
    df, audit = scan_videos(cfg.paths.videos_root)
    write_manifest(df, out / "manifest" / "manifest.csv")
    (out / "manifest" / "inventory_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if not audit.get("ok"):
        logger.error("Manifest inventory incomplete: %s", json.dumps(audit, indent=2))
    else:
        logger.info("Manifest OK: 60 videos, complete 2×2×5×3")
    return {"audit": audit, "n_parsed": int(len(df))}


def _load_optional_dark(cfg: Experiment02bConfig, color_label: str) -> np.ndarray | None:
    if cfg.paths.dark_video is None:
        return None
    p = Path(cfg.paths.dark_video)
    if not p.is_file():
        logger.warning("dark_video configured but missing: %s", p)
        return None
    # Dark must use the same wavelength-matched channel as the science stream.
    dec = decode_channel_video(p, color_label=color_label)
    return median_template([dec.frames[i] for i in range(dec.frames.shape[0])]).astype(np.float32)


def process_one_video(
    row: dict[str, Any],
    cfg: Experiment02bConfig,
    *,
    dark: np.ndarray | None,
    crop_size: int | None = None,
    inner_size: int | None = None,
    sigma: float | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    a = cfg.analysis
    crop_size = int(crop_size or a.crop_size_px)
    inner_size = int(inner_size or a.analysis_inner_size_px)
    sigma = float(sigma if sigma is not None else a.envelope_sigma_px)

    result: dict[str, Any] = dict(row)
    profiles: dict[str, np.ndarray] = {}
    color_label = str(row["color_label"]).lower()
    try:
        dec = decode_channel_video(row["source_path"], color_label=color_label)
        result.update(meta_to_dict(dec.meta))
        result["decode_ok"] = True
        result["decoded_frame_count"] = int(dec.frames.shape[0])
        result["maximum_timestamp_gap"] = maximum_timestamp_gap(dec.timestamps_s)
        result["effective_fps"] = (
            float((dec.frames.shape[0] - 1) / max(dec.timestamps_s[-1] - dec.timestamps_s[0], 1e-9))
            if dec.frames.shape[0] > 1
            else float("nan")
        )
        if dec.bgr_sample is not None:
            leak = audit_bgr_sample(dec.bgr_sample, color_label)
            result.update(leak)
        else:
            result["channel_leakage_status"] = "WARN"
            result["channel_leakage_reasons"] = ["missing_bgr_sample_for_leakage_audit"]
    except Exception as exc:  # noqa: BLE001
        result["decode_ok"] = False
        result["decode_error"] = str(exc)
        result["crop_ok"] = False
        result.update(evaluate_qc(result, cfg))
        return result, profiles

    idx = select_analysis_frames(
        dec.timestamps_s,
        analysis_start_s=a.analysis_start_s,
        analysis_end_margin_s=a.analysis_end_margin_s,
        max_sampled_frames=a.max_sampled_frames,
    )
    result["retained_frame_count"] = int(idx.size)
    blocks_idx = split_time_blocks(dec.timestamps_s, idx, a.n_temporal_blocks)

    # background-corrected retained frames
    bg_frames = []
    bg_mode = None
    for i in idx:
        corr, info = subtract_background(
            dec.frames[int(i)], dark=dark, edge_fraction=a.background_edge_fraction
        )
        bg_mode = info["background_mode"]
        bg_frames.append(corr)
    result["background_mode"] = bg_mode

    global_median = median_template(bg_frames).astype(np.float32)
    beam = beam_envelope_qc(global_median)
    result.update(beam)

    cx, cy, cinfo = intensity_centroid(global_median, smooth_sigma=a.centroid_smooth_sigma_px)
    result["centroid_ok"] = bool(cinfo.get("centroid_ok"))
    crop, cinfo2 = crop_fixed_square(global_median, cx, cy, crop_size)
    result["crop_ok"] = bool(cinfo2.get("ok"))
    result["crop_fail_reason"] = cinfo2.get("reason", "")
    if crop is None:
        result.update(evaluate_qc(result, cfg))
        return result, profiles

    detail = make_detail(crop, sigma=sigma, epsilon=a.epsilon)
    inner_detail = center_inner_square(detail, inner_size)
    result["roi_coverage"] = roi_coverage(inner_detail)
    # intensity drift QC on crop means
    means = [float(np.mean(f)) for f in bg_frames]
    result["intensity_drift"] = float(np.std(means) / (abs(np.mean(means)) + 1e-6))

    patch, mask = prepare_analysis_patch(detail, inner=inner_size)
    acf_m, acf_p = extract_acf_metrics(patch)
    psd_m, psd_p = extract_psd_metrics(
        patch, high_freq_threshold=a.high_frequency_threshold_cyc_per_px
    )
    result.update(acf_m)
    result.update(psd_m)
    result["acf_complexity_oriented"] = complexity_oriented_acf(result["acf_fwhm_px"])
    result["normalized_acf_fwhm"] = (
        float(result["acf_fwhm_px"] / result["equivalent_beam_radius"])
        if np.isfinite(result.get("equivalent_beam_radius", float("nan")))
        and result["equivalent_beam_radius"] > 1e-12
        else float("nan")
    )
    result["normalized_psd_centroid"] = (
        float(result["psd_centroid_cyc_per_px"] * result["equivalent_beam_radius"])
        if np.isfinite(result.get("equivalent_beam_radius", float("nan")))
        else float("nan")
    )

    # temporal stability on detail space (same crop/sigma)
    detail_frames = []
    for f in bg_frames:
        c, info = crop_fixed_square(f, cx, cy, crop_size)
        if c is None:
            continue
        d = make_detail(c, sigma=sigma, epsilon=a.epsilon)
        detail_frames.append(center_inner_square(d, inner_size))
    tmpl = median_template(detail_frames).astype(np.float32) if detail_frames else inner_detail
    result.update(frame_to_template_ncc(detail_frames, tmpl, mask=None))

    block_tmpls = []
    for bidx in blocks_idx:
        if bidx.size == 0:
            continue
        bframes = []
        for i in bidx:
            # map absolute index → position in idx/bg_frames
            # rebuild from decoded for simplicity
            corr, _ = subtract_background(
                dec.frames[int(i)], dark=dark, edge_fraction=a.background_edge_fraction
            )
            c, info = crop_fixed_square(corr, cx, cy, crop_size)
            if c is None:
                continue
            bframes.append(center_inner_square(make_detail(c, sigma=sigma, epsilon=a.epsilon), inner_size))
        if bframes:
            block_tmpls.append(median_template(bframes).astype(np.float32))
    result.update(block_to_block_ncc(block_tmpls))

    profiles = {
        "acf_radius_px": acf_p["acf_radius_px"],
        "acf_radial": acf_p["acf_radial"],
        "psd_freq_cyc_per_px": psd_p["psd_freq_cyc_per_px"],
        "psd_radial_energy": psd_p["psd_radial_energy"],
        "detail_inner": inner_detail.astype(np.float32),
    }
    result["analysis_crop_size_px"] = crop_size
    result["analysis_inner_size_px"] = inner_size
    result["envelope_sigma_px"] = sigma
    result.update(evaluate_qc(result, cfg))
    return result, profiles


def stage_metrics(cfg: Experiment02bConfig) -> dict[str, Any]:
    out = _out(cfg)
    man_path = out / "manifest" / "manifest.csv"
    if not man_path.is_file():
        stage_manifest(cfg)
    man = pd.read_csv(man_path)
    if man.empty:
        raise RuntimeError("Empty manifest — place factorial videos and re-run validate/manifest")

    # Dark is channel-specific; load per color on demand.
    dark_by_color: dict[str, np.ndarray | None] = {"green": None, "red": None}
    if cfg.paths.dark_video is not None:
        dark_by_color["green"] = _load_optional_dark(cfg, "green")
        dark_by_color["red"] = _load_optional_dark(cfg, "red")
    rows = []
    acf_pack: dict[str, Any] = {}
    psd_pack: dict[str, Any] = {}
    detail_by_key: dict[tuple, np.ndarray] = {}

    for i, r in man.iterrows():
        logger.info("metrics %d/%d %s", i + 1, len(man), r["filename"])
        dark = dark_by_color.get(str(r["color_label"]).lower())
        res, prof = process_one_video(r.to_dict(), cfg, dark=dark)
        rows.append(res)
        key = f"F{int(r.fiber_id)}_{int(r.wavelength_nm)}_{r.excitation_geometry}_{r.raw_repeat_label}"
        if prof:
            acf_pack[f"{key}_radius"] = prof["acf_radius_px"]
            acf_pack[f"{key}_radial"] = prof["acf_radial"]
            psd_pack[f"{key}_freq"] = prof["psd_freq_cyc_per_px"]
            psd_pack[f"{key}_radial"] = prof["psd_radial_energy"]
            detail_by_key[
                (int(r.fiber_id), int(r.wavelength_nm), str(r.excitation_geometry), int(r.repeat_id))
            ] = prof["detail_inner"]

    per_video = pd.DataFrame(rows)
    per_video.to_csv(out / "metrics" / "per_video_metrics.csv", index=False)
    qc_cols = [c for c in per_video.columns if c.startswith("qc_") or c in (
        "filename", "fiber_id", "wavelength_nm", "excitation_geometry", "raw_repeat_label",
        "retained_frame_count", "saturation_fraction", "roi_coverage", "decode_ok", "crop_ok",
        "maximum_timestamp_gap", "effective_fps", "background_mode",
    )]
    per_video.loc[:, [c for c in qc_cols if c in per_video.columns]].to_csv(
        out / "qc" / "video_qc.csv", index=False
    )

    # repeat-to-repeat NCC per fiber×λ×geometry
    stab_rows = []
    for (fiber, wl, geom), g in per_video.groupby(
        ["fiber_id", "wavelength_nm", "excitation_geometry"]
    ):
        reps = {}
        for _, rr in g.iterrows():
            k = (int(fiber), int(wl), str(geom), int(rr.repeat_id))
            if k in detail_by_key:
                reps[int(rr.repeat_id)] = detail_by_key[k]
        rt = repeat_to_repeat_ncc(reps)
        stab_rows.append(
            {
                "fiber_id": int(fiber),
                "wavelength_nm": int(wl),
                "excitation_geometry": str(geom),
                **rt,
                "acf_metric_repeat_cv": repeat_cv(g["acf_fwhm_px"].tolist())
                if "acf_fwhm_px" in g
                else float("nan"),
                "psd_metric_repeat_cv": repeat_cv(g["psd_centroid_cyc_per_px"].tolist())
                if "psd_centroid_cyc_per_px" in g
                else float("nan"),
            }
        )
    pd.DataFrame(stab_rows).to_csv(out / "metrics" / "temporal_stability_metrics.csv", index=False)

    fiber = aggregate_fiber_condition(per_video, METRIC_COLS)
    fiber.to_csv(out / "metrics" / "per_fiber_condition_metrics.csv", index=False)

    np.savez_compressed(out / "profiles" / "acf_profiles.npz", **acf_pack)
    np.savez_compressed(out / "profiles" / "psd_profiles.npz", **psd_pack)

    return {
        "n_videos": int(len(per_video)),
        "n_pass": int((per_video["qc_status"] == "PASS").sum()) if "qc_status" in per_video else 0,
        "n_warn": int((per_video["qc_status"] == "WARN").sum()) if "qc_status" in per_video else 0,
        "n_fail": int((per_video["qc_status"] == "FAIL").sum()) if "qc_status" in per_video else 0,
    }


def stage_statistics(cfg: Experiment02bConfig) -> dict[str, Any]:
    out = _out(cfg)
    per_video = pd.read_csv(out / "metrics" / "per_video_metrics.csv")
    tables = build_factorial_tables(
        per_video,
        primary_metrics=PRIMARY_METRICS + ["acf_complexity_oriented", "normalized_acf_fwhm", "normalized_psd_centroid"],
        bootstrap_iterations=cfg.analysis.bootstrap_iterations,
        bootstrap_seed=cfg.analysis.bootstrap_seed,
        mixed_model_enabled=cfg.analysis.mixed_model_enabled,
    )
    tables["contrasts"].to_csv(out / "statistics" / "factorial_contrasts.csv", index=False)
    tables["contrast_summary"].to_csv(out / "statistics" / "contrast_summary.csv", index=False)
    tables["bootstrap_intervals"].to_csv(out / "statistics" / "bootstrap_intervals.csv", index=False)
    # mixed model as JSON lines / csv-ish
    mm = tables["mixed_model_results"]
    pd.DataFrame(
        [{"status": m.get("status"), "metric": m.get("metric"), "reason": m.get("reason", "")} for m in mm]
    ).to_csv(out / "statistics" / "mixed_model_results.csv", index=False)
    (out / "statistics" / "mixed_model_details.json").write_text(json.dumps(mm, indent=2, default=str) + "\n")

    # sensitivity: alternate crop/sigma on a reduced pass is expensive; record protocol placeholders
    # if full metrics already contain primary; run lightweight recompute only if requested later
    sens_rows = []
    for crop, inner in cfg.sensitivity.crop_inner_pairs:
        sens_rows.append(
            {
                "kind": "crop_inner",
                "crop_size_px": crop,
                "analysis_inner_size_px": inner,
                "status": "PRE_SPECIFIED",
                "note": "Recompute with --stage metrics_sensitivity (optional); primary remains 512/384.",
            }
        )
    for sig in cfg.sensitivity.envelope_sigmas_px:
        sens_rows.append(
            {
                "kind": "envelope_sigma",
                "envelope_sigma_px": sig,
                "status": "PRE_SPECIFIED",
                "note": "Primary sigma=42 locked in PROTOCOL_LOCK.md",
            }
        )
    pd.DataFrame(sens_rows).to_csv(out / "statistics" / "sensitivity_analysis.csv", index=False)
    tables["per_fiber_condition"].to_csv(out / "metrics" / "per_fiber_condition_metrics.csv", index=False)
    return {"n_contrast_rows": int(len(tables["contrasts"]))}


def stage_report(cfg: Experiment02bConfig) -> dict[str, Any]:
    from experiment02b.reporting import write_reports

    return write_reports(cfg)
