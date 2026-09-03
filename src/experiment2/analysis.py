"""Experiment 2 analysis orchestration."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from experiment2.challenge_validation import (
    validate_all_challenges,
    write_challenge_validation_csv,
    write_challenge_validation_json,
)
from experiment2.config import Experiment2Config
from experiment2.metadata import (
    MetadataRecord,
    MetadataValidationReport,
    load_metadata_csv,
    validate_metadata,
    write_metadata_validation_json,
)
from experiment2.score_construction import (
    build_detail_cm_for_green_device,
    build_detail_for_red_recording,
    compute_intra_scores,
    compute_inter_challenge_scores,
    compute_inter_device_scores,
    compute_red_drift_metrics,
    compute_short_term_ncc,
)
from experiment2.video_processing import ProcessedRecording, process_video
from puf_common.dark import (
    CameraSettings,
    assert_camera_settings_match,
    load_or_build_experiment1_dark,
    provenance_to_dict,
)
from puf_common.envelope import estimate_speckle_width, local_ratio_detail, speckle_contrast
from puf_common.masks import (
    build_valid_mask_from_refs,
    load_mask_png,
    mask_hash,
    save_mask_png,
)
from puf_common.metrics import (
    auc_roc,
    d_prime,
    equal_error_rate_with_threshold,
    margin,
    robust_gap,
)
from puf_common.statistics import paired_ttest, wilcoxon_signed_rank
from puf_common.tqdm_progress import make_video_progress, mute_console_logging, stage_tqdm

logger = logging.getLogger(__name__)


def _log_stage(message: str) -> None:
    """Immediate stage banner (visible even when stdout is piped to tee)."""
    print(message, flush=True)


def _record_label(device_id: str, key: str) -> str:
    return f"{device_id}/{key}"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    _ensure_dir(path.parent)
    keys = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _resolve_record_video(record: MetadataRecord, cfg: Experiment2Config) -> Path:
    path = Path(record.video_path)
    if not path.is_absolute():
        path = (cfg.root / path).resolve()
    return path


_EXPECTED_DARK_SHAPE = (1536, 2048)


def _load_fixed_state_dark(cfg: Experiment2Config):
    if not cfg.dark_frame.reuse_experiment1:
        raise ValueError("Experiment 2 requires dark_frame.reuse_experiment1 = true")
    artifact = cfg.dark_artifact_path
    if artifact is None or not artifact.exists():
        raise FileNotFoundError(
            "Experiment 2 dark template missing. "
            "Set dark_frame.experiment1_dark_artifact to the 2048×1536 dark NPZ."
        )
    dark_r, dark_g, prov = load_or_build_experiment1_dark(
        artifact_path=artifact,
        video_path=None,
        cache_path=None,
        roi=cfg.camera.roi,
        allow_missing=cfg.dark_frame.allow_missing_dark,
    )
    if dark_r.shape != _EXPECTED_DARK_SHAPE or dark_g.shape != _EXPECTED_DARK_SHAPE:
        raise ValueError(
            f"Dark template shape {dark_r.shape}/{dark_g.shape} does not match "
            f"recordings {_EXPECTED_DARK_SHAPE}."
        )
    return dark_r, dark_g, prov


def validate_dark_reuse(cfg: Experiment2Config) -> dict[str, Any]:
    dark_r, dark_g, prov = _load_fixed_state_dark(cfg)
    return {
        "valid": True,
        "reuse_experiment1": True,
        "dark_red_shape": list(dark_r.shape),
        "dark_green_shape": list(dark_g.shape),
        "provenance": provenance_to_dict(prov),
    }


def _record_camera_settings(record: MetadataRecord) -> CameraSettings:
    return CameraSettings(
        frame_width_px=record.frame_width_px,
        frame_height_px=record.frame_height_px,
        bit_depth=record.bit_depth,
        color_format=record.color_format,
        exposure_ms=record.exposure_ms,
        gain=record.gain,
        frame_rate_fps=record.frame_rate_fps,
        camera_id=record.camera_id,
        roi=None,
        channel_convention="opencv_bgr",
    )


def _load_or_build_device_mask(
    *,
    device_id: str,
    red_before: ProcessedRecording,
    green_c01: ProcessedRecording,
    cfg: Experiment2Config,
    out_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    exp1_mask_path = cfg.experiment1_valid_mask_path
    if exp1_mask_path is not None and exp1_mask_path.exists():
        mask = load_mask_png(exp1_mask_path)
        if mask.shape == red_before.blocks[0].shape:
            meta = {
                "mask_id": f"{device_id}_exp1_valid_mask",
                "mask_source": str(exp1_mask_path),
                "mask_coverage": float(mask.mean()),
                "mask_hash": mask_hash(mask),
                "preprocessing_version": cfg.analysis.preprocessing_version,
            }
            save_mask_png(mask, out_dir / f"{device_id}_valid_mask.png")
            return mask, meta

    mask, coverage, thr = build_valid_mask_from_refs(
        [red_before.blocks[0], green_c01.blocks[0]],
        abs_floor=cfg.analysis.valid_mask_abs_floor,
        percentile=cfg.analysis.valid_mask_percentile,
    )
    meta = {
        "mask_id": f"{device_id}_built_from_red_before_and_green_C01_block1",
        "mask_source": "build_valid_mask_from_refs(red_before.block1, green_C01.block1)",
        "mask_coverage": coverage,
        "threshold": thr,
        "mask_hash": mask_hash(mask),
        "preprocessing_version": cfg.analysis.preprocessing_version,
    }
    save_mask_png(mask, out_dir / f"{device_id}_valid_mask.png")
    return mask, meta


def _spatial_metrics_row(
    *,
    device_id: str,
    channel: str,
    record_type: str,
    challenge_id: str,
    representative: np.ndarray,
    mask: np.ndarray,
    cfg: Experiment2Config,
) -> dict[str, Any]:
    sigma = cfg.analysis.envelope_sigma_px
    eps = cfg.analysis.envelope_eps
    detail = local_ratio_detail(representative, sigma=sigma, eps_env=eps)
    return {
        "device_id": device_id,
        "channel": channel,
        "record_type": record_type,
        "challenge_id": challenge_id,
        "speckle_contrast": speckle_contrast(representative, mask=mask),
        "spatial_acf_width_px": estimate_speckle_width(representative, mask=mask),
        "local_ratio_contrast": speckle_contrast(detail, mask=mask, signed=True),
        "mean_intensity": float(representative[mask].mean()),
        "preprocessing_version": cfg.analysis.preprocessing_version,
    }


def run_validation(
    cfg: Experiment2Config,
    *,
    metadata_path: Path | None = None,
    check_files: bool = True,
    require_metadata: bool = True,
) -> dict[str, Any]:
    meta_path = metadata_path or cfg.metadata_path
    out_validation = cfg.output_dir / "validation"
    _ensure_dir(out_validation)

    challenge_report = validate_all_challenges(cfg)
    write_challenge_validation_csv(
        challenge_report,
        out_validation / "challenge_validation.csv",
    )
    write_challenge_validation_json(
        challenge_report,
        out_validation / "challenge_validation.json",
    )

    dark_report = validate_dark_reuse(cfg)

    metadata_report: MetadataValidationReport | None = None
    if meta_path.exists():
        records = load_metadata_csv(meta_path, cfg)
        metadata_report = validate_metadata(records, cfg, check_files=check_files)
        write_metadata_validation_json(
            metadata_report,
            out_validation / "metadata_validation.json",
        )
    elif require_metadata:
        metadata_report = MetadataValidationReport(
            valid=False,
            errors=[f"Metadata file not found: {meta_path}"],
        )
        write_metadata_validation_json(
            metadata_report,
            out_validation / "metadata_validation.json",
        )
    else:
        metadata_report = MetadataValidationReport(
            valid=True,
            warnings=[f"Metadata file not found (skipped): {meta_path}"],
        )

    metadata_ok = metadata_report.valid if metadata_report else False
    overall_valid = challenge_report.valid and dark_report.get("valid", False) and metadata_ok
    if cfg.quality_control.strict_mode and not overall_valid:
        errors = []
        if not challenge_report.valid:
            errors.extend(challenge_report.errors)
        if metadata_report and not metadata_report.valid:
            errors.extend(metadata_report.errors)
        raise ValueError("Validation failed in strict mode: " + "; ".join(errors))

    return {
        "valid": overall_valid,
        "challenge_validation": challenge_report.to_dict(),
        "dark_reuse_validation": dark_report,
        "metadata_validation": metadata_report.to_dict() if metadata_report else {},
    }


def run_analysis(
    cfg: Experiment2Config,
    *,
    metadata_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    meta_path = metadata_path or cfg.metadata_path
    _log_stage("[fixed_state] Validating config / challenges / metadata / dark...")
    validation = run_validation(
        cfg,
        metadata_path=meta_path,
        check_files=not dry_run,
        require_metadata=not dry_run,
    )

    if dry_run:
        logger.info("Dry-run complete: config, challenges, and dark reuse validated.")
        return {"dry_run": True, "validation": validation}

    _log_stage("[fixed_state] Validation OK. Loading metadata and dark templates...")
    records = load_metadata_csv(meta_path, cfg)
    meta_report = validate_metadata(records, cfg, check_files=True)
    if not meta_report.valid:
        raise ValueError(
            "Metadata validation failed: " + "; ".join(meta_report.errors)
        )

    out_dir = cfg.output_dir
    validation_dir = out_dir / "validation"
    templates_dir = out_dir / "templates"
    metrics_dir = out_dir / "metrics"
    masks_dir = templates_dir / "masks"
    block_dir = templates_dir / "block_templates"
    rep_dir = templates_dir / "representative_templates"
    for d in (validation_dir, templates_dir, metrics_dir, masks_dir, block_dir, rep_dir):
        _ensure_dir(d)

    dark_r, dark_g, dark_prov = _load_fixed_state_dark(cfg)
    dark_hash = dark_prov.dark_artifact_hash
    with (validation_dir / "dark_reuse_validation.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "valid": True,
                "provenance": provenance_to_dict(dark_prov),
            },
            handle,
            indent=2,
        )

    by_device: dict[str, dict[str, MetadataRecord]] = {}
    for rec in records:
        key = rec.challenge_id if rec.record_type == "green_challenge" else rec.record_type
        by_device.setdefault(rec.device_id, {})[key] = rec

    processed: dict[str, dict[str, ProcessedRecording]] = {}
    qc_rows: list[dict[str, Any]] = []
    dark_settings = cfg.dark_frame.camera_settings

    video_jobs: list[tuple[str, str, MetadataRecord]] = []
    for device_id in cfg.experiment.device_ids:
        for key, rec in by_device[device_id].items():
            video_jobs.append((device_id, key, rec))

    _log_stage(
        f"[fixed_state] Processing {len(video_jobs)} videos "
        f"({len(cfg.experiment.device_ids)} devices)..."
    )
    with mute_console_logging():
        pbar = stage_tqdm(
            video_jobs, desc="Processing videos", unit="video", total=len(video_jobs)
        )
        for device_id, key, rec in pbar:
            processed.setdefault(device_id, {})
            label = _record_label(device_id, key)
            video_path = _resolve_record_video(rec, cfg)
            dark = dark_r if rec.channel == "red" else dark_g
            mismatches = assert_camera_settings_match(
                dark_settings,
                _record_camera_settings(rec),
                strict=cfg.dark_frame.strict_parameter_match,
            )
            if mismatches and cfg.quality_control.strict_mode:
                raise ValueError(
                    f"Dark-parameter mismatch for {device_id}/{key}: "
                    + "; ".join(mismatches)
                )

            proc = process_video(
                video_path=video_path,
                channel=rec.channel,
                dark=dark,
                roi=cfg.camera.roi,
                cfg=cfg,
                device_id=device_id,
                record_type=rec.record_type,
                challenge_id=rec.challenge_id,
                session_id=rec.session_id,
                on_progress=make_video_progress(pbar, label),
            )
            processed[device_id][key] = proc
            qc_rows.append(
                {
                    "device_id": device_id,
                    "record_type": rec.record_type,
                    "channel": rec.channel,
                    "challenge_id": rec.challenge_id,
                    "frame_count": proc.n_frames,
                    "expected_frame_count": int(
                        round(cfg.acquisition.recording_duration_s * proc.fps)
                    ),
                    "effective_frame_count": len(proc.retained_indices),
                    "frame_rate_fps": proc.fps,
                    "frame_width_px": rec.frame_width_px,
                    "frame_height_px": rec.frame_height_px,
                    "exposure_ms": rec.exposure_ms,
                    "gain": rec.gain,
                    "saturation_fraction": proc.saturation_fraction,
                    "dark_source_path": dark_prov.dark_source_path,
                    "dark_artifact_hash": dark_hash,
                    "dark_parameter_match": len(mismatches) == 0,
                    "roi_valid": cfg.camera.roi is None or True,
                    "image_variance": float(np.var(proc.representative)),
                    "qc_pass": proc.saturation_fraction <= cfg.analysis.max_saturation_fraction,
                    "qc_flags": ";".join(
                        f for f in [
                            "saturation_high"
                            if proc.saturation_fraction > cfg.analysis.max_saturation_fraction
                            else "",
                            "dark_param_mismatch" if mismatches else "",
                        ]
                        if f
                    ),
                }
            )

    device_masks: dict[str, np.ndarray] = {}
    mask_meta: dict[str, dict[str, Any]] = {}
    _log_stage("[fixed_state] Building masks and computing device metrics...")
    with mute_console_logging():
        for device_id in stage_tqdm(
            cfg.experiment.device_ids,
            desc="Device metrics",
            unit="device",
            total=len(cfg.experiment.device_ids),
        ):
            mask, meta = _load_or_build_device_mask(
                device_id=device_id,
                red_before=processed[device_id]["red_before"],
                green_c01=processed[device_id][cfg.slm.representative_challenge_id],
                cfg=cfg,
                out_dir=masks_dir,
            )
            device_masks[device_id] = mask
            mask_meta[device_id] = meta
            for row in qc_rows:
                if row["device_id"] == device_id:
                    row["mask_coverage"] = meta["mask_coverage"]

    green_features: dict[str, dict[str, Any]] = {}
    red_features: dict[str, dict[str, Any]] = {}
    intra_rows: list[dict[str, Any]] = []
    inter_challenge_rows: list[dict[str, Any]] = []
    short_term_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    red_drift_rows: list[dict[str, Any]] = []

    rep_challenge = cfg.slm.representative_challenge_id

    with mute_console_logging():
        for device_id in stage_tqdm(
            cfg.experiment.device_ids,
            desc="Score construction",
            unit="device",
            total=len(cfg.experiment.device_ids),
        ):
            green_recs = {
                cid: processed[device_id][cid]
                for cid in cfg.slm.challenge_ids
            }
            green_feat = build_detail_cm_for_green_device(green_recs, cfg)
            green_features[device_id] = green_feat

            red_before = processed[device_id]["red_before"]
            red_after = processed[device_id]["red_after"]
            red_before_feat = build_detail_for_red_recording(red_before, cfg)
            red_after_feat = build_detail_for_red_recording(red_after, cfg)
            red_features[device_id] = {
                "red_before": red_before_feat,
                "red_after": red_after_feat,
            }

            mask = device_masks[device_id]
            mask_id = mask_meta[device_id]["mask_id"]
            session_id = by_device[device_id][rep_challenge].session_id

            for cid in cfg.slm.challenge_ids:
                intra_rows.extend(
                    compute_intra_scores(
                        device_id=device_id,
                        challenge_id=cid,
                        session_id=by_device[device_id][cid].session_id,
                        features=green_feat[cid],
                        mask=mask,
                        cfg=cfg,
                        dark_hash=dark_hash,
                        mask_id=mask_id,
                    )
                )

            inter_challenge_rows.extend(
                compute_inter_challenge_scores(
                    device_id=device_id,
                    session_id=session_id,
                    features_by_challenge=green_feat,
                    challenge_ids=cfg.slm.challenge_ids,
                    mask=mask,
                    cfg=cfg,
                    dark_hash=dark_hash,
                    mask_id=mask_id,
                )
            )

            red_short = compute_short_term_ncc(red_before_feat, mask)
            green_short = compute_short_term_ncc(green_feat[rep_challenge], mask)
            short_term_rows.append(
                {
                    "device_id": device_id,
                    "channel": "red",
                    "record_type": "red_before",
                    "challenge_id": "",
                    "mean_ncc": float(np.mean(red_short)),
                    "median_ncc": float(np.median(red_short)),
                    "minimum_ncc": float(np.min(red_short)),
                    "ncc_std": float(np.std(red_short)),
                    "num_pairs": len(red_short),
                }
            )
            short_term_rows.append(
                {
                    "device_id": device_id,
                    "channel": "green",
                    "record_type": "green_challenge",
                    "challenge_id": rep_challenge,
                    "mean_ncc": float(np.mean(green_short)),
                    "median_ncc": float(np.median(green_short)),
                    "minimum_ncc": float(np.min(green_short)),
                    "ncc_std": float(np.std(green_short)),
                    "num_pairs": len(green_short),
                }
            )

            spatial_rows.append(
                _spatial_metrics_row(
                    device_id=device_id,
                    channel="red",
                    record_type="red_before",
                    challenge_id="",
                    representative=red_before.representative,
                    mask=mask,
                    cfg=cfg,
                )
            )
            spatial_rows.append(
                _spatial_metrics_row(
                    device_id=device_id,
                    channel="green",
                    record_type="green_challenge",
                    challenge_id=rep_challenge,
                    representative=green_recs[rep_challenge].representative,
                    mask=mask,
                    cfg=cfg,
                )
            )

            drift = compute_red_drift_metrics(
                red_before,
                red_after,
                red_before_feat,
                red_after_feat,
                mask,
            )
            red_drift_rows.append({"device_id": device_id, **drift})
            for row in qc_rows:
                if row["device_id"] == device_id and row["record_type"] == "red_before":
                    row["red_drift_ncc"] = drift["red_drift_ncc"]

            np.save(rep_dir / f"{device_id}_red_before.npy", red_before.representative)
            for cid in cfg.slm.challenge_ids:
                np.save(
                    rep_dir / f"{device_id}_green_{cid}.npy",
                    green_recs[cid].representative,
                )
                for bi, block in enumerate(green_recs[cid].blocks, start=1):
                    np.save(
                        block_dir / f"{device_id}_{cid}_block{bi}.npy",
                        block,
                    )

    inter_device_rows: list[dict[str, Any]] = []
    _log_stage("[fixed_state] Computing inter-device scores...")
    with mute_console_logging():
        for cid in stage_tqdm(
            cfg.slm.challenge_ids,
            desc="Inter-device NCC",
            unit="challenge",
            total=len(cfg.slm.challenge_ids),
        ):
            device_feat = {
                d: green_features[d][cid] for d in cfg.experiment.device_ids
            }
            inter_device_rows.extend(
                compute_inter_device_scores(
                    challenge_id=cid,
                    device_features=device_feat,
                    device_masks=device_masks,
                    device_ids=cfg.experiment.device_ids,
                    cfg=cfg,
                    dark_hash=dark_hash,
                )
            )

    _log_stage("[fixed_state] Writing metrics...")
    intra_vals = np.asarray([r["ncc"] for r in intra_rows], dtype=np.float64)
    inter_challenge_vals = np.asarray(
        [r["ncc"] for r in inter_challenge_rows], dtype=np.float64
    )
    inter_device_vals = np.asarray(
        [r["ncc"] for r in inter_device_rows], dtype=np.float64
    )

    eer_a, thr_a = equal_error_rate_with_threshold(intra_vals, inter_challenge_vals)
    eer_b, thr_b = equal_error_rate_with_threshold(intra_vals, inter_device_vals)
    separation = {
        "task_a_intra_vs_inter_challenge": {
            "margin": margin(intra_vals, inter_challenge_vals),
            "robust_gap": robust_gap(intra_vals, inter_challenge_vals),
            "d_prime": d_prime(intra_vals, inter_challenge_vals),
            "auc": auc_roc(intra_vals, inter_challenge_vals),
            "eer": eer_a,
            "eer_threshold": thr_a,
        },
        "task_b_intra_vs_inter_device": {
            "margin": margin(intra_vals, inter_device_vals),
            "robust_gap": robust_gap(intra_vals, inter_device_vals),
            "d_prime": d_prime(intra_vals, inter_device_vals),
            "auc": auc_roc(intra_vals, inter_device_vals),
            "eer": eer_b,
            "eer_threshold": thr_b,
        },
    }

    red_medians = [r["median_ncc"] for r in short_term_rows if r["channel"] == "red"]
    green_medians = [r["median_ncc"] for r in short_term_rows if r["channel"] == "green"]
    t_stat, t_p = paired_ttest(
        np.asarray(red_medians), np.asarray(green_medians)
    )
    w_stat, w_p = wilcoxon_signed_rank(
        np.asarray(red_medians), np.asarray(green_medians)
    )
    separation["short_term_red_vs_green"] = {
        "paired_t_statistic": t_stat,
        "paired_t_pvalue": t_p,
        "wilcoxon_statistic": w_stat,
        "wilcoxon_pvalue": w_p,
    }

    _write_csv(metrics_dir / "spatial_metrics.csv", spatial_rows)
    _write_csv(metrics_dir / "short_term_ncc.csv", short_term_rows)
    _write_csv(metrics_dir / "intra_repeatability_scores.csv", intra_rows)
    _write_csv(metrics_dir / "inter_challenge_scores.csv", inter_challenge_rows)
    _write_csv(metrics_dir / "inter_device_scores.csv", inter_device_rows)
    _write_csv(metrics_dir / "red_drift_metrics.csv", red_drift_rows)
    _write_csv(validation_dir / "quality_control.csv", qc_rows)

    device_summary = []
    for device_id in cfg.experiment.device_ids:
        device_intra = [r["ncc"] for r in intra_rows if r["device_id_a"] == device_id]
        device_inter = [
            r["ncc"] for r in inter_challenge_rows if r["device_id_a"] == device_id
        ]
        device_summary.append(
            {
                "device_id": device_id,
                "median_intra_ncc": float(np.median(device_intra)),
                "median_inter_challenge_ncc": float(np.median(device_inter)),
                "mask_coverage": mask_meta[device_id]["mask_coverage"],
            }
        )
    _write_csv(metrics_dir / "device_level_summary.csv", device_summary)

    challenge_summary = []
    for cid in cfg.slm.challenge_ids:
        vals = [r["ncc"] for r in inter_device_rows if r["challenge_id_a"] == cid]
        challenge_summary.append(
            {
                "challenge_id": cid,
                "median_inter_device_ncc": float(np.median(vals)),
                "num_pairs": len(vals),
            }
        )
    _write_csv(metrics_dir / "challenge_level_summary.csv", challenge_summary)

    with (metrics_dir / "separation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(separation, handle, indent=2)

    _log_stage(f"[fixed_state] Done. Outputs written to {out_dir}")
    return {
        "output_dir": str(out_dir),
        "num_intra_scores": len(intra_rows),
        "num_inter_challenge_scores": len(inter_challenge_rows),
        "num_inter_device_scores": len(inter_device_rows),
        "separation_metrics": separation,
    }
