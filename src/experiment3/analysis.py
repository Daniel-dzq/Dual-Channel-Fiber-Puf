"""End-to-end Experiment 3 analysis orchestration."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from experiment3.config import Experiment3Config, config_to_dict
from experiment3.progress import make_video_progress, stage_tqdm
from experiment3.green_features import (
    build_detail_cm_for_green_group,
    green_features_to_dataframe,
)
from experiment3.red_features import extract_red_features_for_recording, features_to_dataframe
from experiment3.reporting import write_full_report, write_scoped_reports
from experiment3.score_construction import (
    construct_green_scores,
    construct_red_scores,
    evaluate_s0_gallery_queries,
)
from experiment3.statistics import (
    build_scope_metric_pack,
    development_readiness_gate,
    frozen_confirmation,
    per_device_metrics,
    per_state_metrics,
    select_threshold_eer,
)
from experiment3.splits import (
    SCOPE_ALL_COHORT_DESCRIPTIVE,
    SCOPE_DEVELOPMENT_ONLY,
    SCOPE_FROZEN_FORMAL,
    annotate_metadata_roles,
    assert_disjoint,
    require_nonempty_dev,
    require_nonempty_frozen,
)
from experiment3.artifacts import (
    save_standardizer_artifacts,
    save_threshold_artifacts,
    write_frozen_protocol_bundle,
)
from experiment3.leakage_audit import (
    audit_common_groups_from_features,
    run_leakage_audit,
    write_leakage_audit,
)
from experiment3.validation import ValidationResult, validate_inputs
from experiment3.video_processing import (
    ProcessedRecording,
    load_dark_channel,
    process_video,
)
from puf_common.masks import build_valid_mask_from_refs

LOG = logging.getLogger("threshold_development.analysis")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_validation(
    cfg: Experiment3Config,
    *,
    metadata_path: Path | None = None,
    check_files: bool = True,
) -> dict[str, Any]:
    result = validate_inputs(
        cfg,
        metadata_path=metadata_path,
        check_files=check_files,
        probe_videos=check_files,
    )
    return {
        "valid": result.valid,
        "is_pilot": result.is_pilot,
        "inventory": result.inventory,
        "missing": result.missing,
        "duplicates": result.duplicates,
        "extras": result.extras,
        "errors": result.errors,
        "warnings": result.warnings,
        "development_devices": result.development_devices,
        "frozen_test_devices_present": result.frozen_test_devices_present,
        "n_qc_rows": len(result.qc_rows),
    }


def _row_label(row: Any) -> str:
    if getattr(row, "video_filename", None):
        return Path(row.video_filename).name
    return Path(row.video_path).name


def _process_one_row(
    row: Any,
    cfg: Experiment3Config,
    dark_cache: dict[str, tuple[np.ndarray | None, str]],
    *,
    on_progress: Any | None = None,
) -> tuple[ProcessedRecording | None, dict[str, Any]]:
    """Decode one metadata row. Returns (recording_or_None, qc_row)."""
    channel = row.channel
    path = Path(row.video_path)
    if channel not in dark_cache:
        dark_cache[channel] = load_dark_channel(cfg, channel, reference_shape=None)
    dark, dark_prov = dark_cache[channel]
    base_qc = {
        "device_id": row.device_id,
        "state_id": row.state_id,
        "channel": channel,
        "round_id": getattr(row, "round_id", "") or "",
        "challenge_id": getattr(row, "challenge_id", "") or "",
        "video_path": str(path),
        "dark_provenance": dark_prov,
    }
    try:
        rec = process_video(
            video_path=path,
            channel=channel,
            dark=dark,
            cfg=cfg,
            device_id=row.device_id,
            state_id=row.state_id,
            round_id=getattr(row, "round_id", "") or "",
            challenge_id=getattr(row, "challenge_id", "") or "",
            on_progress=on_progress,
        )
    except Exception as exc:  # noqa: BLE001
        return None, {
            **base_qc,
            "status": "fail",
            "reason": f"processing_error:{exc}",
        }
    status = "fail" if rec.qc_flags else "ok"
    return rec, {
        **base_qc,
        "status": status,
        "reason": "|".join(rec.qc_flags) if rec.qc_flags else "ok",
        "saturation_fraction": rec.saturation_fraction,
        "mean_intensity": rec.mean_intensity,
        "variance": rec.variance,
        "n_frames": rec.n_frames,
        "fps": rec.fps,
        "width": rec.width,
        "height": rec.height,
    }


def _build_fixed_mask_streaming(
    metadata: pd.DataFrame,
    cfg: Experiment3Config,
    run_dir: Path,
    dark_cache: dict[str, tuple[np.ndarray | None, str]],
    *,
    development_devices: list[str],
    frozen_devices: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build one fixed mask from development enrollment refs only."""
    from experiment3.artifacts import save_mask_artifacts

    if not development_devices:
        raise RuntimeError(
            "Development device set is empty. Cannot fit valid mask."
        )
    enroll = cfg.experiment.enrollment_state
    c0 = cfg.experiment.challenge_ids[0]
    dev_set = set(development_devices)
    frozen_set = set(frozen_devices)
    if not dev_set.isdisjoint(frozen_set):
        raise RuntimeError("Mask fit: development and frozen device sets overlap")

    ref_rows = metadata[
        metadata["device_id"].isin(dev_set)
        & (
            (
                (metadata["channel"] == "red")
                & (metadata["state_id"] == enroll)
            )
            | (
                (metadata["channel"] == "green")
                & (metadata["state_id"] == enroll)
                & (metadata["round_id"] == "A")
                & (metadata["challenge_id"] == c0)
            )
        )
    ]
    if ref_rows.empty:
        raise RuntimeError(
            "Could not build valid mask: no development enrollment references. "
            "Refusing metadata.head fallback and refusing frozen enrollment refs."
        )
    leak = sorted(set(ref_rows["device_id"]) & frozen_set)
    if leak:
        raise RuntimeError(f"Frozen devices in mask fit refs: {leak}")

    refs: list[np.ndarray] = []
    source_files: list[str] = []
    rows = list(ref_rows.itertuples(index=False))
    pbar = stage_tqdm(rows, desc="Mask references (dev S0)", unit="video")
    for row in pbar:
        label = _row_label(row)
        rec, _qc = _process_one_row(
            row,
            cfg,
            dark_cache,
            on_progress=make_video_progress(pbar, label),
        )
        if rec is not None and rec.blocks:
            refs.append(np.asarray(rec.blocks[0], dtype=np.float32))
            source_files.append(str(row.video_path))
        del rec

    if not refs:
        raise RuntimeError("Could not build valid mask: no reference templates decoded")

    mask, cov, thr = build_valid_mask_from_refs(
        refs,
        abs_floor=cfg.analysis.valid_mask_abs_floor,
        percentile=cfg.analysis.valid_mask_percentile,
    )
    del refs
    meta = {
        "source_devices": sorted(set(ref_rows["device_id"])),
        "source_states": [enroll],
        "source_rounds": ["enrollment_A_or_red"],
        "source_files": source_files,
        "generation_method": "build_valid_mask_from_refs",
        "percentile_parameters": {
            "abs_floor": cfg.analysis.valid_mask_abs_floor,
            "percentile": cfg.analysis.valid_mask_percentile,
            "threshold_used": float(thr),
            "coverage": float(cov),
        },
        "contains_frozen_data": False,
    }
    assert set(meta["source_devices"]).issubset(dev_set)
    assert set(meta["source_devices"]).isdisjoint(frozen_set)
    assert meta["contains_frozen_data"] is False
    paths = save_mask_artifacts(mask, run_dir, metadata=meta)
    meta["sha256"] = paths["sha256"]
    LOG.info(
        "Fixed valid mask (dev-only) coverage=%.4f threshold=%.3f -> %s",
        cov,
        thr,
        paths.get("valid_mask.png"),
    )
    return mask, meta


def _extract_red_packs_streaming(
    metadata: pd.DataFrame,
    mask: np.ndarray,
    cfg: Experiment3Config,
    dark_cache: dict[str, tuple[np.ndarray | None, str]],
    qc_extra: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    red_meta = metadata[metadata["channel"] == "red"]
    packs: list[dict[str, Any]] = []
    rows = list(red_meta.itertuples(index=False))
    pbar = stage_tqdm(rows, desc="Red features", unit="video")
    for row in pbar:
        label = _row_label(row)
        rec, qc = _process_one_row(
            row,
            cfg,
            dark_cache,
            on_progress=make_video_progress(pbar, label),
        )
        qc_extra.append(qc)
        if rec is None:
            continue
        packs.append(extract_red_features_for_recording(rec, mask, cfg))
        del rec
    return packs


def _extract_green_feats_streaming(
    metadata: pd.DataFrame,
    cfg: Experiment3Config,
    dark_cache: dict[str, tuple[np.ndarray | None, str]],
    qc_extra: list[dict[str, Any]],
) -> dict:
    """Decode green videos group-by-group so peak RAM stays near one device×state×round."""
    green_meta = metadata[metadata["channel"] == "green"].copy()
    group_cols = ["device_id", "state_id", "round_id"]
    group_keys = (
        green_meta[group_cols]
        .drop_duplicates()
        .sort_values(group_cols)
        .itertuples(index=False, name=None)
    )
    out: dict = {}
    pbar = stage_tqdm(list(group_keys), desc="Green detail_cm", unit="group")
    for device, state, rnd in pbar:
        pbar.set_postfix_str(f"{device}/{state}/{rnd}", refresh=False)
        sub = green_meta[
            (green_meta["device_id"] == device)
            & (green_meta["state_id"] == state)
            & (green_meta["round_id"] == rnd)
        ]
        by_cid: dict[str, ProcessedRecording] = {}
        for row in sub.itertuples(index=False):
            label = _row_label(row)
            rec, qc = _process_one_row(row, cfg, dark_cache)
            qc_extra.append(qc)
            if rec is None:
                continue
            by_cid[rec.challenge_id] = rec
        if not by_cid:
            continue
        feats = build_detail_cm_for_green_group(
            by_cid,
            cfg,
            device_id=device,
            state_id=state,
            round_id=rnd,
        )
        for cid, feat in feats.items():
            out[(device, state, rnd, cid)] = feat
        del by_cid, feats
    return out


def run_analysis(
    cfg: Experiment3Config,
    *,
    metadata_path: Path | None = None,
    dry_run: bool = False,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    np.random.seed(cfg.statistics.random_seed)
    validation = validate_inputs(
        cfg,
        metadata_path=metadata_path,
        check_files=True,
        probe_videos=True,
    )
    base_out = Path(output_dir) if output_dir is not None else cfg.output_dir
    run_dir = base_out / "runs" / _timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)

    validation.metadata.to_csv(run_dir / "validated_metadata.csv", index=False)
    with (run_dir / "run_config_snapshot.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config_to_dict(cfg), fh, sort_keys=False)

    qc_df = pd.DataFrame(validation.qc_rows)
    if dry_run:
        qc_df.to_csv(run_dir / "qc_report.csv", index=False)
        summary = {
            "dry_run": True,
            "run_dir": str(run_dir),
            "valid": validation.valid,
            "is_pilot": validation.is_pilot,
            "inventory": validation.inventory,
            "warnings": validation.warnings,
            "errors": validation.errors,
            "diagnostics": {
                "RED_IDENTITY": "inconclusive",
                "GREEN_SAME_STATE_RELIABILITY": "inconclusive",
                "GREEN_STATE_REVOCATION": "inconclusive",
                "FULL_IDENTITY_STATE_STORY": "fix acquisition or processing before continuing",
            },
        }
        write_full_report(run_dir, summary, validation=validation)
        return summary

    if not validation.valid and cfg.quality_control.strict_mode:
        raise RuntimeError(f"Validation failed: {validation.errors}")

    # Stream decode: never retain all 2048×1536 templates in RAM at once.
    dark_cache: dict[str, tuple[np.ndarray | None, str]] = {}
    qc_extra: list[dict[str, Any]] = []

    dev_devices = list(validation.development_devices)
    frozen_devices = list(validation.frozen_test_devices_present)
    require_nonempty_dev(dev_devices)
    assert_disjoint(dev_devices, frozen_devices)
    require_for_formal = not validation.is_pilot and not cfg.splits.pilot_development_only
    require_nonempty_frozen(frozen_devices, require_for_formal=require_for_formal)

    validation.metadata = annotate_metadata_roles(
        validation.metadata,
        development_devices=dev_devices,
        frozen_devices=frozen_devices,
    )
    validation.metadata.to_csv(run_dir / "validated_metadata.csv", index=False)

    mask, mask_meta = _build_fixed_mask_streaming(
        validation.metadata,
        cfg,
        run_dir,
        dark_cache,
        development_devices=dev_devices,
        frozen_devices=frozen_devices,
    )

    red_packs = _extract_red_packs_streaming(
        validation.metadata, mask, cfg, dark_cache, qc_extra
    )
    if not red_packs:
        raise RuntimeError("No red recordings processed successfully")
    red_feat_df = features_to_dataframe(red_packs)
    red_feat_df.to_csv(run_dir / "red_features.csv", index=False)

    green_feats = _extract_green_feats_streaming(
        validation.metadata, cfg, dark_cache, qc_extra
    )
    if not green_feats:
        raise RuntimeError("No green recordings processed successfully")
    green_feat_df = green_features_to_dataframe(green_feats)
    green_feat_df.to_csv(run_dir / "green_features.csv", index=False)

    qc_df = pd.DataFrame(qc_extra)
    qc_df.to_csv(run_dir / "qc_report.csv", index=False)
    n_recordings_ok = int((qc_df["status"] == "ok").sum()) if not qc_df.empty else 0
    if n_recordings_ok == 0:
        raise RuntimeError("No recordings processed successfully")

    # Sample inclusion manifest (QC only; never auth-score based)
    inclusion_rows = []
    for r in qc_df.itertuples(index=False):
        role = (
            "development"
            if r.device_id in dev_devices
            else ("frozen_test" if r.device_id in frozen_devices else "unknown")
        )
        inclusion_rows.append(
            {
                "device": r.device_id,
                "state": r.state_id,
                "round": getattr(r, "round_id", "") or "",
                "video_id": Path(r.video_path).name,
                "dataset_role": role,
                "included": r.status == "ok",
                "exclusion_reason": "" if r.status == "ok" else r.reason,
                "qc_rule": r.reason,
                "qc_rule_predefined": True,
                "auth_score_used_for_qc": False,
            }
        )
    inclusion_df = pd.DataFrame(inclusion_rows)
    inclusion_df.to_csv(run_dir / "sample_inclusion_manifest.csv", index=False)

    # Acquisition manifest (file-based only; no old/new mixing claims)
    acq = validation.metadata.copy()
    acq_out = pd.DataFrame(
        {
            "device": acq["device_id"],
            "state": acq["state_id"],
            "channel": acq["channel"],
            "round": acq.get("round_id", ""),
            "challenge": acq.get("challenge_id", ""),
            "video_filename": acq["video_filename"],
            "video_path": acq["video_path"],
            "dataset_role": acq["dataset_role"],
            "source": "metadata_or_directory_discovery",
            "old_new_mixing": False,
            "score_based_selection": False,
            "post_hoc_replacement": False,
        }
    )
    acq_out.to_csv(run_dir / "acquisition_manifest.csv", index=False)

    red_scores, red_meta = construct_red_scores(
        red_packs, development_devices=dev_devices, cfg=cfg
    )
    std_art = save_standardizer_artifacts(
        run_dir,
        mu=np.asarray(red_meta["mu"], dtype=float),
        sd=np.asarray(red_meta["sd"], dtype=float),
        metadata={
            "fit_devices": red_meta["fit_devices"],
            "fit_states": red_meta["fit_states"],
            "fit_rounds": red_meta["fit_rounds"],
            "feature_names": "RED_LOWDIM / LOWDIM_NAMES",
            "sample_count": red_meta["n_fit"],
            "contains_frozen_data": False,
        },
    )
    assert set(red_meta["fit_devices"]).issubset(set(dev_devices))
    assert set(red_meta["fit_devices"]).isdisjoint(set(frozen_devices))
    assert set(red_meta["fit_states"]) == {"S0"}

    query_df = evaluate_s0_gallery_queries(
        red_meta["standardized"],
        enrollment_state=cfg.experiment.enrollment_state,
        query_states=list(cfg.experiment.query_states),
    )
    green_scores = construct_green_scores(
        green_feats,
        mask,
        state_order=list(cfg.experiment.state_ids),
    )

    # Annotate scores with evaluation helpers
    role_map = {d: "development" for d in dev_devices}
    role_map.update({d: "frozen_test" for d in frozen_devices})
    for df_ in (red_scores, green_scores):
        if not df_.empty:
            df_["dataset_role_a"] = df_["device_id_a"].map(role_map)
            df_["dataset_role_b"] = df_["device_id_b"].map(role_map)
    if not query_df.empty:
        query_df["dataset_role"] = query_df["query_device"].map(role_map)

    all_scores = pd.concat([red_scores, green_scores], ignore_index=True, sort=False)
    all_scores.to_csv(run_dir / "all_scores.csv", index=False)
    all_scores.to_csv(run_dir / "all_cohort_descriptive_score_table.csv", index=False)

    # Thresholds from development data only
    red_dev = red_scores[
        red_scores["device_id_a"].isin(dev_devices)
        & red_scores["device_id_b"].isin(dev_devices)
    ]
    green_dev = green_scores[
        green_scores["device_id_a"].isin(dev_devices)
        & green_scores["device_id_b"].isin(dev_devices)
    ]
    threshold_fit_devices = sorted(
        set(red_dev["device_id_a"]).union(set(red_dev["device_id_b"]))
        if not red_dev.empty
        else set(dev_devices)
    )
    tau_r = cfg.thresholds.tau_R
    if tau_r is None:
        tau_r = select_threshold_eer(
            red_dev.loc[red_dev["group"] == "same_device_diff_state", "score"].to_numpy(),
            red_dev.loc[red_dev["group"] == "diff_device", "score"].to_numpy(),
        )
    tau_g = cfg.thresholds.tau_G
    if tau_g is None:
        tau_g = select_threshold_eer(
            green_dev.loc[
                green_dev["group"] == "same_device_same_state_same_challenge", "score"
            ].to_numpy(),
            green_dev.loc[
                green_dev["group"] == "same_device_same_state_diff_challenge", "score"
            ].to_numpy(),
        )
    thr_art = save_threshold_artifacts(
        run_dir,
        tau_r=float(tau_r),
        tau_g=float(tau_g),
        source={
            "selection_mode": cfg.thresholds.selection_mode,
            "fit_devices": threshold_fit_devices,
            "note": "Operational thresholds fixed on development cohort only.",
        },
    )

    all_devices = sorted(set(dev_devices) | set(frozen_devices))
    pack_kwargs = dict(
        red_scores=red_scores,
        green_scores=green_scores,
        query_df=query_df,
        tau_r=float(tau_r),
        tau_g=float(tau_g),
        bootstrap_iterations=cfg.statistics.bootstrap_iterations,
        random_seed=cfg.statistics.random_seed,
        permutation_iterations=cfg.statistics.permutation_iterations,
    )

    dev_pack = build_scope_metric_pack(
        devices=dev_devices,
        evaluation_scope=SCOPE_DEVELOPMENT_ONLY,
        claim_label="DEVELOPMENT / EXPLORATORY / NOT AN INDEPENDENT TEST",
        include_bootstrap=True,
        include_permutation=True,
        **pack_kwargs,
    )
    frozen_pack = None
    if frozen_devices:
        frozen_pack = build_scope_metric_pack(
            devices=frozen_devices,
            evaluation_scope=SCOPE_FROZEN_FORMAL,
            claim_label="FROZEN TEST / FORMAL HELD-OUT EVALUATION",
            include_bootstrap=True,
            include_permutation=True,
            **pack_kwargs,
        )
    pooled_pack = build_scope_metric_pack(
        devices=all_devices,
        evaluation_scope=SCOPE_ALL_COHORT_DESCRIPTIVE,
        claim_label="ALL-COHORT DESCRIPTIVE / NOT HELD-OUT / NOT USED FOR FORMAL CLAIMS",
        include_bootstrap=False,
        include_permutation=False,
        **pack_kwargs,
    )

    # Gates
    readiness = development_readiness_gate(
        query_df=dev_pack["query_df"],
        green_scores=dev_pack["green_scores"],
        system=dev_pack["system_metrics"],
        n_devices=len(dev_devices),
    )
    confirmation = None
    if frozen_pack is not None:
        confirmation = frozen_confirmation(
            query_df=frozen_pack["query_df"],
            green_scores=frozen_pack["green_scores"],
            system=frozen_pack["system_metrics"],
            n_devices=len(frozen_devices),
        )
    pooled_descriptive_diagnostics = {
        "note": "Descriptive overview only; proceed is intentionally null.",
        "proceed": None,
        "n_devices": len(all_devices),
    }

    # Persist scoped tables/metrics
    def _write_scope(name: str, pack: dict[str, Any]) -> None:
        scores = pd.concat(
            [pack["red_scores"], pack["green_scores"]], ignore_index=True, sort=False
        )
        scores["evaluation_scope"] = pack["evaluation_scope"]
        scores.to_csv(run_dir / f"{name}_score_table.csv", index=False)
        pack["query_df"].to_csv(run_dir / f"{name}_decision_table.csv", index=False)
        payload = {
            "red_metrics": pack["red_metrics"],
            "green_metrics": pack["green_metrics"],
            "system_metrics": pack["system_metrics"],
            "evaluation_scope": pack["evaluation_scope"],
            "claim_label": pack["claim_label"],
            "devices": pack["devices"],
            "tau_R_dev": float(tau_r),
            "tau_G_dev": float(tau_g),
        }
        with (run_dir / f"{name}_metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)

    _write_scope("development", dev_pack)
    if frozen_pack is not None:
        _write_scope("frozen_test", frozen_pack)
        if frozen_pack.get("bootstrap"):
            with (run_dir / "frozen_bootstrap_summary.json").open("w", encoding="utf-8") as fh:
                json.dump(frozen_pack["bootstrap"], fh, indent=2, default=str)
            # Flat CSV of bootstrap metric CIs
            rows = []
            for k, v in (frozen_pack["bootstrap"].get("metrics") or {}).items():
                rows.append({"metric": k, **v})
            pd.DataFrame(rows).to_csv(run_dir / "frozen_device_bootstrap.csv", index=False)
    _write_scope("all_cohort_descriptive", pooled_pack)

    # Deprecated aliases (compat) — point to scoped files, not formal by default
    deprecated_note = {
        "deprecated": True,
        "message": (
            "red_metrics.json / green_metrics.json / system_metrics.json are compatibility "
            "aliases. Formal claims must use frozen_test_metrics.json when available; "
            "otherwise development_metrics.json (not held-out)."
        ),
        "prefer": (
            "frozen_test_metrics.json"
            if frozen_pack is not None
            else "development_metrics.json"
        ),
    }
    compat_red = (
        frozen_pack["red_metrics"] if frozen_pack is not None else dev_pack["red_metrics"]
    )
    compat_green = (
        frozen_pack["green_metrics"]
        if frozen_pack is not None
        else dev_pack["green_metrics"]
    )
    compat_sys = (
        frozen_pack["system_metrics"]
        if frozen_pack is not None
        else dev_pack["system_metrics"]
    )
    with (run_dir / "red_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump({**compat_red, **deprecated_note}, fh, indent=2, default=str)
    with (run_dir / "green_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump({**compat_green, **deprecated_note}, fh, indent=2, default=str)
    with (run_dir / "system_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump({**compat_sys, **deprecated_note}, fh, indent=2, default=str)
    LOG.warning(
        "Deprecated compat metrics written; prefer frozen_test_metrics.json / "
        "development_metrics.json / all_cohort_descriptive_metrics.json"
    )

    per_dev = per_device_metrics(red_scores, green_scores, query_df)
    per_st = per_state_metrics(green_scores, query_df)
    per_dev.to_csv(run_dir / "per_device_metrics.csv", index=False)
    per_st.to_csv(run_dir / "per_state_metrics.csv", index=False)

    # Gallery / query identity sets for audit
    enroll = cfg.experiment.enrollment_state
    gallery_keys = {(d, s) for (d, s) in red_meta["standardized"] if s == enroll}
    query_keys = {(d, s) for (d, s) in red_meta["standardized"] if s in cfg.experiment.query_states}
    gallery_videos = set(
        validation.metadata.loc[
            (validation.metadata["channel"] == "red")
            & (validation.metadata["state_id"] == enroll),
            "video_filename",
        ].astype(str)
    )
    query_videos = set(
        validation.metadata.loc[
            (validation.metadata["channel"] == "red")
            & (validation.metadata["state_id"].isin(cfg.experiment.query_states)),
            "video_filename",
        ].astype(str)
    )
    common_groups = audit_common_groups_from_features(
        list(green_feats.keys()), role_map
    )

    audit = run_leakage_audit(
        development_devices=dev_devices,
        frozen_devices=frozen_devices,
        threshold_fit_devices=threshold_fit_devices,
        standardizer_fit_devices=list(red_meta["fit_devices"]),
        standardizer_fit_states=list(red_meta["fit_states"]),
        mask_fit_devices=list(mask_meta["source_devices"]),
        formal_metric_devices=list(frozen_devices) if frozen_pack else [],
        development_metric_devices=list(dev_devices),
        pooled_metric_devices=list(all_devices),
        readiness_gate_devices=list(dev_devices),
        frozen_confirmation_devices=list(frozen_devices) if confirmation else [],
        gallery_states=[enroll],
        query_states=list(cfg.experiment.query_states),
        gallery_keys=gallery_keys,
        query_keys=query_keys,
        gallery_video_ids=gallery_videos,
        query_video_ids=query_videos,
        common_groups=common_groups,
        qc_auth_score_used=False,
        unsafe_fallback_triggered=False,
        pooled_labeled_formal=False,
        development_labeled_held_out=False,
        frozen_eer_used_as_operational=False,
        acquisition_has_false_mixing_claim=False,
        mask_contains_frozen=bool(mask_meta.get("contains_frozen_data")),
        standardizer_contains_frozen=bool(red_meta.get("contains_frozen_data")),
        pooled_proceed_is_none=pooled_descriptive_diagnostics.get("proceed") is None,
    )
    # When frozen is required, formal_metrics_frozen_only must match
    if require_for_formal and not frozen_devices:
        audit["overall_status"] = "FAIL"
        audit["allow_formal_report"] = False
    write_leakage_audit(run_dir, audit)

    protocol = {
        "development_devices": dev_devices,
        "frozen_devices": frozen_devices,
        "gallery_states": [enroll],
        "query_states": list(cfg.experiment.query_states),
        "mask_source": "development_enrollment_only",
        "standardizer_source": "development_S0_only",
        "threshold_source": "development_eer",
        "tau_R_dev": float(tau_r),
        "tau_G_dev": float(tau_g),
        "detail_sigma": cfg.analysis.envelope_sigma_px,
        "detail_eps": cfg.analysis.envelope_eps,
        "common_subtraction_grouping": "device_x_state_x_round",
        "bootstrap_unit": "device",
        "report_scopes": [
            SCOPE_DEVELOPMENT_ONLY,
            SCOPE_FROZEN_FORMAL,
            SCOPE_ALL_COHORT_DESCRIPTIVE,
        ],
    }
    write_frozen_protocol_bundle(
        run_dir,
        protocol=protocol,
        artifact_hashes={
            "valid_mask": mask_meta.get("sha256", ""),
            "red_standardizer": std_art.get("sha256", ""),
            "tau_R": thr_art.get("sha256_tau_R", ""),
            "tau_G": thr_art.get("sha256_tau_G", ""),
        },
    )

    split_manifest = pd.DataFrame(
        [
            {"device": d, "dataset_role": "development", "split": "development"}
            for d in dev_devices
        ]
        + [
            {"device": d, "dataset_role": "frozen_test", "split": "frozen_test"}
            for d in frozen_devices
        ]
    )
    split_manifest.to_csv(run_dir / "split_manifest.csv", index=False)

    formal_allowed = bool(audit.get("allow_formal_report")) and frozen_pack is not None
    claim_label = (
        "FROZEN TEST / FORMAL HELD-OUT EVALUATION"
        if formal_allowed
        else (
            "DEVELOPMENT / EXPLORATORY / NOT AN INDEPENDENT TEST"
            if validation.is_pilot or frozen_pack is None
            else "LEAKAGE AUDIT FAILED — FORMAL REPORT BLOCKED"
        )
    )

    summary = {
        "dry_run": False,
        "run_dir": str(run_dir),
        "valid": validation.valid,
        "is_pilot": validation.is_pilot,
        "inventory": validation.inventory,
        "warnings": validation.warnings,
        "errors": validation.errors,
        "development_devices": dev_devices,
        "frozen_test_devices_present": frozen_devices,
        "development_metrics": {
            "red": dev_pack["red_metrics"],
            "green": dev_pack["green_metrics"],
            "system": dev_pack["system_metrics"],
        },
        "frozen_test_metrics": None
        if frozen_pack is None
        else {
            "red": frozen_pack["red_metrics"],
            "green": frozen_pack["green_metrics"],
            "system": frozen_pack["system_metrics"],
        },
        "all_cohort_descriptive_metrics": {
            "red": pooled_pack["red_metrics"],
            "green": pooled_pack["green_metrics"],
            "system": pooled_pack["system_metrics"],
        },
        # Deprecated keys kept for older readers — NEVER treat as formal by default
        "red_metrics": {**compat_red, **deprecated_note},
        "green_metrics": {**compat_green, **deprecated_note},
        "system_metrics": {**compat_sys, **deprecated_note},
        "diagnostics": readiness,
        "development_readiness_gate": readiness,
        "frozen_confirmation": confirmation,
        "all_cohort_descriptive_diagnostics": pooled_descriptive_diagnostics,
        "tau_R": float(tau_r),
        "tau_G": float(tau_g),
        "tau_R_dev": float(tau_r),
        "tau_G_dev": float(tau_g),
        "n_recordings": n_recordings_ok,
        "claim_label": claim_label,
        "leakage_audit": audit,
        "mask_metadata": mask_meta,
        "standardizer_fit_devices": red_meta["fit_devices"],
        "standardizer_fit_states": red_meta["fit_states"],
        "formal_report_allowed": formal_allowed,
    }

    write_scoped_reports(
        run_dir,
        summary,
        validation=validation,
        formal_allowed=formal_allowed,
    )
    # Compatibility full_report.md
    write_full_report(run_dir, summary, validation=validation, query_df=query_df)
    return summary
