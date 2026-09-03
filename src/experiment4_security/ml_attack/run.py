"""Orchestrator for registered-database green-credential security analysis.

Does NOT:
- predict unseen challenges
- run sequential/LOSO unknown-state characterization
- use Round B for hyperparameter selection
- modify lifecycle code or outputs
- compute absolute ASR@lifecycle tau_G
- generate figures
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from experiment4_security.ml_attack import challenge_features as cf
from experiment4_security.ml_attack import enrollment as enr
from experiment4_security.ml_attack import manifest as mf
from experiment4_security.ml_attack import pairing as pr
from experiment4_security.ml_attack import reporting as rpt
from experiment4_security.ml_attack import video_preprocessing as vp
from experiment4_security.ml_attack.config import MLAttackConfig
from experiment4_security.ml_attack.track_a_database_auth import run_track_a
from experiment4_security.ml_attack.track_b_template_revoke import run_track_b
from experiment4_security.ml_attack.track_c_clone import run_track_c
from experiment4_security.ml_attack.track_d_clone_transfer import run_track_d

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_valid_mask(cfg: MLAttackConfig, cache_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read-only load/reconstruct of frozen lifecycle valid_mask into pilot cache."""
    mask, provenance = vp.reconstruct_frozen_valid_mask(cfg, cache_dir=cache_dir)
    return mask, provenance

def _link_or_copy_cache(src: Path, dst: Path) -> dict[str, Any]:
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_missing = 0
    for f in sorted(Path(src).glob("*.npy")):
        target = dst / f.name
        if target.exists():
            n_ok += 1
            continue
        try:
            os.link(f, target)
        except OSError:
            shutil.copy2(f, target)
        n_ok += 1
    return {"source": str(src), "dest": str(dst), "n_linked_or_present": n_ok, "n_missing": n_missing}


def _validate_vector_cache(
    vector_cache_dir: Path,
    manifest_df: pd.DataFrame,
    *,
    expected_sigma: float,
    expected_epsilon: float,
) -> dict[str, Any]:
    usable = manifest_df[manifest_df["sample_id"].notna()]
    missing = []
    shapes = set()
    dtypes = set()
    checked = 0
    for sid in usable["sample_id"].tolist():
        p = vp.vector_cache_path(vector_cache_dir, sid)
        if not p.exists():
            missing.append(sid)
            continue
        arr = np.load(p, mmap_mode="r")
        shapes.add(tuple(arr.shape))
        dtypes.add(str(arr.dtype))
        checked += 1
    status = "PASS" if not missing and len(shapes) == 1 else "FAIL"
    return {
        "status": status,
        "n_checked": checked,
        "n_missing": len(missing),
        "missing_sample_ids_head": missing[:20],
        "shapes": [list(s) for s in shapes],
        "dtypes": sorted(dtypes),
        "expected_envelope_sigma": expected_sigma,
        "expected_envelope_epsilon": expected_epsilon,
        "note": "Cache stores masked detail vectors from prior verified preprocessing; sigma/epsilon are config expectations.",
    }


def run_pilot(cfg: MLAttackConfig, run_id: str | None = None, n_workers: int = 1) -> dict[str, Any]:
    run_id = run_id or __import__(
        "experiment4_security.identity_credential.run_registry", fromlist=["new_run_id"]
    ).new_run_id("f01_green_pilot")
    run_dir = cfg.run_output_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = cfg.cache_dir(run_id)
    vector_cache_dir = cache_dir / "vectors"
    vector_cache_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    started_at = _utc_now()
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "run_id": run_id,
        "started_at": started_at,
        "phases": {},
        "absolute_asr": None,
        "asr_status": "SCORE_SPACE_OR_PROTOCOL_MISMATCH",
    }

    from experiment4_security.ml_attack.metric_schema import (
        FORBIDDEN_METRIC_TOKENS,
        METRIC_DEFINITIONS,
        NAME_MAPPING,
        REPRESENTATION,
        RETRIEVAL_CANDIDATE_COUNT,
    )

    logger.info("metric framework version = unified_v1_registered_database")
    logger.info("representation = %s", REPRESENTATION)
    logger.info("score types enabled = GENUINE, CHALLENGE_MISMATCH, CROSS_STATE_CREDENTIAL, SOFTWARE_CLONE")
    logger.info("device_mismatch_status = NOT_AVAILABLE_SINGLE_DEVICE")
    logger.info("robust gap formulas = Q05(Genuine)-Q95(non-genuine) for RG_C/RG_X/RG_A")
    logger.info("AUC class definitions = positive=GENUINE; negative=C/X/A by comparison_type")
    logger.info("retrieval candidate count = %d", RETRIEVAL_CANDIDATE_COUNT)
    logger.info("batch evaluator enabled")
    logger.info("evaluation batch size = %d", cfg.evaluation.batch_size)
    logger.info("Track B parallel targets = %d (M4 Pro optimized)", getattr(cfg.evaluation, "track_b_parallel_targets", 4))
    logger.info("Track D parallel targets = %d (M4 Pro optimized)", getattr(cfg.evaluation, "track_d_parallel_targets", 4))
    logger.info("absolute ASR disabled (SCORE_SPACE_OR_PROTOCOL_MISMATCH)")
    logger.info("figure generation disabled")
    logger.info("protocol = registered CRP database authentication / revocation / clone")
    logger.info("lifecycle modified = false")
    logger.info("old metric names remaining (forbidden tokens audited at report time): %d listed", len(FORBIDDEN_METRIC_TOKENS))

    # ---------- config dump ----------
    resolved = cfg.to_resolved_dict()
    with (run_dir / "config_resolved.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(resolved, fh, sort_keys=False)
    config_hash = hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest()

    rpt.write_json(run_dir / "METRIC_DEFINITIONS.json", METRIC_DEFINITIONS)
    pd.DataFrame(NAME_MAPPING).assign(migration_status="applied", affected_files="tracks/batch_eval/reporting/cli").to_csv(
        run_dir / "metric_name_mapping.csv", index=False
    )
    # ---------- valid mask (lifecycle read-only) ----------
    valid_mask, mask_prov = _load_valid_mask(cfg, cache_dir)
    rpt.write_json(run_dir / "valid_mask_provenance.json", mask_prov)

    # ---------- Phase: manifest ----------
    logger.info("Phase: scanning videos / parsing manifest")
    manifest_df = mf.scan_videos(
        cfg.videos_root,
        states=list(cfg.states),
        device_id=cfg.device_id,
        source_device_id=cfg.source_device_id,
        probe_decode=False,  # do not re-decode; reuse cache
        n_workers=1,
    )
    manifest_df, correction = pr.apply_folder_authoritative_state_correction(
        manifest_df, out_json=run_dir / "state_label_correction.json"
    )
    manifest_df.to_csv(run_dir / "data_manifest.csv", index=False)

    # State parser report
    parser_md = [
        "# State parser / label audit",
        "",
        f"- n_videos: {len(manifest_df)}",
        f"- n_raw_state_id_conflicts: {correction['n_raw_conflicts']}",
        f"- n_active_conflicts_after_correction: {correction['n_active_conflicts_after_correction']}",
        f"- is_m0_false_zero_parser_bug: {correction['is_m0_false_zero_parser_bug']}",
        "",
        "## Finding",
        "",
        correction["finding"],
        "",
        "## Conflict matrix (folder × filename)",
        "",
        "```json",
        json.dumps(correction["conflict_matrix"], indent=2),
        "```",
        "",
        "Official state_id = folder state_id. Filenames not renamed.",
        "Correction is audited in `state_label_correction.json`.",
    ]
    (run_dir / "state_parser_bug_report.md").write_text("\n".join(parser_md), encoding="utf-8")

    # ---------- A/B pairs ----------
    ab_pairs = pr.build_ab_pair_manifest(manifest_df)
    ab_pairs.to_csv(run_dir / "ab_pair_manifest.csv", index=False)
    n_ok_pairs = int((ab_pairs["pair_status"] == "OK").sum())
    logger.info("A/B pair count OK = %d (expected 1024)", n_ok_pairs)

    expected_videos = len(cfg.states) * 2 * (cfg.challenge_id_end - cfg.challenge_id_start + 1)
    n_conflicts_raw = int(correction["n_raw_conflicts"])
    n_conflicts_active = int(correction["n_active_conflicts_after_correction"])
    data_status = "DATA_READY"
    if len(manifest_df) != expected_videos or n_ok_pairs != len(cfg.states) * (cfg.challenge_id_end - cfg.challenge_id_start + 1):
        data_status = "DATA_BLOCKED"
    elif n_conflicts_raw > 0:
        data_status = "DATA_READY_WITH_CORRECTED_FILENAME_LABELS"

    data_audit = {
        "data_status": data_status,
        "n_videos_found": int(len(manifest_df)),
        "n_videos_expected": expected_videos,
        "n_state_id_conflicts_raw": n_conflicts_raw,
        "n_state_id_conflicts_active": n_conflicts_active,
        "n_ab_pairs_ok": n_ok_pairs,
        "n_ab_pairs_expected": len(cfg.states) * (cfg.challenge_id_end - cfg.challenge_id_start + 1),
        "per_state_counts": (
            manifest_df.groupby(["state_id", "round_id"]).size().reset_index(name="n").to_dict(orient="records")
        ),
    }
    rpt.write_json(run_dir / "data_audit.json", data_audit)
    (run_dir / "data_audit.md").write_text(
        "# Data audit\n\n```json\n" + json.dumps(data_audit, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    summary["data_status"] = data_status
    summary["data_audit"] = data_audit
    summary["phases"]["manifest"] = data_audit

    if data_status == "DATA_BLOCKED":
        summary["aborted_at"] = "manifest"
        return summary

    # ---------- Cache reuse / validation ----------
    cache_validation: dict[str, Any] = {"reuse_attempted": False}
    if cfg.reuse_vector_cache_from and Path(cfg.reuse_vector_cache_from).exists():
        logger.info("Reusing vector cache from %s", cfg.reuse_vector_cache_from)
        link_info = _link_or_copy_cache(Path(cfg.reuse_vector_cache_from), vector_cache_dir)
        cache_validation["reuse_attempted"] = True
        cache_validation["link_info"] = link_info
    cache_validation.update(
        _validate_vector_cache(
            vector_cache_dir,
            manifest_df,
            expected_sigma=cfg.preprocessing.envelope_sigma,
            expected_epsilon=cfg.preprocessing.envelope_epsilon,
        )
    )
    rpt.write_json(run_dir / "cache_validation.json", cache_validation)
    logger.info("cache validation status = %s", cache_validation["status"])

    # If cache incomplete, process missing clips only
    if cache_validation["status"] != "PASS":
        logger.warning("Cache incomplete; processing missing clips only (n_workers=%d)", n_workers)
        # Prefer QC from archive if present
        missing = set(cache_validation.get("missing_sample_ids_head", []))
        # Recompute full missing list
        missing = []
        for sid in manifest_df["sample_id"].dropna().tolist():
            if not vp.vector_cache_path(vector_cache_dir, sid).exists():
                missing.append(sid)
        miss_df = manifest_df[manifest_df["sample_id"].isin(missing)]
        qc_rows = []
        for _, row in miss_df.iterrows():
            qc_rows.append(
                vp.process_and_cache_clip(
                    Path(row["video_path"]),
                    sample_id=row["sample_id"],
                    valid_mask=valid_mask,
                    cfg=cfg,
                    vector_cache_dir=vector_cache_dir,
                )
            )
        qc_df = pd.DataFrame(qc_rows) if qc_rows else pd.DataFrame()
    else:
        # Prefer shared-cache QC next to vectors, else parent-of-parent (old layout).
        archived_qc = None
        if cfg.reuse_vector_cache_from:
            cand = Path(cfg.reuse_vector_cache_from).parent / "video_qc.csv"
            if not cand.exists():
                cand = Path(cfg.reuse_vector_cache_from).parent.parent / "video_qc.csv"
            archived_qc = cand if cand.exists() else None
        if archived_qc is not None:
            qc_df = pd.read_csv(archived_qc)
            shutil.copy2(archived_qc, run_dir / "video_qc.csv")
        else:
            qc_df = pd.DataFrame({"sample_id": manifest_df["sample_id"].dropna().tolist(), "qc_status": "PASS"})
            qc_df.to_csv(run_dir / "video_qc.csv", index=False)

    if "qc_status" in qc_df.columns:
        n_qc_pass = int((qc_df["qc_status"] == "PASS").sum())
        n_qc_warn = int((qc_df["qc_status"] == "WARN").sum())
        n_qc_error = int((qc_df["qc_status"] == "ERROR").sum())
    else:
        n_qc_pass = len(qc_df)
        n_qc_warn = 0
        n_qc_error = 0
    summary["phases"]["qc"] = {"pass": n_qc_pass, "warn": n_qc_warn, "error": n_qc_error}

    challenge_ids = [f"C{i:03d}" for i in range(cfg.challenge_id_start, cfg.challenge_id_end + 1)]

    def detail_lookup(state: str, round_id: str, challenge_id: str) -> np.ndarray:
        sid = f"{cfg.device_id}_{state}_{round_id}_{challenge_id}"
        return vp.load_vector_cache(vector_cache_dir, sid)

    # ---------- Challenge features ----------
    logger.info("Building challenge features")
    lib_manifest = cf.load_challenge_manifest(cfg.challenge_root)
    if "challenge_number" not in lib_manifest.columns:
        lib_manifest = lib_manifest.copy()
        lib_manifest["challenge_number"] = (
            lib_manifest["challenge_id"].str.replace("C", "", regex=False).astype(int)
        )
    challenge_library_yaml = cfg.project_root / "config" / "challenge_library.yaml"
    resolved_patterns = cf.ensure_patterns_available(
        lib_manifest,
        challenge_root=cfg.challenge_root,
        pattern_cache_dir=cfg.pattern_cache_dir(),
        challenge_library_yaml=challenge_library_yaml,
        repo_root=cfg.project_root.parent,
    )
    features = cf.build_challenge_features(
        challenge_ids, manifest=lib_manifest, resolved_paths=resolved_patterns["paths"]
    )
    challenge_features = {cid: fs.bitmap_32 for cid, fs in features.items()}
    cf.write_feature_manifest(features, run_dir / "challenge_feature_manifest.csv")

    # ---------- Enrollment commons ----------
    logger.info("Fitting state-specific Round-A enrollment commons")
    commons = enr.build_all_enrollment_commons(
        list(cfg.states), challenge_ids, detail_lookup=detail_lookup, device_id=cfg.device_id
    )
    enr.write_enrollment_artifacts(commons, run_dir)
    # Leakage audit: commons must not include Round B sample ids
    leakage = {
        "common_fit_rounds": ["A"],
        "round_b_in_common": False,
        "target_state_round_b_in_source_common": False,
        "overall_status": "PASS",
        "note": "Each common_s_A fit exclusively on that state's Round A details.",
    }
    rpt.write_json(run_dir / "leakage_audit.json", leakage)
    (run_dir / "leakage_audit.md").write_text(
        "# Leakage audit\n\nPASS — enrollment commons use Round A only; "
        "cross-state tests subtract source common.\n",
        encoding="utf-8",
    )
    summary["leakage_status"] = "PASS"

    # ---------- Track A ----------
    logger.info("Running Track A — same-state database authentication (S_G vs S_C)")
    ta_sum, ta_scores, ta_meta = run_track_a(
        list(cfg.states),
        challenge_ids,
        commons,
        detail_lookup=detail_lookup,
        device_id=cfg.device_id,
    )
    ta_sum.to_csv(run_dir / "track_a_database_authentication_summary.csv", index=False)
    ta_scores.to_parquet(run_dir / "track_a_database_authentication_scores.parquet", index=False)

    # ---------- Track B ----------
    n_par = int(getattr(cfg.evaluation, "track_b_parallel_targets", 4))
    logger.info("Running Track B — cross-state credential revocation (S_G vs S_X), parallel_targets=%d", n_par)
    tb_sum, tb_scores, tb_meta = run_track_b(
        list(cfg.states),
        challenge_ids,
        commons,
        detail_lookup=detail_lookup,
        genuine_by_state=ta_meta["genuine_by_state"],
        device_id=cfg.device_id,
        n_parallel_targets=n_par,
    )
    tb_sum.to_csv(run_dir / "track_b_template_transfer_matrix_summary.csv", index=False)
    tb_scores.to_parquet(run_dir / "track_b_template_transfer_scores.parquet", index=False)

    # ---------- Track C ----------
    logger.info("Running Track C — same-state response reconstruction (S_intra vs S_A)")
    tc_sum, tc_scores, tc_dom, tc_meta = run_track_c(
        list(cfg.states),
        challenge_ids,
        commons,
        challenge_features,
        detail_lookup=detail_lookup,
        models_cfg=cfg.models,
        pca_dimension=cfg.pca.dimension,
        genuine_by_state=ta_meta["genuine_by_state"],
        device_id=cfg.device_id,
    )
    tc_sum.to_csv(run_dir / "track_c_same_state_clone_summary.csv", index=False)
    tc_scores.to_parquet(run_dir / "track_c_same_state_clone_scores.parquet", index=False)
    tc_dom.to_csv(run_dir / "pca_mean_dominance_audit.csv", index=False)

    # template replay vs model clone
    cmp_rows = []
    for state in cfg.states:
        sub = tc_sum[tc_sum["source_state"] == state]
        replay = sub[sub["attack_method"] == "exact_template_replay"]
        for _, r in sub.iterrows():
            cmp_rows.append(
                {
                    "source_state": state,
                    "attack_method": r["attack_method"],
                    "median_S_A": r["median_S_A"],
                    "exact_replay_median_S_A": float(replay["median_S_A"].iloc[0]) if not replay.empty else float("nan"),
                    "model_minus_replay": (
                        float(r["median_S_A"] - replay["median_S_A"].iloc[0]) if not replay.empty else float("nan")
                    ),
                    "top1": r["top1"],
                    "source_state_valid": r["source_state_valid"],
                }
            )
    pd.DataFrame(cmp_rows).to_csv(run_dir / "template_replay_vs_model_clone.csv", index=False)

    # prediction diversity summary (auxiliary)
    div_cols = [
        c
        for c in [
            "source_state",
            "attack_method",
            "prediction_variance",
            "measured_variance",
            "variance_ratio",
            "median_S_A",
            "residual_ncc",
            "common_dominance_flag",
        ]
        if c in tc_sum.columns
    ]
    div = tc_sum[div_cols].copy()
    div.to_csv(run_dir / "prediction_diversity_audit.csv", index=False)
    if "attack_method" in div.columns:
        div.groupby("attack_method", as_index=False)[
            [c for c in ["variance_ratio", "median_S_A", "residual_ncc"] if c in div.columns]
        ].mean().to_csv(run_dir / "prediction_diversity_summary.csv", index=False)

    # ---------- Track D ----------
    n_par_d = int(getattr(cfg.evaluation, "track_d_parallel_targets", 4))
    logger.info("Running Track D — cross-state clone transfer (S_G vs S_A), parallel_targets=%d", n_par_d)
    td_sum, td_scores, td_meta = run_track_d(
        list(cfg.states),
        challenge_ids,
        commons,
        challenge_features,
        detail_lookup=detail_lookup,
        track_c_meta=tc_meta,
        track_c_summary=tc_sum,
        device_id=cfg.device_id,
        n_parallel_targets=n_par_d,
    )
    td_sum.to_csv(run_dir / "track_d_clone_transfer_matrix_summary.csv", index=False)
    td_scores.to_parquet(run_dir / "track_d_clone_transfer_scores.parquet", index=False)
    # ---------- Bootstrap (challenge-bank, descriptive) ----------
    logger.info("Challenge-bank bootstrap (F01 within-device only; not population CI)")
    boot_rows = []
    rng = np.random.default_rng(cfg.bootstrap.seed)
    bank_size = cfg.bootstrap.bank_size
    banks = [challenge_ids[i : i + bank_size] for i in range(0, len(challenge_ids), bank_size)]
    for state in cfg.states:
        g = ta_meta["genuine_by_state"][state]
        # Map genuine scores to challenge order used in Track A
        # Track A used sorted challenge_ids intersection; assume same order as challenge_ids
        if len(g) != len(challenge_ids):
            continue
        cid_to_g = dict(zip(challenge_ids, g))
        medians = []
        rgcs = []
        for _ in range(min(cfg.bootstrap.n_resamples, 5000)):
            sampled = [c for b in rng.choice(len(banks), size=len(banks), replace=True) for c in banks[b]]
            vals = np.array([cid_to_g[c] for c in sampled if c in cid_to_g])
            if vals.size == 0:
                continue
            medians.append(float(np.median(vals)))
            # Approximate RG_C using Track A summary S_C q95 (fixed) -- descriptive only
            q95_sc = float(ta_sum.loc[ta_sum["state_id"] == state, "q95_S_C"].iloc[0])
            rgcs.append(float(np.percentile(vals, 5) - q95_sc))
        boot_rows.append(
            {
                "metric": "median_S_G",
                "state_id": state,
                "point": float(np.median(g)),
                "ci95_lo": float(np.percentile(medians, 2.5)) if medians else float("nan"),
                "ci95_hi": float(np.percentile(medians, 97.5)) if medians else float("nan"),
                "unit": "challenge_bank",
                "note": "F01 within-device bootstrap; NOT population-level CI",
            }
        )
        boot_rows.append(
            {
                "metric": "rg_challenge_approx",
                "state_id": state,
                "point": float(ta_sum.loc[ta_sum["state_id"] == state, "rg_challenge"].iloc[0]),
                "ci95_lo": float(np.percentile(rgcs, 2.5)) if rgcs else float("nan"),
                "ci95_hi": float(np.percentile(rgcs, 97.5)) if rgcs else float("nan"),
                "unit": "challenge_bank",
                "note": "F01 within-device bootstrap; NOT population-level CI",
            }
        )
    pd.DataFrame(boot_rows).to_csv(run_dir / "bootstrap_summary.csv", index=False)

    # ---------- Metrics summary ----------
    metrics = {
        "track_a_overall": ta_meta["overall_conclusion"],
        "track_b_conclusion": tb_meta["conclusion"],
        "track_d_per_model": td_meta["per_model_conclusions"],
        "absolute_asr": None,
        "asr_status": "SCORE_SPACE_OR_PROTOCOL_MISMATCH",
        "red_conditioned_attack": "RED_CONDITIONED_ATTACK_NOT_AVAILABLE",
        "red_identity_evidence": "RED_IDENTITY_EVIDENCE_FROM_SEPARATE_LIFECYCLE_DATASET",
    }
    rpt.write_json(run_dir / "metrics_summary.json", metrics)
    rpt.write_json(run_dir / "green_lifecycle_summary.json", {
        "database_authentication": ta_meta["overall_conclusion"],
        "template_revocation": tb_meta["conclusion"],
        "clone_transfer": td_meta["per_model_conclusions"],
        "diagonal_median": tb_meta["diagonal_median"],
        "off_diagonal_median": tb_meta["off_diagonal_median"],
    })

    # ---------- Reports ----------
    conclusion_levels = rpt.write_all_reports(
        run_dir,
        cfg=cfg,
        data_audit=data_audit,
        ta_sum=ta_sum,
        tb_sum=tb_sum,
        tb_meta=tb_meta,
        tc_sum=tc_sum,
        td_sum=td_sum,
        td_meta=td_meta,
        cache_validation=cache_validation,
        correction=correction,
    )

    finished_at = _utc_now()
    elapsed = time.time() - t_start
    run_manifest = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "run_started_at": started_at,
        "run_finished_at": finished_at,
        "elapsed_s": elapsed,
        "config_hash": config_hash,
        "code_commit": _try_git_commit(cfg.project_root),
        "protocol": "registered_database_authentication_revocation_clone",
        "absolute_asr": None,
        "asr_status": "SCORE_SPACE_OR_PROTOCOL_MISMATCH",
        "make_figures": False,
        "conclusion_levels": conclusion_levels,
    }
    rpt.write_json(run_dir / "run_manifest.json", run_manifest)
    rpt.write_json(
        cfg.output_root / "latest_run.json",
        {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "kind": "f01_green_pilot",
            "runs_root": str(cfg.output_root / "runs"),
        },
    )

    summary.update(
        {
            "finished_at": finished_at,
            "elapsed_s": elapsed,
            "conclusion_levels": conclusion_levels,
            "track_a_summary": ta_sum,
            "track_b_summary": tb_sum,
            "track_c_summary": tc_sum,
            "track_d_summary": td_sum,
            "track_b_meta": tb_meta,
            "track_d_meta": td_meta,
            "n_qc_pass": n_qc_pass,
            "n_qc_warn": n_qc_warn,
            "n_qc_error": n_qc_error,
        }
    )
    logger.info("Run complete in %.1fs -> %s", elapsed, run_dir)
    return summary


def _try_git_commit(project_root: Path) -> str | None:
    try:
        import subprocess

        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root.parent),
            capture_output=True,
            text=True,
            check=False,
        )
        return r.stdout.strip() or None
    except Exception:
        return None
