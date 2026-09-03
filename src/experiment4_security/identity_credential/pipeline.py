"""Orchestration: prepare / validate / formal with timestamped self-contained runs."""

from __future__ import annotations

import importlib
import json
import logging
import traceback
from pathlib import Path
from typing import Any

import yaml

from experiment4_security.identity_credential.cache_store import ensure_shared_cache_layout
from experiment4_security.identity_credential.config import IdentityCredentialConfig
from experiment4_security.identity_credential.credential_lifecycle import lifecycle_narrative
from experiment4_security.identity_credential.data_validation import (
    validate_dataset,
    write_validation_artifacts,
)
from experiment4_security.identity_credential.gates import GateError
from experiment4_security.identity_credential.joint_decision import run_track_j_structure
from experiment4_security.identity_credential.leakage_audit import leakage_audit_structure
from experiment4_security.identity_credential.red_conditioned_attack import run_track_e_structure
from experiment4_security.identity_credential.red_feature_adapter import assert_feature_contract
from experiment4_security.identity_credential.red_identity_evaluation import run_track_r0_r1
from experiment4_security.identity_credential.reporting import emit_prepare_artifacts
from experiment4_security.identity_credential.run_registry import (
    STATUS_SUCCESS,
    assert_validate_ready_for_formal,
    create_run,
    isoformat_local,
    sha256_file,
    sha256_text,
)
from experiment4_security.identity_credential.schemas import (
    EXPECTED_INVENTORY,
    MODE_FORMAL,
    MODE_PREPARE,
    N_AB_PAIRS_EXPECTED,
    N_GREEN_VIDEOS_EXPECTED,
    N_RED_VIDEOS_EXPECTED,
    STATUS_DATA_READY,
)

logger = logging.getLogger(__name__)

_REQUIRED_MODULES = [
    "formal.identity_credential.config",
    "formal.identity_credential.schemas",
    "formal.identity_credential.dataset_layout",
    "formal.identity_credential.manifest",
    "formal.identity_credential.data_validation",
    "formal.identity_credential.cache_adapter",
    "formal.identity_credential.cache_store",
    "formal.identity_credential.run_registry",
    "formal.identity_credential.red_feature_adapter",
    "formal.identity_credential.red_video_processing",
    "formal.identity_credential.red_capture_qc",
    "formal.identity_credential.red_identity_enrollment",
    "formal.identity_credential.red_identity_evaluation",
    "formal.identity_credential.green_result_adapter",
    "formal.identity_credential.green_multidevice_metrics",
    "formal.identity_credential.green_database_tracks",
    "formal.identity_credential.green_clone_tracks",
    "formal.identity_credential.threshold_freeze",
    "formal.identity_credential.joint_decision",
    "formal.identity_credential.credential_lifecycle",
    "formal.identity_credential.joint_episode_generation",
    "formal.identity_credential.red_conditioned_attack",
    "formal.identity_credential.attack_models",
    "formal.identity_credential.attack_evaluation",
    "formal.identity_credential.bootstrap",
    "formal.identity_credential.leakage_audit",
    "formal.identity_credential.reporting",
    "formal.identity_credential.cli",
    "formal.lifecycle.red_identity",
    "formal.ml_attack.batch_eval",
    "formal.ml_attack.track_a_database_auth",
    "formal.ml_attack.track_d_clone_transfer",
]


def check_code_readiness() -> dict[str, Any]:
    ok, failed = [], []
    for name in _REQUIRED_MODULES:
        try:
            importlib.import_module(name)
            ok.append(name)
        except Exception as exc:  # noqa: BLE001
            failed.append({"module": name, "error": f"{type(exc).__name__}: {exc}"})
    return {
        "ready": len(failed) == 0,
        "n_ok": len(ok),
        "n_failed": len(failed),
        "failed": failed,
        "red_feature_contract": assert_feature_contract(),
        "lifecycle_narrative": lifecycle_narrative(),
        "checked_at": isoformat_local(),
    }


def _run_inline_synthetic_checks() -> dict[str, Any]:
    from experiment4_security.identity_credential.dataset_layout import iter_expected_paths
    from experiment4_security.identity_credential.manifest import parse_green_filename, parse_red_filename

    checks = []
    try:
        assert len(iter_expected_paths(Path("/tmp"))) == EXPECTED_INVENTORY["n_total_videos"]
        checks.append({"name": "expected_manifest_len", "ok": True})
    except Exception as exc:  # noqa: BLE001
        checks.append({"name": "expected_manifest_len", "ok": False, "error": str(exc)})
    try:
        assert parse_green_filename("A_1_C001_M0_F01.mp4")["challenge_id"] == "C001"
        assert parse_red_filename("R_after_M7_F06.mp4")["phase"] == "after"
        checks.append({"name": "filename_parsers", "ok": True})
    except Exception as exc:  # noqa: BLE001
        checks.append({"name": "filename_parsers", "ok": False, "error": str(exc)})
    try:
        assert assert_feature_contract()["dim"] == 9
        checks.append({"name": "red_9d_contract", "ok": True})
    except Exception as exc:  # noqa: BLE001
        checks.append({"name": "red_9d_contract", "ok": False, "error": str(exc)})
    return {
        "status": "PASS" if all(c.get("ok") for c in checks) else "FAIL",
        "checks": checks,
        "note": "Synthetic/layout checks only — not formal science",
    }


def _write_config_resolved(ctx, cfg: IdentityCredentialConfig) -> str:
    text = yaml.safe_dump(cfg.to_resolved_dict(), sort_keys=True)
    ctx.write_text("config_resolved.yaml", text)
    return sha256_text(text)


def run_prepare(cfg: IdentityCredentialConfig, *, synthetic_report: dict[str, Any] | None = None) -> dict[str, Any]:
    ensure_shared_cache_layout(cfg.output_root)
    ctx = create_run(
        cfg.output_root,
        "prepare",
        project_root=cfg.project_root,
        formal_scientific_run=False,
    )
    manifest = ctx.base_manifest()
    try:
        ctx.log("prepare start")
        config_hash = _write_config_resolved(ctx, cfg)
        readiness = check_code_readiness()
        if not readiness["ready"]:
            raise GateError(f"code_readiness failed: {readiness['failed']}")
        validation = validate_dataset(cfg.videos_root)
        write_validation_artifacts(validation, ctx.run_dir)
        # Canonical names required by run protocol
        if (ctx.run_dir / "found_videos_manifest.csv").exists():
            (ctx.run_dir / "found_videos_manifest.csv").replace(ctx.run_dir / "data_manifest.csv")
        elif (ctx.run_dir / "expected_dataset_manifest.csv").exists():
            # keep expected as expected_*; also copy summary stub to data_manifest if no found
            validation["expected"].to_csv(ctx.run_dir / "data_manifest.csv", index=False)
        audit = validation["audit"]
        ctx.write_json("data_audit.json", audit)
        syn = synthetic_report or _run_inline_synthetic_checks()
        emit_prepare_artifacts(
            prepared_dir=ctx.run_dir,
            project_root=cfg.project_root,
            cfg_resolved=cfg.to_resolved_dict(),
            data_audit=audit,
            code_readiness=readiness,
            synthetic_report=syn,
        )
        status = audit["data_status"]
        r01 = run_track_r0_r1({}, mode=MODE_PREPARE, data_status=status)
        j = run_track_j_structure(mode=MODE_PREPARE, data_status=status)
        e = run_track_e_structure(mode=MODE_PREPARE, data_status=status)
        leak = leakage_audit_structure(mode=MODE_PREPARE, data_status=status)
        track_struct = {
            "R0_R1": {k: v for k, v in r01.items() if k != "pairs"},
            "J": j,
            "E": e,
            "leakage": leak,
        }
        ctx.write_json("track_structures.json", track_struct)
        ctx.write_json(
            "metrics_summary.json",
            {
                "formal_scientific_run": False,
                "metrics_emitted": False,
                "note": "prepare run — no formal scientific metrics",
            },
        )
        manifest.update(
            {
                "config_hash": config_hash,
                "dataset_manifest_hash": sha256_file(ctx.run_dir / "data_manifest.csv"),
                "data_status": status,
                "code_ready": True,
            }
        )
        ctx.finalize_success(manifest)
        ctx.log("prepare SUCCESS")
        return {
            "mode": MODE_PREPARE,
            "run_id": ctx.run_id,
            "run_dir": str(ctx.run_dir),
            "status": STATUS_SUCCESS,
            "data_status": status,
            "code_ready": True,
            "formal_scientific_run": False,
        }
    except Exception as exc:
        ctx.log(f"prepare FAILED: {exc}")
        ctx.log(traceback.format_exc())
        ctx.finalize_failed(manifest, str(exc))
        raise


def run_validate(cfg: IdentityCredentialConfig, *, strict: bool = False) -> dict[str, Any]:
    ensure_shared_cache_layout(cfg.output_root)
    ctx = create_run(
        cfg.output_root,
        "validate",
        project_root=cfg.project_root,
        formal_scientific_run=False,
    )
    manifest = ctx.base_manifest(strict=strict)
    try:
        ctx.log(f"validate start strict={strict}")
        config_hash = _write_config_resolved(ctx, cfg)
        validation = validate_dataset(cfg.videos_root)
        write_validation_artifacts(validation, ctx.run_dir)
        if (ctx.run_dir / "found_videos_manifest.csv").exists():
            (ctx.run_dir / "found_videos_manifest.csv").replace(ctx.run_dir / "data_manifest.csv")
        else:
            validation["expected"].to_csv(ctx.run_dir / "data_manifest.csv", index=False)
        audit = validation["audit"]
        ctx.write_json("data_audit.json", audit)
        ctx.write_json(
            "metrics_summary.json",
            {
                "formal_scientific_run": False,
                "metrics_emitted": False,
                "note": "validate run — inventory only, no paper conclusions",
            },
        )
        ok = audit["data_status"] == STATUS_DATA_READY
        if strict:
            ok = ok and (
                audit.get("n_green_found") == N_GREEN_VIDEOS_EXPECTED
                and audit.get("n_red_found") == N_RED_VIDEOS_EXPECTED
                and audit.get("n_ab_pairs") == N_AB_PAIRS_EXPECTED
                and audit.get("n_missing", 1) == 0
                and audit.get("n_state_conflicts", 1) == 0
                and len(audit.get("devices_present", [])) >= 6
            )
        manifest.update(
            {
                "config_hash": config_hash,
                "dataset_manifest_hash": sha256_file(ctx.run_dir / "data_manifest.csv"),
                "data_status": audit["data_status"],
                "strict": strict,
                "strict_pass": ok if strict else None,
            }
        )
        if strict and not ok:
            raise GateError(f"strict validate failed; audit={ {k: audit.get(k) for k in ['data_status','n_green_found','n_red_found','n_ab_pairs','n_missing','n_state_conflicts']} }")
        ctx.finalize_success(manifest)
        ctx.log("validate SUCCESS")
        out = dict(audit)
        out["run_id"] = ctx.run_id
        out["run_dir"] = str(ctx.run_dir)
        out["status"] = STATUS_SUCCESS
        return out
    except Exception as exc:
        ctx.log(f"validate FAILED: {exc}")
        ctx.log(traceback.format_exc())
        ctx.finalize_failed(manifest, str(exc))
        raise


def run_formal(cfg: IdentityCredentialConfig, *, require_latest_validate: bool = True) -> dict[str, Any]:
    """Formal full run — requires SUCCESS validate with DATA_READY; delegates to formal_run."""
    from experiment4_security.identity_credential.formal_run import run_formal_pipeline

    parent = None
    if require_latest_validate:
        parent = assert_validate_ready_for_formal(cfg.output_root, strict=True)
    return run_formal_pipeline(
        cfg,
        validated_by_run_id=(parent["latest"]["run_id"] if parent else None),
        parent_validate_audit=(parent["audit"] if parent else None),
    )
