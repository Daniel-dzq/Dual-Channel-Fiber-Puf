"""Formal multi-device run orchestration (executes only when DATA_READY).

Scientific formulas delegated to ml_attack + lifecycle adapters.
Performance: M4 Pro defaults from IdentityCredentialConfig.performance.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiment4_security.identity_credential.attack_models import list_attack_models
from experiment4_security.identity_credential.cache_adapter import (
    green_sample_id,
    make_detail_lookup,
    red_sample_id,
    save_red_vector,
    vector_dir,
)
from experiment4_security.identity_credential.config import IdentityCredentialConfig
from experiment4_security.identity_credential.data_validation import (
    validate_dataset,
    write_validation_artifacts,
)
from experiment4_security.identity_credential.gates import (
    GateError,
    assert_not_writing_protected,
    require_data_ready_for_formal,
)
from experiment4_security.identity_credential.green_clone_tracks import run_green_tracks_cd_multidevice
from experiment4_security.identity_credential.green_database_tracks import run_green_tracks_ab_multidevice
from experiment4_security.identity_credential.joint_decision import run_track_j_structure
from experiment4_security.identity_credential.joint_episode_generation import build_episodes_from_manifests
from experiment4_security.identity_credential.leakage_audit import leakage_audit_structure
from experiment4_security.identity_credential.red_conditioned_attack import run_track_e_structure
from experiment4_security.identity_credential.red_feature_adapter import extract_red_vector
from experiment4_security.identity_credential.red_identity_enrollment import (
    apply_all,
    fit_red_standardizer_on_development,
)
from experiment4_security.identity_credential.red_identity_evaluation import run_track_r0_r1
from experiment4_security.identity_credential.red_video_processing import process_red_video
from experiment4_security.identity_credential.schemas import MODE_FORMAL, challenge_ids
from experiment4_security.ml_attack import challenge_features as cf
from experiment4_security.ml_attack import video_preprocessing as vp
from experiment4_security.ml_attack.config import ModelsConfig

logger = logging.getLogger(__name__)

_AB_CHECKPOINT_FILES = (
    "track_a_database_authentication_summary.csv",
    "track_b_template_transfer_matrix_summary.csv",
    "track_s_d_device_mismatch_summary.csv",
)


def _find_ab_checkpoint(output_root: Path) -> Path | None:
    """Newest formal run that already wrote Track A/B/S_D summaries (resume C/D from there)."""
    runs = Path(output_root) / "runs"
    if not runs.is_dir():
        return None
    candidates: list[Path] = []
    for d in runs.iterdir():
        if not d.is_dir() or not d.name.endswith("_formal"):
            continue
        if all((d / name).is_file() for name in _AB_CHECKPOINT_FILES):
            candidates.append(d)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _canonical_formal_run_id(output_root: Path) -> str | None:
    """Optional pointer: outputs/.../security/canonical_formal_run.json → {run_id}."""
    pointer = Path(output_root) / "canonical_formal_run.json"
    if not pointer.is_file():
        return None
    try:
        payload = json.loads(pointer.read_text())
    except json.JSONDecodeError:
        return None
    rid = payload.get("run_id")
    return str(rid) if rid else None


def _find_green_vector_resume_checkpoint(output_root: Path) -> Path | None:
    """Incomplete formal (no Track A yet) — resume green-vector phase in the canonical run dir."""
    runs = Path(output_root) / "runs"
    if not runs.is_dir():
        return None
    rid = _canonical_formal_run_id(output_root)
    if rid:
        d = runs / rid
        if d.is_dir() and not (d / _AB_CHECKPOINT_FILES[0]).is_file():
            return d
    candidates: list[Path] = []
    for d in runs.iterdir():
        if not d.is_dir() or not d.name.endswith("_formal"):
            continue
        if (d / _AB_CHECKPOINT_FILES[0]).is_file():
            continue
        vdir = d / "_cache" / "vectors"
        if vdir.is_dir() and any(vdir.glob("*.npy")):
            candidates.append(d)
    if not candidates:
        return None
    return min(candidates, key=lambda p: p.name)


def _recompute_genuine_by_device_state(
    devices: list[str],
    states: list[str],
    cids: list[str],
    detail_lookups: dict[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    """Track A only — needed for Track C reference scores when resuming past A/B."""
    from experiment4_security.identity_credential.green_result_adapter import build_device_commons
    from experiment4_security.ml_attack.track_a_database_auth import run_track_a

    out: dict[str, dict[str, np.ndarray]] = {}
    for device_id in devices:
        logger.info("[formal resume] Track A genuine scores for %s", device_id)
        commons = build_device_commons(
            device_id, states, cids, detail_lookup=detail_lookups[device_id]
        )
        _sum, _scores, ta_meta = run_track_a(
            states,
            cids,
            commons,
            detail_lookup=detail_lookups[device_id],
            device_id=device_id,
        )
        out[device_id] = ta_meta["genuine_by_state"]
    return out


def _ensure_green_vectors(
    found: pd.DataFrame,
    *,
    run_dir: Path,
    mask: np.ndarray,
    n_workers: int,
    ml_cfg: Any,
    shared_vector_dirs: list[Path] | None = None,
) -> None:
    """Process missing green clips into vector cache (parallel).

    Reuses identical sample_id vectors from shared/prior caches via hardlink/symlink
    when present (same frozen mask + preprocessing protocol).
    """
    vdir = vector_dir(run_dir)
    reuse_dirs = [Path(p) for p in (shared_vector_dirs or []) if Path(p).is_dir()]
    green = found[(found["channel"] == "green") & (found["parse_ok"])].copy()
    jobs = []
    n_reused = 0
    for _, r in green.iterrows():
        sid = green_sample_id(r["device_id"], r["state_id"], r["round_id"], r["challenge_id"])
        out = vdir / f"{sid}.npy"
        if out.exists():
            continue
        reused = False
        for src_dir in reuse_dirs:
            src = src_dir / f"{sid}.npy"
            if not src.is_file():
                continue
            try:
                out.hardlink_to(src)
            except OSError:
                try:
                    out.symlink_to(src.resolve())
                except OSError:
                    continue
            n_reused += 1
            reused = True
            break
        if not reused:
            jobs.append((sid, Path(r["abspath"])))

    logger.info(
        "[formal] green clips to process: %d (workers=%d; reused_from_cache=%d)",
        len(jobs),
        n_workers,
        n_reused,
    )
    if not jobs:
        return

    def _one(item: tuple[str, Path]) -> str:
        sid, path = item
        vp.process_and_cache_clip(
            path,
            sample_id=sid,
            valid_mask=mask,
            cfg=ml_cfg,
            vector_cache_dir=vdir,
        )
        # Persist into shared cache for future resumes / devices-partial reruns.
        for src_dir in reuse_dirs[:1]:
            shared_out = src_dir / f"{sid}.npy"
            if shared_out.exists():
                break
            try:
                shared_out.hardlink_to(vdir / f"{sid}.npy")
            except OSError:
                break
        return sid

    with ThreadPoolExecutor(max_workers=max(1, n_workers)) as ex:
        futs = [ex.submit(_one, j) for j in jobs]
        done = 0
        for fut in as_completed(futs):
            fut.result()
            done += 1
            if done % 64 == 0 or done == len(jobs):
                logger.info("[formal] green cache progress %d/%d", done, len(jobs))


def _read_mask_file(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        with np.load(path) as z:
            return np.asarray(z[z.files[0]]).astype(bool)
    if path.suffix == ".npy":
        return np.load(path).astype(bool)
    from puf_common.masks import load_mask_png

    return load_mask_png(path)


def _load_or_build_mask(run_dir: Path, project_root: Path, explicit: Path | None = None) -> np.ndarray:
    """Load the frozen valid mask (explicit path, run cache, or shared cache)."""
    candidates = [
        explicit,
        run_dir / "_cache" / "valid_mask.npy",
        project_root / "outputs/experiment4/security/_shared_cache" / "valid_mask.npy",
    ]
    for c in candidates:
        if c is not None and c.exists():
            logger.info("[formal] using valid_mask %s", c)
            return _read_mask_file(c)
    raise GateError(
        "Frozen valid mask not found. Set `valid_mask:` in the formal config "
        "(e.g. the public masks/exp04_formal_valid_mask.npz) or place valid_mask.npy under "
        f"{run_dir / '_cache'}. Checked: {[str(c) for c in candidates if c is not None]}"
    )


def _extract_all_red(
    found: pd.DataFrame,
    *,
    run_dir: Path,
    mask: np.ndarray,
    n_workers: int,
) -> dict[tuple[str, str, str], np.ndarray]:
    red = found[(found["channel"] == "red") & (found["parse_ok"])].copy()
    out: dict[tuple[str, str, str], np.ndarray] = {}

    def _one(row: pd.Series) -> tuple[tuple[str, str, str], np.ndarray]:
        key = (row["device_id"], row["state_id"], row["phase"])
        rec = process_red_video(
            row["abspath"],
            device_id=row["device_id"],
            state_id=row["state_id"],
            phase=row["phase"],
        )
        vec = extract_red_vector(rec.representative, mask)
        save_red_vector(run_dir, red_sample_id(*key), vec)
        return key, vec

    rows = [r for _, r in red.iterrows()]
    logger.info("[formal] red videos to process: %d", len(rows))
    with ThreadPoolExecutor(max_workers=max(1, min(n_workers, 8))) as ex:
        futs = [ex.submit(_one, r) for r in rows]
        for fut in as_completed(futs):
            k, v = fut.result()
            out[k] = v
    return out


def run_formal_pipeline(
    cfg: IdentityCredentialConfig,
    *,
    validated_by_run_id: str | None = None,
    parent_validate_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from experiment4_security.identity_credential.cache_store import ensure_shared_cache_layout
    from experiment4_security.identity_credential.run_registry import (
        create_run,
        reopen_run,
        sha256_text,
    )

    ensure_shared_cache_layout(cfg.output_root)

    # Re-validate strictly inside formal (do not trust a stale pointer alone)
    validation = validate_dataset(cfg.videos_root)
    audit = validation["audit"]
    require_data_ready_for_formal(audit["data_status"], MODE_FORMAL)
    from experiment4_security.identity_credential.schemas import (
        N_AB_PAIRS_EXPECTED,
        N_GREEN_VIDEOS_EXPECTED,
        N_RED_VIDEOS_EXPECTED,
    )

    if (
        audit.get("n_green_found") != N_GREEN_VIDEOS_EXPECTED
        or audit.get("n_red_found") != N_RED_VIDEOS_EXPECTED
        or audit.get("n_ab_pairs") != N_AB_PAIRS_EXPECTED
        or audit.get("n_missing", 1) != 0
        or audit.get("n_state_conflicts", 1) != 0
    ):
        raise GateError(f"formal re-validation inventory failed: {audit.get('data_status')}")

    ab_checkpoint = _find_ab_checkpoint(cfg.output_root)
    green_resume = None if ab_checkpoint is not None else _find_green_vector_resume_checkpoint(cfg.output_root)
    if ab_checkpoint is not None:
        logger.info("[formal] resuming in-place from A/B checkpoint %s", ab_checkpoint.name)
        ctx = reopen_run(
            cfg.output_root,
            ab_checkpoint.name,
            project_root=cfg.project_root,
            parent_run_id=validated_by_run_id,
            formal_scientific_run=True,
        )
    elif green_resume is not None:
        logger.info("[formal] resuming green-vector phase in-place from %s", green_resume.name)
        ctx = reopen_run(
            cfg.output_root,
            green_resume.name,
            project_root=cfg.project_root,
            parent_run_id=validated_by_run_id,
            formal_scientific_run=True,
        )
    else:
        ctx = create_run(
            cfg.output_root,
            "formal",
            project_root=cfg.project_root,
            parent_run_id=validated_by_run_id,
            formal_scientific_run=True,
        )
    run_dir = ctx.run_dir
    run_id = ctx.run_id
    assert_not_writing_protected(run_dir / "run_manifest.json", cfg.project_root)
    manifest = ctx.base_manifest(validated_by_run_id=validated_by_run_id)
    try:
        cfg_text = json.dumps(cfg.to_resolved_dict(), indent=2, sort_keys=True) + "\n"
        # Prefer yaml extension content (json is fine for resolved dump)
        (run_dir / "config_resolved.yaml").write_text(cfg_text)
        manifest["config_hash"] = sha256_text(cfg_text)
        write_validation_artifacts(validation, run_dir)
        if (run_dir / "found_videos_manifest.csv").exists():
            (run_dir / "found_videos_manifest.csv").replace(run_dir / "data_manifest.csv")
        ctx.write_json("data_audit.json", audit)
        manifest["dataset_manifest_hash"] = None

        found = validation["found"]
        explicit_mask = cfg.raw.get("valid_mask")
        mask = _load_or_build_mask(
            run_dir,
            cfg.project_root,
            cfg.companion_path("valid_mask", "") if explicit_mask else None,
        )
        perf = cfg.performance

        from experiment4_security.ml_attack.config import MLAttackConfig

        ml_cfg_path = cfg.companion_path("green_preprocessing_config", "configs/formal_green_preprocessing.yaml")
        if not ml_cfg_path.exists():
            raise GateError(f"Need {ml_cfg_path} for green preprocessing hyperparameters")
        ml_cfg = MLAttackConfig.from_yaml(ml_cfg_path)

        _ensure_green_vectors(
            found,
            run_dir=run_dir,
            mask=mask,
            n_workers=perf.n_video_workers,
            ml_cfg=ml_cfg,
            shared_vector_dirs=[
                # Permanent resume pool (hardlinked); do not depend on failed run dirs.
                cfg.output_root / "_shared_cache" / "vectors",
            ],
        )

        devices = list(cfg.devices)
        states = list(cfg.states)
        cids = challenge_ids()
        detail_lookups = {d: make_detail_lookup(vector_dir(run_dir), d) for d in devices}

        features: dict[str, np.ndarray] = {}
        try:
            if ml_cfg_path.exists():
                lib_manifest = cf.load_challenge_manifest(ml_cfg.challenge_root)
                if "challenge_number" not in lib_manifest.columns:
                    lib_manifest = lib_manifest.copy()
                    lib_manifest["challenge_number"] = (
                        lib_manifest["challenge_id"].str.replace("C", "", regex=False).astype(int)
                    )
                resolved = cf.ensure_patterns_available(
                    lib_manifest,
                    challenge_root=ml_cfg.challenge_root,
                    pattern_cache_dir=ml_cfg.pattern_cache_dir(),
                    challenge_library_yaml=cfg.companion_path("challenge_library_yaml", "configs/challenge_library.yaml"),
                    repo_root=cfg.project_root,
                )
                feature_sets = cf.build_challenge_features(
                    cids, manifest=lib_manifest, resolved_paths=resolved["paths"]
                )
                # Track C/D expect 1024-d bitmap vectors (same as ml_attack/run.py).
                features = {cid: fs.bitmap_32 for cid, fs in feature_sets.items()}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Challenge features unavailable (%s); Track C/D will fail if invoked", exc)
            features = {}

        if ab_checkpoint is not None and ab_checkpoint.resolve() == run_dir.resolve():
            logger.info(
                "[formal] skipping Track B/S_D recompute; refreshing Track A genuine scores only"
            )
            genuine_by_device_state = _recompute_genuine_by_device_state(
                devices, states, cids, detail_lookups
            )
            manifest["resumed_from_ab_checkpoint"] = True
        else:
            logger.info(
                "[formal] Track A/B multi-device (parallel devices=%d, B targets=%d)",
                perf.green_device_parallel,
                perf.track_b_parallel_targets,
            )
            ab = run_green_tracks_ab_multidevice(
                devices,
                states,
                cids,
                detail_lookups=detail_lookups,
                n_parallel_targets=perf.track_b_parallel_targets,
                green_device_parallel=perf.green_device_parallel,
                run_s_d=True,
            )
            ab["track_a_summary"].to_csv(
                run_dir / "track_a_database_authentication_summary.csv", index=False
            )
            ab["track_b_summary"].to_csv(
                run_dir / "track_b_template_transfer_matrix_summary.csv", index=False
            )
            if ab.get("track_s_d_summary") is not None:
                ab["track_s_d_summary"].to_csv(
                    run_dir / "track_s_d_device_mismatch_summary.csv", index=False
                )
            genuine_by_device_state = {
                d: ab["per_device"][d]["ta_meta"]["genuine_by_state"] for d in devices
            }

        if features:
            logger.info(
                "[formal] Track C/D multi-device (D parallel targets=%d)", perf.track_d_parallel_targets
            )
            cd_ckpt = run_dir / "_checkpoints" / "cd"
            logger.info("[formal] Track C/D checkpoint dir %s", cd_ckpt)
            cd = run_green_tracks_cd_multidevice(
                devices,
                states,
                cids,
                features,
                detail_lookups=detail_lookups,
                genuine_by_device_state=genuine_by_device_state,
                models_cfg=ModelsConfig(),
                pca_dimension=64,
                n_parallel_targets=perf.track_d_parallel_targets,
                checkpoint_dir=cd_ckpt,
            )
            cd["track_c_summary"].to_csv(run_dir / "track_c_same_state_clone_summary.csv", index=False)
            cd["track_d_summary"].to_csv(run_dir / "track_d_clone_transfer_matrix_summary.csv", index=False)
        else:
            cd = {"track_d_conclusions": [], "note": "skipped_no_challenge_features"}

        red_raw = _extract_all_red(
            found, run_dir=run_dir, mask=mask, n_workers=perf.n_video_workers
        )
        std = fit_red_standardizer_on_development(red_raw, cfg.development_devices)
        z = apply_all(red_raw, std["mu"], std["sd"])
        r01 = run_track_r0_r1(z, mode=MODE_FORMAL, data_status=audit["data_status"])
        if r01.get("pairs") is not None and not r01["pairs"].empty:
            cols = [c for c in r01["pairs"].columns if "auc" not in c.lower() and "eer" not in c.lower()]
            r01["pairs"][cols].to_csv(run_dir / "red_pair_scores.csv", index=False)

        j = run_track_j_structure(mode=MODE_FORMAL, data_status=audit["data_status"])
        episodes = build_episodes_from_manifests(
            red_rows=found[found["channel"] == "red"],
            green_ab_pairs=validation["ab_pairs"],
            mode=MODE_FORMAL,
            data_status=audit["data_status"],
        )
        e = run_track_e_structure(mode=MODE_FORMAL, data_status=audit["data_status"])
        leak = leakage_audit_structure(mode=MODE_FORMAL, data_status=audit["data_status"])

        metrics = {
            "formal_scientific_run": True,
            "metrics_emitted": True,
            "validated_by_run_id": validated_by_run_id,
            "absolute_asr": None,
            "asr_status": "SCORE_SPACE_OR_PROTOCOL_MISMATCH",
            "green_track_d_conclusions": cd.get("track_d_conclusions"),
            "red_formal_metrics_emitted": r01.get("formal_metrics_emitted", False),
            "joint_formal_conclusions_emitted": False,
            "track_e_status": e.get("status"),
            "track_j_status": j.get("status"),
            "episodes_n": episodes.get("n_episodes"),
            "leakage_status": leak.get("status"),
            "attack_models": list_attack_models(),
            "performance": cfg.performance.__dict__,
            "parent_validate_data_status": (parent_validate_audit or {}).get("data_status"),
        }
        ctx.write_json("metrics_summary.json", metrics)
        manifest.update(
            {
                "validated_by_run_id": validated_by_run_id,
                "data_status": audit["data_status"],
                "formal_scientific_run": True,
            }
        )
        ctx.finalize_success(manifest)
        summary = {"run_id": run_id, "run_dir": str(run_dir), "status": "SUCCESS", **metrics}
        return summary
    except Exception as exc:
        ctx.log(f"formal FAILED: {exc}")
        ctx.finalize_failed(manifest, str(exc))
        raise
