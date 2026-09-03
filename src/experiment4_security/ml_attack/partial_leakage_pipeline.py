"""Orchestration for the partial-leakage registered-bank supplementary
experiment (Tracks C-PL / D-PL): config loading, split/manifest/feature
construction, per-device checkpointed execution, resume, aggregation,
integrated-summary + run-record writing. Never touches the base formal
run's Track A-D / red files.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from experiment4_security.ml_attack.challenge_features import (
    build_challenge_features,
    ensure_patterns_available,
    load_challenge_manifest,
)
from experiment4_security.ml_attack.config import ModelsConfig
from experiment4_security.ml_attack.partial_leakage_models import load_hamming_distance_matrix
from experiment4_security.ml_attack.partial_leakage_reporting import (
    collapse_repetitions,
    model_level_summary,
    panel_heatmap,
    panel_learning_curve,
    panel_same_cross_dumbbell,
    render_analysis_markdown,
    select_narrative,
)
from experiment4_security.ml_attack.partial_leakage_splits import (
    all_splits_by_repetition,
    build_split_manifest,
    load_bank_map,
)
from experiment4_security.ml_attack.track_d_pl_transfer import run_device_partial_leakage

logger = logging.getLogger(__name__)

TZ_SGT = timezone(timedelta(hours=8))


def _now_iso() -> str:
    return datetime.now(TZ_SGT).isoformat()


def _resolve(base: Path, p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (base / p).resolve()


@dataclass
class PartialLeakageConfig:
    project_root: Path
    base_run: Path
    shared_cache_root: Path
    challenge_root: Path
    challenge_manifest_csv: Path
    hamming_distance_csv: Path
    challenge_library_yaml: Path
    output_root: Path
    figure_data_root: Path
    devices: tuple[str, ...]
    states: tuple[str, ...]
    challenge_id_start: int
    challenge_id_end: int
    n_banks: int
    bank_size: int
    leak_sizes: tuple[int, ...]
    per_bank_leak_counts: tuple[int, ...]
    repetitions: int
    seeds: tuple[int, ...]
    pca_dimension_max: int
    bootstrap_iterations: int
    bootstrap_seed: int
    representative_condition: dict[str, Any]
    models_cfg: ModelsConfig
    raw: dict[str, Any] = field(default_factory=dict)
    config_path: Path | None = None

    @classmethod
    def from_yaml(cls, path: Path | str) -> "PartialLeakageConfig":
        path = Path(path)
        raw = yaml.safe_load(path.read_text()) or {}
        default_root = path.resolve().parent.parent
        pr = raw.get("project_root")
        project_root = default_root if pr in (None, "", ".") else _resolve(default_root, pr)

        mdl = raw.get("models", {}) or {}
        models_cfg = ModelsConfig(
            ridge_alpha=float(mdl.get("ridge_alpha", 10.0)),
            kernel_ridge_alpha=float(mdl.get("kernel_ridge_alpha", 1.0)),
            kernel_ridge_gamma=float(mdl.get("kernel_ridge_gamma", 0.01)),
            rff_n_components=int(mdl.get("rff_n_components", 256)),
            rff_gamma=float(mdl.get("rff_gamma", 0.01)),
            rff_alpha=float(mdl.get("rff_alpha", 1.0)),
            mlp_hidden_sizes=tuple(mdl.get("mlp_hidden_sizes", [64, 32])),
            mlp_max_iter=int(mdl.get("mlp_max_iter", 150)),
            mlp_learning_rate=float(mdl.get("mlp_learning_rate", 1e-3)),
            mlp_random_seed=int(mdl.get("mlp_random_seed", 20260721)),
        )

        return cls(
            project_root=project_root,
            base_run=_resolve(project_root, raw["base_run"]),
            shared_cache_root=_resolve(project_root, raw["shared_cache_root"]),
            challenge_root=_resolve(project_root, raw["challenge_root"]),
            challenge_manifest_csv=_resolve(project_root, raw["challenge_manifest_csv"]),
            hamming_distance_csv=_resolve(project_root, raw["hamming_distance_csv"]),
            challenge_library_yaml=_resolve(project_root, raw["challenge_library_yaml"]),
            output_root=_resolve(project_root, raw["output_root"]),
            figure_data_root=_resolve(project_root, raw["figure_data_root"]),
            devices=tuple(
                raw.get(
                    "devices",
                    [
                        "F01",
                        "F02",
                        "F03",
                        "F04",
                        "F05",
                        "F06",
                        "F07",
                        "F08",
                        "F09",
                        "F10",
                    ],
                )
            ),
            states=tuple(raw.get("states", [f"M{i}" for i in range(8)])),
            challenge_id_start=int(raw.get("challenge_id_start", 1)),
            challenge_id_end=int(raw.get("challenge_id_end", 128)),
            n_banks=int(raw.get("n_banks", 16)),
            bank_size=int(raw.get("bank_size", 8)),
            leak_sizes=tuple(raw.get("leak_sizes", [16, 32, 64, 96])),
            per_bank_leak_counts=tuple(raw.get("per_bank_leak_counts", [1, 2, 4, 6])),
            repetitions=int(raw.get("repetitions", 5)),
            seeds=tuple(raw.get("seeds", [20260801, 20260802, 20260803, 20260804, 20260805])),
            pca_dimension_max=int(raw.get("pca_dimension_max", 64)),
            bootstrap_iterations=int(raw.get("bootstrap_iterations", 10000)),
            bootstrap_seed=int(raw.get("bootstrap_seed", 42)),
            representative_condition=raw.get(
                "representative_condition", {"leak_size": 96, "attack_method": "ridge_clone"}
            ),
            models_cfg=models_cfg,
            raw=raw,
            config_path=path,
        )

    def config_hash(self) -> str:
        blob = json.dumps(self.raw, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def challenge_ids(self) -> list[str]:
        return [f"C{i:03d}" for i in range(self.challenge_id_start, self.challenge_id_end + 1)]


def build_challenge_feature_bank(cfg: PartialLeakageConfig) -> dict[str, np.ndarray]:
    manifest = load_challenge_manifest(cfg.challenge_root / "patterns")
    resolved = ensure_patterns_available(
        manifest,
        challenge_root=cfg.challenge_root / "patterns",
        pattern_cache_dir=cfg.output_root / "_challenge_pattern_cache",
        challenge_library_yaml=cfg.challenge_library_yaml,
        repo_root=cfg.project_root,
    )
    features = build_challenge_features(cfg.challenge_ids(), manifest=manifest, resolved_paths=resolved["paths"])
    return {cid: fs.bitmap_32 for cid, fs in features.items()}


def build_splits(cfg: PartialLeakageConfig) -> dict[str, Any]:
    bank_map = load_bank_map(cfg.challenge_manifest_csv)
    challenge_to_bank = {cid: bank_id for bank_id, cids in bank_map.items() for cid in cids}
    manifest = build_split_manifest(
        cfg.seeds, bank_map, leak_sizes=cfg.leak_sizes, per_bank_counts=cfg.per_bank_leak_counts
    )
    splits_by_rep = all_splits_by_repetition(manifest, leak_sizes=cfg.leak_sizes)
    return {
        "bank_map": bank_map,
        "challenge_to_bank": challenge_to_bank,
        "split_manifest": manifest,
        "splits_by_rep": splits_by_rep,
    }


def _checkpoint_paths(cfg: PartialLeakageConfig, device_id: str) -> dict[str, Path]:
    d = cfg.output_root / "checkpoints" / device_id
    d.mkdir(parents=True, exist_ok=True)
    return {
        "dir": d,
        "c_pl_summary": d / "c_pl_summary.parquet",
        "c_pl_hidden": d / "c_pl_hidden_scores.parquet",
        "d_pl_summary": d / "d_pl_transfer_summary.parquet",
        "d_pl_hidden": d / "d_pl_transfer_hidden_scores.parquet",
        "done": d / "device_done.json",
    }


def run_partial_leakage(
    cfg: PartialLeakageConfig,
    *,
    devices: list[str] | None = None,
    states: list[str] | None = None,
    leak_sizes: list[int] | None = None,
    model_keys: tuple[str, ...] | None = None,
    dry_run: bool = False,
    resume: bool = True,
    strict: bool = True,
) -> dict[str, Any]:
    t0 = time.time()
    devices = devices or list(cfg.devices)
    states = states or list(cfg.states)
    leak_sizes = leak_sizes or list(cfg.leak_sizes)

    logger.info("[partial-leakage] devices=%s states=%s leak_sizes=%s dry_run=%s", devices, states, leak_sizes, dry_run)

    all_ids = cfg.challenge_ids()
    split_data = build_splits(cfg)
    if strict:
        _strict_validate_splits(split_data["split_manifest"], all_ids, cfg)

    splits_by_rep_full = split_data["splits_by_rep"]
    splits_by_rep = {
        rep: {ls: split for ls, split in leak_map.items() if ls in leak_sizes} for rep, leak_map in splits_by_rep_full.items()
    }

    hamming = load_hamming_distance_matrix(str(cfg.hamming_distance_csv), all_ids)

    for state in states:
        for device in devices:
            for round_id in ("A", "B"):
                for cid in all_ids[:1]:
                    p = cfg.shared_cache_root / "vectors" / f"{device}_{state}_{round_id}_{cid}.npy"
                    if not p.exists():
                        raise FileNotFoundError(f"Missing shared-cache vector required by partial-leakage: {p}")

    if dry_run:
        n_cells = len(devices) * len(states) * cfg.repetitions * len(leak_sizes)
        methods = model_keys or (
            "PL0_mean_leaked_response",
            "PL1_nearest_leaked_challenge",
            "PL2_ridge_clone",
            "PL3_kernel_ridge_clone",
            "PL4_random_fourier_ridge_clone",
            "PL5_small_mlp_clone",
        )
        return {
            "dry_run": True,
            "devices": devices,
            "states": states,
            "leak_sizes": leak_sizes,
            "repetitions": cfg.repetitions,
            "methods": list(methods),
            "n_source_state_cells": n_cells,
            "n_fit_events_estimate": n_cells * len(methods),
            "config_hash": cfg.config_hash(),
        }

    challenge_features = build_challenge_feature_bank(cfg)

    device_outputs: dict[str, dict[str, pd.DataFrame]] = {}
    device_timings: dict[str, float] = {}
    pending: list[str] = []
    for device in devices:
        cps = _checkpoint_paths(cfg, device)
        if resume and cps["done"].exists():
            logger.info("[partial-leakage] device=%s: resume from checkpoint", device)
            device_outputs[device] = {
                "c_pl_summary": pd.read_parquet(cps["c_pl_summary"]),
                "c_pl_hidden_scores": pd.read_parquet(cps["c_pl_hidden"]),
                "d_pl_transfer_summary": pd.read_parquet(cps["d_pl_summary"]),
                "d_pl_transfer_hidden_scores": pd.read_parquet(cps["d_pl_hidden"]),
            }
            device_timings[device] = json.loads(cps["done"].read_text()).get("elapsed_s", 0.0)
            continue
        pending.append(device)

    device_workers = max(1, int((cfg.raw.get("performance") or {}).get("device_workers", 1)))
    # Cap by free RAM budget: each live device fit is ~15–18 GB RSS.
    device_workers = min(device_workers, max(1, len(pending)))
    logger.info(
        "[partial-leakage] pending_devices=%s device_workers=%d",
        pending, device_workers,
    )

    def _run_one(device: str) -> tuple[str, dict[str, Any], float]:
        cps = _checkpoint_paths(cfg, device)
        cps["c_pl_summary"].parent.mkdir(parents=True, exist_ok=True)
        logger.info("[partial-leakage] device=%s: start", device)
        result = run_device_partial_leakage(
            device,
            shared_cache_root=cfg.shared_cache_root,
            states=states,
            all_challenge_ids=all_ids,
            challenge_to_bank=split_data["challenge_to_bank"],
            challenge_features=challenge_features,
            hamming_matrix=hamming,
            splits_by_rep=splits_by_rep,
            models_cfg=cfg.models_cfg,
            pca_dimension_max=cfg.pca_dimension_max,
            source_states=states,
            model_keys=tuple(model_keys) if model_keys else None,
        )
        result["c_pl_summary"].to_parquet(cps["c_pl_summary"], index=False)
        result["c_pl_hidden_scores"].to_parquet(cps["c_pl_hidden"], index=False)
        result["d_pl_transfer_summary"].to_parquet(cps["d_pl_summary"], index=False)
        result["d_pl_transfer_hidden_scores"].to_parquet(cps["d_pl_hidden"], index=False)
        cps["done"].write_text(
            json.dumps(
                {"device_id": device, "elapsed_s": result["elapsed_s"], "completed_at": _now_iso()},
                indent=2,
            )
        )
        logger.info("[partial-leakage] device=%s done in %.1fs", device, result["elapsed_s"])
        return device, result, float(result["elapsed_s"])

    if device_workers <= 1 or len(pending) <= 1:
        for device in pending:
            device, result, elapsed = _run_one(device)
            device_outputs[device] = result
            device_timings[device] = elapsed
    else:
        with ThreadPoolExecutor(max_workers=device_workers) as pool:
            futs = {pool.submit(_run_one, d): d for d in pending}
            for fut in as_completed(futs):
                device, result, elapsed = fut.result()
                device_outputs[device] = result
                device_timings[device] = elapsed

    c_pl_summary_all = pd.concat([v["c_pl_summary"] for v in device_outputs.values()], ignore_index=True)
    c_pl_hidden_all = pd.concat([v["c_pl_hidden_scores"] for v in device_outputs.values()], ignore_index=True)
    d_pl_summary_all = pd.concat([v["d_pl_transfer_summary"] for v in device_outputs.values()], ignore_index=True)
    d_pl_hidden_all = pd.concat([v["d_pl_transfer_hidden_scores"] for v in device_outputs.values()], ignore_index=True)

    collapsed = collapse_repetitions(d_pl_summary_all)
    model_summary = model_level_summary(
        collapsed, bootstrap_iterations=cfg.bootstrap_iterations, bootstrap_seed=cfg.bootstrap_seed
    )
    narrative = select_narrative(model_summary)

    return {
        "config_hash": cfg.config_hash(),
        "devices": devices,
        "states": states,
        "leak_sizes": leak_sizes,
        "repetitions": cfg.repetitions,
        "split_manifest": split_data["split_manifest"],
        "c_pl_partial_leakage_summary": c_pl_summary_all,
        "track_c_pl_hidden_scores": c_pl_hidden_all,
        "track_d_pl_transfer_summary": d_pl_summary_all,
        "track_d_pl_transfer_hidden_scores": d_pl_hidden_all,
        "collapsed_by_device_state": collapsed,
        "model_level_summary": model_summary,
        "narrative": narrative,
        "device_timings": device_timings,
        "elapsed_s": time.time() - t0,
    }


def write_all_outputs(cfg: PartialLeakageConfig, results: dict[str, Any]) -> dict[str, str]:
    """Write every supplementary output file (Section 10 / 12) and the
    integrated metrics summary + updated FORMAL_FULL_RUN_RECORD.md. Never
    touches the base run's original Track A-D / red files."""
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    cfg.figure_data_root.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    def _w_csv(df: pd.DataFrame, name: str, root: Path = cfg.output_root) -> None:
        p = root / name
        df.to_csv(p, index=False)
        written[name] = str(p)

    def _w_parquet(df: pd.DataFrame, name: str) -> None:
        p = cfg.output_root / name
        df.to_parquet(p, index=False)
        written[name] = str(p)

    _w_csv(results["split_manifest"], "split_manifest.csv")
    _w_parquet(results["track_c_pl_hidden_scores"], "track_c_pl_hidden_scores.parquet")
    _w_csv(results["c_pl_partial_leakage_summary"], "track_c_pl_partial_leakage_summary.csv")
    _w_parquet(results["track_d_pl_transfer_hidden_scores"], "track_d_pl_transfer_scores.parquet")
    _w_csv(results["track_d_pl_transfer_summary"], "track_d_pl_partial_leakage_transfer_summary.csv")
    _w_csv(results["model_level_summary"], "track_d_pl_model_level_summary.csv")

    rep = cfg.representative_condition
    heatmap = panel_heatmap(
        results["collapsed_by_device_state"],
        leak_size=int(rep.get("leak_size", 96)),
        attack_method=str(rep.get("attack_method", "ridge_clone")),
    )
    _w_csv(panel_learning_curve(results["collapsed_by_device_state"]), "panel_partial_leakage_learning_curve.csv", cfg.figure_data_root)
    _w_csv(panel_same_cross_dumbbell(results["model_level_summary"]), "panel_partial_leakage_same_cross.csv", cfg.figure_data_root)
    _w_csv(heatmap, "panel_partial_leakage_heatmap.csv", cfg.figure_data_root)

    metrics_summary = {
        "protocol": "partial_leakage_registered_bank_v1",
        "generated_at": _now_iso(),
        "config_hash": results["config_hash"],
        "devices": results["devices"],
        "states": results["states"],
        "leak_sizes": results["leak_sizes"],
        "repetitions": results["repetitions"],
        "representative_condition": cfg.representative_condition,
        "device_timings_s": results["device_timings"],
        "elapsed_s": results["elapsed_s"],
        "narrative": results["narrative"],
        "model_level_summary": results["model_level_summary"].to_dict(orient="records"),
        "notes": {
            "threat_model": "conservative_partial_database_compromise; leaked records are server-stored enrollment-processed templates",
            "exact_template_replay": "not_applicable_to_attacker_held_out_registered_challenges",
            "absolute_asr": None,
            "no_ml_proof_claim": True,
            "no_unseen_challenge_secure_claim": True,
            "no_red_green_interaction": True,
        },
    }
    (cfg.output_root / "partial_leakage_metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2, default=str))
    written["partial_leakage_metrics_summary.json"] = str(cfg.output_root / "partial_leakage_metrics_summary.json")

    (cfg.output_root / "config_resolved.yaml").write_text(yaml.safe_dump(cfg.raw, sort_keys=False))
    written["config_resolved.yaml"] = str(cfg.output_root / "config_resolved.yaml")

    analysis_md = render_analysis_markdown(
        generated_at=metrics_summary["generated_at"],
        devices=results["devices"],
        states=results["states"],
        leak_sizes=results["leak_sizes"],
        repetitions=results["repetitions"],
        seeds=cfg.seeds,
        n_banks=cfg.n_banks,
        model_summary=results["model_level_summary"],
        narrative=results["narrative"],
        representative_condition=cfg.representative_condition,
    )
    (cfg.output_root / "PARTIAL_LEAKAGE_ANALYSIS.md").write_text(analysis_md)
    written["PARTIAL_LEAKAGE_ANALYSIS.md"] = str(cfg.output_root / "PARTIAL_LEAKAGE_ANALYSIS.md")

    write_integrated_summary(cfg, metrics_summary)
    write_run_record(cfg, results, metrics_summary)
    appended = append_formal_full_run_record(cfg, results, metrics_summary)
    if appended is not None:
        written["FORMAL_FULL_RUN_RECORD.md"] = str(appended)
    return written


def write_integrated_summary(cfg: PartialLeakageConfig, pl_metrics_summary: dict[str, Any]) -> Path:
    """metrics_summary_integrated.json = exact copy of the original
    metrics_summary.json + a new 'green_partial_leakage' key. Original file
    is never modified."""
    original_path = cfg.base_run / "metrics_summary.json"
    original = json.loads(original_path.read_text())
    integrated = dict(original)
    try:
        supplementary_path = str(cfg.output_root.relative_to(cfg.project_root))
    except ValueError:
        supplementary_path = str(cfg.output_root)
    integrated["green_partial_leakage"] = {
        "supplementary_path": supplementary_path,
        "config_hash": pl_metrics_summary["config_hash"],
        "generated_at": pl_metrics_summary["generated_at"],
        "run_completion_status": "SUCCESS",
        "narrative_key": pl_metrics_summary["narrative"].get("narrative_key"),
        "narrative": pl_metrics_summary["narrative"].get("narrative"),
        "devices": pl_metrics_summary["devices"],
        "leak_sizes": pl_metrics_summary["leak_sizes"],
        "repetitions": pl_metrics_summary["repetitions"],
        "output_files": [
            "supplementary/partial_leakage/split_manifest.csv",
            "supplementary/partial_leakage/track_c_pl_hidden_scores.parquet",
            "supplementary/partial_leakage/track_c_pl_partial_leakage_summary.csv",
            "supplementary/partial_leakage/track_d_pl_transfer_scores.parquet",
            "supplementary/partial_leakage/track_d_pl_partial_leakage_transfer_summary.csv",
            "supplementary/partial_leakage/track_d_pl_model_level_summary.csv",
            "supplementary/partial_leakage/partial_leakage_metrics_summary.json",
            "supplementary/partial_leakage/PARTIAL_LEAKAGE_ANALYSIS.md",
        ],
        "note": (
            "This key is additive only. All Track A/B/S_D/C/D and red R0/R1 "
            "values above are byte-identical to the original metrics_summary.json."
        ),
    }
    out_path = cfg.base_run / "metrics_summary_integrated.json"
    out_path.write_text(json.dumps(integrated, indent=2, default=str))
    # Defense-in-depth: verify the non-additive keys are untouched.
    for k in original:
        if k != "green_partial_leakage" and integrated[k] != original[k]:
            raise AssertionError(f"metrics_summary_integrated.json mutated original key {k}")
    return out_path


def write_run_record(cfg: PartialLeakageConfig, results: dict[str, Any], pl_metrics_summary: dict[str, Any]) -> Path:
    narrative = results["narrative"]
    ms = results["model_level_summary"]
    lines = [
        "# Partial-Leakage Registered-Bank Supplementary Run Record",
        "",
        f"Generated: {pl_metrics_summary['generated_at']}",
        "",
        "This is a SUPPLEMENTARY experiment. It does not modify, overwrite, or "
        "recompute the base formal run's Track A/B/S_D/C/D or red R0/R1 results "
        f"in `{cfg.base_run.name}/`.",
        "",
        "## Threat model",
        "",
        "Conservative partial database-compromise model: leaked records are the "
        "server-stored, enrollment-processed templates T_full[d,s,c] for a "
        "subset of the registered C001-C128 bank. Hidden-challenge templates, "
        "Round B, and target-state data never enter any fit/PCA/normalization/"
        "model-selection step.",
        "",
        "## Scope",
        "",
        f"- devices: {results['devices']}",
        f"- states: {results['states']}",
        f"- leak sizes: {results['leak_sizes']}",
        f"- repetitions: {results['repetitions']} (seeds 20260801-20260805)",
        f"- config_hash: {results['config_hash']}",
        f"- elapsed_s: {results['elapsed_s']:.1f}",
        "",
        "## Model-level summary (device-cluster bootstrap, 10000 iters, seed 42)",
        "",
        ms.to_string(index=False) if not ms.empty else "(no rows)",
        "",
        "## Narrative (rule-based, Section 14)",
        "",
        f"Selected: **{narrative.get('narrative_key')}**",
        "",
        narrative.get("narrative", ""),
        "",
        "## Output files",
        "",
        "All under `supplementary/partial_leakage/`:",
        "",
        "- split_manifest.csv",
        "- track_c_pl_hidden_scores.parquet",
        "- track_c_pl_partial_leakage_summary.csv",
        "- track_d_pl_transfer_scores.parquet",
        "- track_d_pl_partial_leakage_transfer_summary.csv",
        "- track_d_pl_model_level_summary.csv",
        "- partial_leakage_metrics_summary.json",
        "- PARTIAL_LEAKAGE_ANALYSIS.md",
        "- config_resolved.yaml",
        "",
        "Figure data under `figure/security/data/`:",
        "",
        "- panel_partial_leakage_learning_curve.csv",
        "- panel_partial_leakage_same_cross.csv",
        "- panel_partial_leakage_heatmap.csv",
        "",
        "Integrated summary (base run root, original `metrics_summary.json` unchanged): "
        "`metrics_summary_integrated.json`.",
        "",
    ]
    out = cfg.output_root / "PARTIAL_LEAKAGE_RUN_RECORD.md"
    out.write_text("\n".join(lines))
    return out


_FORMAL_RECORD_MARKER = "## 8. Partial-disclosure supplementary experiment (Track C-PL / D-PL)"


def append_formal_full_run_record(cfg: PartialLeakageConfig, results: dict[str, Any], pl_metrics_summary: dict[str, Any]) -> Path | None:
    """Append a supplementary section to the security-suite-level
    ``FORMAL_FULL_RUN_RECORD.md`` (one level above ``runs/``). Idempotent:
    if the marker section already exists, the trailing content after it is
    replaced in-place rather than duplicated; the pre-existing timeline
    (sections 0-7) above the marker is never touched.
    """
    record_path = cfg.base_run.parent.parent / "FORMAL_FULL_RUN_RECORD.md"
    if not record_path.exists():
        logger.warning("[partial-leakage] FORMAL_FULL_RUN_RECORD.md not found at %s; skipping append", record_path)
        return None

    original = record_path.read_text()
    head = original.split(_FORMAL_RECORD_MARKER)[0].rstrip() + "\n"

    ms = results["model_level_summary"]
    narrative = results["narrative"]
    section_lines = [
        "",
        "---",
        "",
        _FORMAL_RECORD_MARKER,
        "",
        "> This section is appended automatically by `partial_leakage_pipeline.append_formal_full_run_record`; "
        "**sections 0-7 above are not rewritten**. This is a supplementary security experiment stored inside "
        f"the same formal run directory `{cfg.base_run.name}/`; it evaluates hidden-challenge generalization "
        "under partial enrollment-database disclosure (Track C-PL) and frozen cross-state transfer after mechanical reconfiguration (Track D-PL).",
        "",
        f"- generated_at: {pl_metrics_summary['generated_at']}",
        f"- config_hash: {results['config_hash']}",
        f"- devices: {results['devices']}",
        f"- states: {results['states']}",
        f"- leak_sizes: {results['leak_sizes']}",
        f"- repetitions: {results['repetitions']} (seeds 20260801-20260805)",
        f"- elapsed_s: {results['elapsed_s']:.1f}",
        f"- per-device timings (s): {json.dumps(results['device_timings'], default=str)}",
        "",
        "All result files are located in "
        f"`runs/{cfg.base_run.name}/supplementary/partial_leakage/`，"
        "Existing Track A/B/S_D/C/D and red R0/R1 files are byte-identical before and after this run (SHA-256 verified)."
        " See `PARTIAL_LEAKAGE_RUN_RECORD.md` and `PARTIAL_LEAKAGE_ANALYSIS.md` in that directory, "
        "and `metrics_summary_integrated.json` added at the base run root "
        "(`green_partial_leakage` key; the original `metrics_summary.json` is unchanged).",
        "",
        "### Model-level summary (device-cluster bootstrap, 10000 iterations, seed 42)",
        "",
        "```",
        ms.to_string(index=False) if not ms.empty else "(no rows)",
        "```",
        "",
        "### Automatic narrative selection (Section 14 rule)",
        "",
        f"Selected: **{narrative.get('narrative_key')}**",
        "",
        narrative.get("narrative", ""),
        "",
    ]
    record_path.write_text(head + "\n".join(section_lines))
    logger.info("[partial-leakage] appended supplementary section to %s", record_path)
    return record_path


def _strict_validate_splits(manifest: pd.DataFrame, all_ids: list[str], cfg: PartialLeakageConfig) -> None:
    all_ids_set = set(all_ids)
    for (rep, leak_size), grp in manifest.groupby(["repetition", "leak_size"]):
        leaked = set(grp.loc[grp["role"] == "leaked", "challenge_id"])
        hidden = set(grp.loc[grp["role"] == "hidden", "challenge_id"])
        if leaked & hidden:
            raise AssertionError(f"rep={rep} leak_size={leak_size}: leaked/hidden overlap")
        if leaked | hidden != all_ids_set:
            raise AssertionError(f"rep={rep} leak_size={leak_size}: union != full 128-challenge bank")
        if len(leaked) != leak_size:
            raise AssertionError(f"rep={rep} leak_size={leak_size}: wrong leaked count {len(leaked)}")
        n_banks_covered = grp.loc[grp["role"] == "leaked", "bank_id"].nunique()
        if n_banks_covered != cfg.n_banks:
            raise AssertionError(f"rep={rep} leak_size={leak_size}: only {n_banks_covered}/{cfg.n_banks} banks covered")
