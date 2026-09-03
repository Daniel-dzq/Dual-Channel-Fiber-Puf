"""CLI with unified metric terminal summary."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from experiment4_security.ml_attack.config import MLAttackConfig
from experiment4_security.ml_attack.run import run_pilot


def _print_terminal_summary(summary: dict) -> None:
    print("\n" + "=" * 78)
    print("Experiment 4 registered-database security summary")
    print("=" * 78)
    if summary.get("aborted_at"):
        print(f"ABORTED at stage: {summary['aborted_at']}")
        print(f"data_status: {summary.get('data_status')}")
        return
    print(f"active_run_dir: {summary.get('run_dir')}")
    print(f"elapsed_s: {summary.get('elapsed_s'):.1f}" if summary.get("elapsed_s") is not None else "")
    print(f"data_status: {summary.get('data_status')}")
    da = summary.get("data_audit", {})
    print(f"n_videos_found: {da.get('n_videos_found')}")
    print(f"n_state_id_conflicts_raw/active: {da.get('n_state_id_conflicts_raw')}/{da.get('n_state_id_conflicts_active')}")
    print(f"n_ab_pairs: {da.get('n_ab_pairs_ok')}")
    print(f"n_qc_pass/warn/error: {summary.get('n_qc_pass')}/{summary.get('n_qc_warn')}/{summary.get('n_qc_error')}")
    print(f"absolute_asr: {summary.get('absolute_asr')}")
    print(f"asr_status: {summary.get('asr_status')}")
    print(f"conclusion_levels: {summary.get('conclusion_levels')}")

    ta = summary.get("track_a_summary")
    if ta is not None and hasattr(ta, "empty") and not ta.empty:
        cols = [c for c in ["state_id", "median_S_G", "q05_S_G", "median_S_C", "q95_S_C", "rg_challenge", "auc_challenge", "eer_challenge", "top1"] if c in ta.columns]
        print("\nTrack A (S_G vs S_C):")
        print(ta[cols].to_string(index=False))

    tb_meta = summary.get("track_b_meta") or {}
    if tb_meta:
        print("\nTrack B (S_G vs S_X):")
        print(f"  diagonal_median_S_X: {tb_meta.get('diagonal_median')}")
        print(f"  off_diagonal_median_S_X: {tb_meta.get('off_diagonal_median')}")
        print(f"  minimum_RG_X: {tb_meta.get('minimum_rg_cross_state_credential')}")
        print(f"  worst pair: {tb_meta.get('source_state_worst_target')}")
        print(f"  conclusion: {tb_meta.get('conclusion')}")

    tc = summary.get("track_c_summary")
    if tc is not None and hasattr(tc, "empty") and not tc.empty:
        print("\nTrack C (S_G vs S_A, same-state; by attack_method):")
        gcols = [c for c in ["median_S_A", "rg_software_clone", "auc_software_clone", "eer_software_clone", "top1", "residual_ncc", "source_state_valid"] if c in tc.columns]
        print(tc.groupby("attack_method")[gcols].mean().to_string())

    td = summary.get("track_d_summary")
    td_meta = summary.get("track_d_meta") or {}
    if td is not None and hasattr(td, "empty") and not td.empty:
        off = td[~td["is_diagonal"]] if "is_diagonal" in td.columns else td
        print("\nTrack D (cross-state S_A):")
        if "attack_method" in off.columns and "median_S_A" in off.columns:
            print(off.groupby("attack_method")[["median_S_A", "rg_software_clone", "delta_clone_same_to_cross"]].mean().to_string())
    if td_meta.get("per_model_conclusions"):
        print("Track D conclusions:")
        for c in td_meta["per_model_conclusions"]:
            print(f"  {c}")

    print("\nRed identity: RED_IDENTITY_EVIDENCE_FROM_SEPARATE_LIFECYCLE_DATASET")
    print("Red-conditioned attack: RED_CONDITIONED_ATTACK_NOT_AVAILABLE")
    print("=" * 78 + "\n")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run Experiment 4 registered-database security analysis")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run id under output_root/runs/ (default: UTC timestamp + _f01_green_pilot).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 2),
        help="Workers for missing-clip reprocessing only (default cpu_count-2 for M4 Pro).",
    )
    args = parser.parse_args(argv)
    cfg = MLAttackConfig.from_yaml(args.config)
    summary = run_pilot(cfg, run_id=args.run_id, n_workers=args.workers)
    _print_terminal_summary(summary)
    return 0 if not summary.get("aborted_at") else 2


if __name__ == "__main__":
    sys.exit(main())
