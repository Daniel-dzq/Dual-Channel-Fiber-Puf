"""Fixed-state dual-channel pipeline (Fig. 5a-b, Fig. 6), raw-video mode.

    python -m experiment2.cli validate --config configs/fixed_state_dual_channel.yaml
    python -m experiment2.cli run      --config configs/fixed_state_dual_channel.yaml   # decode videos, score tables
    python -m experiment2.cli analyze  --config configs/fixed_state_dual_channel.yaml   # Fig. 6c-f: 9-D red identity,
                                                                                        # retrieval, binary PUF metrics
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from experiment2.analysis import run_analysis, run_validation
from experiment2.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("fixed_state.cli")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/fixed_state_dual_channel.yaml"),
        help="Pipeline configuration YAML",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path to metadata CSV (overrides config paths.metadata_csv)",
    )
    parser.add_argument(
        "--paths-override",
        type=Path,
        default=None,
        help="Optional YAML with path overrides",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (overrides config paths.output_dir)",
    )


def _load_cfg_from_args(args: argparse.Namespace):
    cfg = load_config(args.config, paths_override=args.paths_override)
    if args.output is not None:
        cfg.paths.output_dir = str(args.output)
    metadata_path = args.metadata or cfg.metadata_path
    return cfg, metadata_path


def cmd_validate(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Experiment 2 inputs")
    _add_common_args(parser)
    parser.add_argument(
        "--no-check-files",
        action="store_true",
        help="Skip video file existence checks",
    )
    args = parser.parse_args(argv)
    cfg, metadata_path = _load_cfg_from_args(args)
    result = run_validation(
        cfg,
        metadata_path=metadata_path,
        check_files=not args.no_check_files,
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("valid") else 1


def cmd_run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Experiment 2 analysis")
    _add_common_args(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs without processing video frames",
    )
    args = parser.parse_args(argv)
    cfg, metadata_path = _load_cfg_from_args(args)
    result = run_analysis(
        cfg,
        metadata_path=metadata_path,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_analyze(argv: list[str] | None = None) -> int:
    from experiment2.binary_puf import run_binary_puf_analysis, write_binary_outputs
    from experiment2.fixed_state_analysis import run_fixed_state_analysis

    parser = argparse.ArgumentParser(description="Fixed-state authentication analysis on the outputs of `run`")
    _add_common_args(parser)
    parser.add_argument("--bootstrap", type=int, default=5000, help="Bootstrap replicates (authority: 5000)")
    args = parser.parse_args(argv)
    cfg, _ = _load_cfg_from_args(args)
    result = run_fixed_state_analysis(cfg, bootstrap_iterations=args.bootstrap)
    if result.get("status") != "ANALYSIS_COMPLETE":
        print(json.dumps({k: result[k] for k in ("status", "run_dir", "audit")}, indent=2, default=str))
        return 1
    run_dir = Path(result["run_dir"])
    binary = run_binary_puf_analysis(cfg.output_dir, n_boot=args.bootstrap)
    write_binary_outputs(binary, run_dir)
    print(json.dumps({"run_dir": str(run_dir), "metric_summary": result["metric_summary"], "binary_metrics": binary["summary"]["metrics"]}, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 2 dual-channel characterization pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="Validate config, challenges, metadata, dark reuse")
    sub.add_parser("run", help="Decode the recordings and write templates and score tables")
    sub.add_parser("analyze", help="Fixed-state authentication analysis (Fig. 6c-f) from the outputs of `run`")

    args, remainder = parser.parse_known_args(argv)
    if args.command == "validate":
        return cmd_validate(remainder)
    if args.command == "run":
        return cmd_run(remainder)
    if args.command == "analyze":
        return cmd_analyze(remainder)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
