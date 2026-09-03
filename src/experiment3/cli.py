"""Command-line interface for Experiment 3."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from experiment3.analysis import run_analysis, run_validation
from experiment3.config import load_config
from experiment3.reporting import print_terminal_summary

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
# Keep analysis progress readable: tqdm on stderr, quiet routine INFO spam.
logging.getLogger("threshold_development").setLevel(logging.WARNING)
logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger("threshold_development.cli")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/threshold_development.yaml"),
        help="Pipeline configuration YAML",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path to metadata CSV (overrides config)",
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
    parser = argparse.ArgumentParser(description="Validate Experiment 3 inputs")
    _add_common_args(parser)
    parser.add_argument(
        "--no-check-files",
        action="store_true",
        help="Skip video file existence / probe checks",
    )
    args = parser.parse_args(argv)
    cfg, metadata_path = _load_cfg_from_args(args)
    result = run_validation(
        cfg,
        metadata_path=metadata_path,
        check_files=not args.no_check_files,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("valid") else 1


def cmd_run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Experiment 3 analysis")
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
        output_dir=args.output,
    )
    print_terminal_summary(result)
    print(json.dumps({k: result[k] for k in result if k not in {"red_metrics", "green_metrics"}}, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print("Usage: threshold_development [validate|run] ...")
        return 0
    cmd = argv[0]
    rest = argv[1:]
    if cmd == "validate":
        return cmd_validate(rest)
    if cmd == "run":
        return cmd_run(rest)
    print(f"Unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
