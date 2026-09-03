"""Top-level security CLI: `python -m experiment4_security.cli partial-leakage ...`.

Hosts the partial enrollment-database disclosure experiment (Tracks C-PL / D-PL,
Fig. 8g-h and Supplementary Fig. S4). The formal 80-unit pipeline lives in
``experiment4_security.identity_credential.cli``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from experiment4_security.ml_attack.partial_leakage_pipeline import (
    PartialLeakageConfig,
    run_partial_leakage,
    write_all_outputs,
)


def _print_summary(result: dict) -> None:
    print("\n" + "=" * 78)
    print("Partial-disclosure (registered-bank) supplementary experiment")
    print("=" * 78)
    if result.get("dry_run"):
        print(json.dumps(result, indent=2, default=str))
        return
    print(f"devices: {result['devices']}")
    print(f"states: {result['states']}")
    print(f"leak_sizes: {result['leak_sizes']}")
    print(f"repetitions: {result['repetitions']}")
    print(f"elapsed_s: {result['elapsed_s']:.1f}")
    ms = result["model_level_summary"]
    if not ms.empty:
        print("\nModel-level summary (device-cluster bootstrap CI):")
        print(ms.to_string(index=False))
    nar = result.get("narrative", {})
    print(f"\nNarrative: {nar.get('narrative_key')}: {nar.get('narrative')}")
    print("=" * 78 + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Security CLI (partial enrollment-database disclosure)")
    sub = p.add_subparsers(dest="command", required=True)

    pl = sub.add_parser("partial-leakage", help="Partial-disclosure registered-bank supplementary experiment")
    pl.add_argument("--base-run", type=Path, default=None, help="Override base_run from config")
    pl.add_argument("--config", type=Path, required=True)
    pl.add_argument("--resume", action="store_true", default=True)
    pl.add_argument("--no-resume", dest="resume", action="store_false")
    pl.add_argument("--device", action="append", default=None, help="Restrict to one or more devices (repeatable)")
    pl.add_argument("--leak-size", type=int, action="append", default=None, help="Restrict to one or more leak sizes (repeatable)")
    pl.add_argument("--model", action="append", default=None, help="Restrict to one or more PL model keys (repeatable)")
    pl.add_argument("--dry-run", action="store_true", default=False)
    pl.add_argument("--strict", action="store_true", default=True)
    pl.add_argument("--no-strict", dest="strict", action="store_false")
    pl.add_argument("--skip-write", action="store_true", default=False, help="Debug: compute but do not write output files")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    if args.command == "partial-leakage":
        cfg = PartialLeakageConfig.from_yaml(args.config)
        if args.base_run:
            cfg.base_run = args.base_run if args.base_run.is_absolute() else (cfg.project_root / args.base_run).resolve()
        result = run_partial_leakage(
            cfg,
            devices=args.device,
            leak_sizes=args.leak_size,
            model_keys=tuple(args.model) if args.model else None,
            dry_run=args.dry_run,
            resume=args.resume,
            strict=args.strict,
        )
        _print_summary(result)
        if not result.get("dry_run") and not args.skip_write:
            written = write_all_outputs(cfg, result)
            print(f"Wrote {len(written)} output files under {cfg.output_root}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
