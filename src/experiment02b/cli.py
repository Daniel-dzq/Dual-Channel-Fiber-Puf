"""CLI stages for experiment 02b."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from experiment02b.config import load_config
from experiment02b.pipeline import stage_manifest, stage_metrics, stage_report, stage_statistics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Experiment 02b factorial optical control")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--paths-override", type=Path, default=None)
    p.add_argument(
        "--stage",
        choices=["manifest", "qc", "metrics", "statistics", "report", "all", "plots"],
        default="all",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if args.stage == "plots":
        logging.error(
            "Publication figures intentionally deferred until the numerical analysis, "
            "quality control and factorial interpretation are reviewed."
        )
        return 2
    cfg = load_config(args.config, paths_override=args.paths_override)
    results: dict = {}
    stages = {
        "manifest": ["manifest"],
        "qc": ["manifest", "metrics"],  # QC CSV emitted by metrics stage
        "metrics": ["manifest", "metrics"],
        "statistics": ["statistics"],
        "report": ["report"],
        "all": ["manifest", "metrics", "statistics", "report"],
    }[args.stage]

    if "manifest" in stages:
        results["manifest"] = stage_manifest(cfg)
    if "metrics" in stages:
        results["metrics"] = stage_metrics(cfg)
    if "statistics" in stages:
        results["statistics"] = stage_statistics(cfg)
    if "report" in stages:
        results["report"] = stage_report(cfg)

    print(json.dumps(results, indent=2, default=str))
    if "manifest" in results and not results["manifest"]["audit"].get("ok"):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
