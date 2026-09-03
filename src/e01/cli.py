"""Macro-pixel screening CLI (Fig. 3, raw-video mode).

    python -m e01.cli generate-screening --config configs/macro_pixel_screening.yaml
    python -m e01.cli analyze-screening  --config configs/macro_pixel_screening.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from e01.analysis.green_screening import run_green_screening
from e01.config import load_config
from e01.naming import PathLayout
from e01.patterns.screening import (
    generate_crosstalk_slm_patterns,
    generate_screening_challenges,
    write_randomized_acquisition_order,
)
from e01.utils.io_utils import ensure_dir


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, required=True)
    common.add_argument("--force", action="store_true", help="Recompute even if outputs exist")
    parser = argparse.ArgumentParser(description="Green-only macro-pixel screening", parents=[common])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("generate-screening", parents=[common], help="Regenerate the C01-C08 screening challenges and acquisition order")
    sub.add_parser("analyze-screening", parents=[common], help="Score the screening videos and rank macro-pixel sizes")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    layout = PathLayout(cfg.root)
    ensure_dir(layout.logs)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(layout.logs / "run.log", encoding="utf-8")], force=True)
    log = logging.getLogger("e01")
    log.info("Experiment root: %s", cfg.root)

    if args.command == "generate-screening":
        generate_screening_challenges(cfg, force=args.force)
        generate_crosstalk_slm_patterns(cfg, force=args.force)
        write_randomized_acquisition_order(cfg, force=args.force)
        return 0
    paths = run_green_screening(cfg, force=args.force)
    log.info("Wrote %s", paths["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
