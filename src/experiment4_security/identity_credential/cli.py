"""CLI for identity–credential prepare / validate / experiment4_security."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from experiment4_security.identity_credential.config import IdentityCredentialConfig
from experiment4_security.identity_credential.gates import GateError
from experiment4_security.identity_credential.pipeline import (
    check_code_readiness,
    run_formal,
    run_prepare,
    run_validate,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Experiment 4 identity–credential multi-device pipeline")
    p.add_argument("--config", type=Path, required=True)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare", help="Prepare-only dry-run (no formal science)")
    v = sub.add_parser("validate", help="Validate dataset against expected inventory")
    v.add_argument(
        "--strict",
        action="store_true",
        help="Require DATA_READY + full 10-device inventory counts before SUCCESS",
    )
    sub.add_parser("readiness", help="Import/code readiness check only")
    f = sub.add_parser("formal", help="Formal full run (requires latest SUCCESS validate)")
    f.add_argument(
        "--skip-validate-gate",
        action="store_true",
        help="Dev only: skip latest_validate.json gate (still re-validates inventory)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    cfg = IdentityCredentialConfig.from_yaml(args.config)

    try:
        if args.command == "readiness":
            r = check_code_readiness()
            print(json.dumps(r, indent=2))
            return 0 if r["ready"] else 2
        if args.command == "prepare":
            out = run_prepare(cfg)
            print(json.dumps(out, indent=2))
            return 0 if out.get("status") == "SUCCESS" else 2
        if args.command == "validate":
            audit = run_validate(cfg, strict=bool(getattr(args, "strict", False)))
            print(json.dumps(audit, indent=2, default=str))
            if getattr(args, "strict", False):
                return 0 if audit.get("status") == "SUCCESS" and audit.get("data_status") == "DATA_READY" else 3
            return 0 if audit.get("status") == "SUCCESS" else 3
        if args.command == "formal":
            out = run_formal(cfg, require_latest_validate=not bool(getattr(args, "skip_validate_gate", False)))
            print(json.dumps({k: out[k] for k in ("run_id", "run_dir", "status") if k in out}, indent=2))
            return 0 if out.get("status") == "SUCCESS" else 4
    except GateError as exc:
        logging.error("GATE: %s", exc)
        return 4
    except RuntimeError as exc:
        logging.error("RUNTIME: %s", exc)
        return 4
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
