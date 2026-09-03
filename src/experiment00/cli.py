"""CLI entry points for the fiber-length optimization pipeline: validate | audit | analyze."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="experiment00", description="Fiber-length optimization (Fig. 4)")
    parser.add_argument("--config", required=True, help="Path to fiber_length_optimization.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_val = sub.add_parser("validate", help="Validate dataset (probe OpenCV by default)")
    p_val.add_argument("--no-probe", action="store_true", help="Skip OpenCV probing")

    p_aud = sub.add_parser("audit", help="Run information-leakage audit")

    p_an = sub.add_parser("analyze", help="Formal one-shot analysis")
    p_an.add_argument("--videos", default=None)
    p_an.add_argument("--smoke", action="store_true")
    p_an.add_argument("--allow-incomplete", action="store_true")
    p_an.add_argument("--allow-missing-dark", action="store_true")
    p_an.add_argument("--probe", action="store_true", help="Probe all videos during analyze validate")

    args = parser.parse_args(argv)

    from experiment00.config import load_config

    cfg = load_config(args.config)
    if getattr(args, "videos", None):
        cfg.paths.videos_dir = args.videos
    if getattr(args, "allow_incomplete", False):
        cfg.experiment.allow_incomplete = True
        cfg.experiment.strict_dataset = False
    if getattr(args, "allow_missing_dark", False):
        cfg.experiment.allow_missing_dark = True

    if args.cmd == "validate":
        from experiment00.validation import (
            print_validation_summary,
            validate_dataset,
            write_validation_tables,
        )

        out = cfg.outputs_path() / "_validation_latest"
        res = validate_dataset(cfg, probe_video=not bool(args.no_probe))
        write_validation_tables(res, out, cfg=cfg)
        print_validation_summary(res)
        print(f"tables -> {out}")
        return 0 if res.ok else 2

    if args.cmd == "audit":
        from experiment00.leakage_audit import run_leakage_audit

        out = cfg.outputs_path() / "_leakage_audit"
        rep = run_leakage_audit(cfg, out)
        print(f"LEAKAGE AUDIT ok={rep['ok']} -> {out}")
        return 0 if rep["ok"] else 3

    if args.cmd == "analyze":
        from experiment00.discovery import run_analysis
        from experiment00.validation import print_validation_summary, validate_dataset

        smoke = bool(getattr(args, "smoke", False))
        if smoke:
            print("NOTE: --smoke reduces bootstrap iterations (not for formal claims).")
        formal = (
            not cfg.experiment.allow_incomplete
            and not cfg.experiment.allow_missing_dark
            and not smoke
        )
        # Formal runs always probe OpenCV; exploratory may skip unless --probe
        do_probe = formal or bool(getattr(args, "probe", False))
        pre = validate_dataset(cfg, probe_video=do_probe)
        print_validation_summary(pre)
        if formal and not pre.ok:
            print("REFUSING formal analysis: validation failed. See FIX LIST above.")
            print("Do not use --allow-incomplete / --allow-missing-dark / --smoke for formal runs.")
            return 2
        run_dir = run_analysis(
            cfg,
            smoke=smoke,
            probe_on_validate=False,  # already probed above
            require_leakage_pass=True,
        )
        print(f"DONE -> {run_dir}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
