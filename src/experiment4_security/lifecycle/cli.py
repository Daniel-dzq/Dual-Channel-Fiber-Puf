"""Lifecycle analysis orchestration."""

from __future__ import annotations

import itertools
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from experiment4_security.common.config import LifecycleConfig
from experiment4_security.common.progress_util import stage_tqdm
from experiment4_security.common.video_io import process_video
from experiment4_security.lifecycle.discovery import load_experiment3_metadata
from experiment4_security.lifecycle.event_construction import (
    build_lifecycle_events,
    lifecycle_metrics,
)
from experiment4_security.lifecycle.evaluation import group_pair_metrics
from experiment4_security.lifecycle.green_credential import (
    GreenFeat,
    build_detail_cm_group,
    green_score,
)
from experiment4_security.lifecycle.red_identity import (
    apply_standardizer,
    extract_red_pack,
    fit_standardizer,
    red_score,
)
from experiment4_security.lifecycle.thresholds import select_session_rule, select_tau
from experiment4_security.lifecycle.plotting import generate_lifecycle_figures
from experiment4_security.lifecycle.reporting import write_lifecycle_reports
from puf_common.masks import build_valid_mask_from_refs, save_mask_png


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_lifecycle_analysis(
    cfg: LifecycleConfig,
    exclusions: set[tuple[str, str, str, str]] | None = None,
) -> dict[str, Any]:
    """Run the eight-challenge threshold-development analysis (Supplementary Note 7.1).

    ``exclusions`` is an optional set of ``(device_id, state_id, round_id, challenge_id)``
    green recordings that are removed *before* any processing. The public dataset's
    ``data_quality_exclusions.csv`` lists 16 invalid S1 copies whose removal defines the
    corrected S7.1 operating point (T_G, n_req); the remaining challenges of an affected
    (device, state, round) group form its common component.
    """
    run_dir = cfg.output_dir / "runs" / _timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)

    with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(
            {
                "threshold_development_root": str(cfg.threshold_development_root),
                "metadata_csv": str(cfg.metadata_csv),
                "videos_root": str(cfg.videos_root),
                "development_devices": cfg.development_devices,
                "frozen_test_devices": cfg.frozen_test_devices,
                "dry_run": cfg.dry_run,
                "envelope_sigma_px": cfg.envelope_sigma_px,
                "envelope_eps": cfg.envelope_eps,
            },
            fh,
            sort_keys=False,
        )

    meta = load_experiment3_metadata(cfg.metadata_csv, cfg.videos_root)
    if exclusions:
        key = list(zip(meta["device_id"], meta["state_id"], meta["round_id"], meta["challenge_id"]))
        drop = [k in exclusions for k in key]
        meta = meta.loc[[not d for d in drop]].reset_index(drop=True)
        pd.DataFrame(sorted(exclusions), columns=["device_id", "state_id", "round_id", "challenge_id"]).to_csv(
            run_dir / "excluded_recordings.csv", index=False
        )
    meta.to_csv(run_dir / "validated_metadata.csv", index=False)

    if cfg.dry_run:
        summary = _dry_run_summary(meta, cfg, run_dir)
        write_lifecycle_reports(run_dir, summary)
        return summary

    # Stream processing: mask refs, red, green
    dark = None
    qc_rows: list[dict[str, Any]] = []

    # Mask from red S0 + green C01 A
    refs: list[np.ndarray] = []
    ref_meta = meta[
        ((meta["channel"] == "red") & (meta["state_id"] == "S0"))
        | (
            (meta["channel"] == "green")
            & (meta["state_id"] == "S0")
            & (meta["round_id"] == "A")
            & (meta["challenge_id"].isin(["C01", "C001"]))
        )
    ]
    for row in stage_tqdm(list(ref_meta.itertuples(index=False)), desc="Mask refs", unit="vid"):
        rec = process_video(
            video_path=Path(row.video_path),
            channel=row.channel,
            dark=dark,
            device_id=row.device_id,
            state_id=row.state_id,
            round_id=getattr(row, "round_id", "") or "",
            challenge_id=getattr(row, "challenge_id", "") or "",
        )
        refs.append(rec.blocks[0])
        del rec
    mask, cov, thr = build_valid_mask_from_refs(refs, abs_floor=5.0, percentile=10.0)
    save_mask_png(mask, run_dir / "valid_mask.png")
    del refs

    # Red packs
    red_meta = meta[meta["channel"] == "red"]
    red_packs = []
    for row in stage_tqdm(list(red_meta.itertuples(index=False)), desc="Red", unit="vid"):
        rec = process_video(
            video_path=Path(row.video_path),
            channel="red",
            dark=dark,
            device_id=row.device_id,
            state_id=row.state_id,
        )
        qc_rows.append(
            {
                "device_id": rec.device_id,
                "state_id": rec.state_id,
                "channel": "red",
                "status": "fail" if rec.qc_flags else "ok",
                "reason": "|".join(rec.qc_flags) or "ok",
                "saturation_fraction": rec.saturation_fraction,
            }
        )
        red_packs.append(extract_red_pack(rec, mask))
        del rec

    # Green feats by group
    green_meta = meta[meta["channel"] == "green"]
    green_feats: dict[tuple[str, str, str, str], GreenFeat] = {}
    keys = (
        green_meta[["device_id", "state_id", "round_id"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    for device, state, rnd in stage_tqdm(list(keys), desc="Green detail_cm", unit="group"):
        sub = green_meta[
            (green_meta["device_id"] == device)
            & (green_meta["state_id"] == state)
            & (green_meta["round_id"] == rnd)
        ]
        by_cid = {}
        for row in sub.itertuples(index=False):
            rec = process_video(
                video_path=Path(row.video_path),
                channel="green",
                dark=dark,
                device_id=row.device_id,
                state_id=row.state_id,
                round_id=row.round_id,
                challenge_id=row.challenge_id,
            )
            qc_rows.append(
                {
                    "device_id": rec.device_id,
                    "state_id": rec.state_id,
                    "channel": "green",
                    "round_id": rec.round_id,
                    "challenge_id": rec.challenge_id,
                    "status": "fail" if rec.qc_flags else "ok",
                    "reason": "|".join(rec.qc_flags) or "ok",
                    "saturation_fraction": rec.saturation_fraction,
                }
            )
            by_cid[rec.challenge_id] = rec
        feats = build_detail_cm_group(
            by_cid,
            device_id=device,
            state_id=state,
            round_id=rnd,
            sigma=cfg.envelope_sigma_px,
            eps=cfg.envelope_eps,
        )
        for cid, feat in feats.items():
            green_feats[(device, state, rnd, cid)] = feat
        del by_cid, feats

    pd.DataFrame(qc_rows).to_csv(run_dir / "qc_report.csv", index=False)

    # Standardizer on development red only
    dev = set(cfg.development_devices)
    fit_vecs = [
        p["representative_vector"]
        for p in red_packs
        if p["device_id"] in dev
    ]
    mu, sd = fit_standardizer(fit_vecs)
    std = {
        (p["device_id"], p["state_id"]): apply_standardizer(p["representative_vector"], mu, sd)
        for p in red_packs
    }

    red_scores = _construct_red_scores(std)
    green_scores = _construct_green_scores(green_feats, mask)
    red_scores.to_csv(run_dir / "red_pair_scores.csv", index=False)
    green_scores.to_csv(run_dir / "green_pair_scores.csv", index=False)

    red_dev = red_scores[
        red_scores["device_id_a"].isin(dev) & red_scores["device_id_b"].isin(dev)
    ]
    green_dev = green_scores[
        green_scores["device_id_a"].isin(dev) & green_scores["device_id_b"].isin(dev)
    ]
    tau_r = select_tau(
        red_dev.loc[red_dev["group"] == "same_device_diff_state", "score"].to_numpy(),
        red_dev.loc[red_dev["group"] == "diff_device", "score"].to_numpy(),
    )
    tau_g = select_tau(
        green_dev.loc[
            green_dev["group"] == "same_device_same_state_same_challenge", "score"
        ].to_numpy(),
        green_dev.loc[
            green_dev["group"] == "same_device_same_state_diff_challenge", "score"
        ].to_numpy(),
    )
    session = select_session_rule(green_dev, tau_g)
    k = int(session["selected_k_of_8"])

    thr = {"tau_R": tau_r, "tau_G": tau_g, "k_of_8": k, "session_rule": session}
    (run_dir / "thresholds_frozen.json").write_text(json.dumps(thr, indent=2), encoding="utf-8")
    pd.DataFrame(session["candidates"]).to_csv(
        run_dir / "development_threshold_search.csv", index=False
    )

    green_metrics = [
        group_pair_metrics(
            green_scores,
            "same_device_same_state_same_challenge",
            "same_device_same_state_diff_challenge",
        ),
        group_pair_metrics(
            green_scores,
            "same_device_same_state_same_challenge",
            "same_device_diff_state_same_challenge",
        ),
        group_pair_metrics(
            green_scores,
            "same_device_same_state_same_challenge",
            "diff_device_same_challenge",
        ),
    ]
    red_metrics = [
        group_pair_metrics(red_scores, "same_device_diff_state", "diff_device"),
        group_pair_metrics(red_scores, "same_device_same_state", "diff_device"),
    ]
    pd.DataFrame(green_metrics).to_csv(run_dir / "green_metrics.csv", index=False)
    pd.DataFrame(red_metrics).to_csv(run_dir / "red_metrics.csv", index=False)

    devices = sorted(meta["device_id"].unique())
    events = build_lifecycle_events(
        devices=devices,
        red_scores=red_scores,
        green_scores=green_scores,
        tau_r=tau_r,
        tau_g=tau_g,
        k_of_8=k,
    )
    events.to_csv(run_dir / "lifecycle_events.csv", index=False)
    life = lifecycle_metrics(events, cfg.frozen_test_devices)
    (run_dir / "lifecycle_metrics.json").write_text(json.dumps(life, indent=2), encoding="utf-8")

    froz_events = events[events["device_id"].isin(cfg.frozen_test_devices)]
    frozen_results = {
        "n_events": int(len(froz_events)),
        "authenticated_reenrollment_success": float(
            froz_events["authenticated_reenrollment_success"].mean()
        )
        if len(froz_events)
        else float("nan"),
        "identity_continuity": float(froz_events["red_pass"].mean()) if len(froz_events) else float("nan"),
        "revocation_rate": float(froz_events["previous_green_fail"].mean())
        if len(froz_events)
        else float("nan"),
    }
    (run_dir / "frozen_test_results.json").write_text(
        json.dumps(frozen_results, indent=2), encoding="utf-8"
    )
    events.to_csv(run_dir / "per_device_audit.csv", index=False)

    generate_lifecycle_figures(figures, red_scores, green_scores, events, thr)
    summary = {
        "run_dir": str(run_dir),
        "dry_run": False,
        "thresholds": thr,
        "lifecycle_metrics": life,
        "frozen_test_results": frozen_results,
        "mask_coverage": float(cov),
        "n_red": len(red_packs),
        "n_green": len(green_feats),
        "claim_label": "FORMAL",
    }
    write_lifecycle_reports(run_dir, summary)
    return summary


def _construct_red_scores(std: dict[tuple[str, str], np.ndarray]) -> pd.DataFrame:
    rows = []
    keys = list(std.keys())
    for (d1, s1), (d2, s2) in itertools.combinations(keys, 2):
        score = red_score(std[(d1, s1)], std[(d2, s2)])
        if d1 == d2 and s1 == s2:
            group = "same_device_same_state"
        elif d1 == d2:
            group = "same_device_diff_state"
        else:
            group = "diff_device"
        rows.append(
            {
                "device_id_a": d1,
                "state_id_a": s1,
                "device_id_b": d2,
                "state_id_b": s2,
                "group": group,
                "score": score,
            }
        )
    return pd.DataFrame(rows)


def _construct_green_scores(
    feats: dict[tuple[str, str, str, str], GreenFeat],
    mask: np.ndarray,
) -> pd.DataFrame:
    rows = []
    keys = list(feats.keys())
    for ka, kb in itertools.combinations(keys, 2):
        fa, fb = feats[ka], feats[kb]
        # Prefer A vs B comparisons for same challenge
        score = green_score(fa, fb, mask)
        if fa.device_id == fb.device_id and fa.state_id == fb.state_id:
            if fa.challenge_id == fb.challenge_id and fa.round_id != fb.round_id:
                group = "same_device_same_state_same_challenge"
            elif fa.challenge_id != fb.challenge_id and fa.round_id == fb.round_id:
                group = "same_device_same_state_diff_challenge"
            else:
                continue
        elif (
            fa.device_id == fb.device_id
            and fa.state_id != fb.state_id
            and fa.challenge_id == fb.challenge_id
            and fa.round_id == fb.round_id
        ):
            group = "same_device_diff_state_same_challenge"
        elif (
            fa.device_id != fb.device_id
            and fa.state_id == fb.state_id
            and fa.challenge_id == fb.challenge_id
            and fa.round_id == fb.round_id
        ):
            group = "diff_device_same_challenge"
        else:
            continue
        rows.append(
            {
                "device_id_a": fa.device_id,
                "state_id_a": fa.state_id,
                "round_id_a": fa.round_id,
                "challenge_id_a": fa.challenge_id,
                "device_id_b": fb.device_id,
                "state_id_b": fb.state_id,
                "round_id_b": fb.round_id,
                "challenge_id_b": fb.challenge_id,
                "group": group,
                "score": score,
            }
        )
    return pd.DataFrame(rows)


def _dry_run_summary(meta: pd.DataFrame, cfg: LifecycleConfig, run_dir: Path) -> dict[str, Any]:
    inv = {
        "n_rows": int(len(meta)),
        "n_devices": int(meta["device_id"].nunique()),
        "devices": sorted(meta["device_id"].unique().tolist()),
        "n_green": int((meta["channel"] == "green").sum()),
        "n_red": int((meta["channel"] == "red").sum()),
        "expected_lifecycle_events": int(meta["device_id"].nunique() * 2),
    }
    summary = {
        "run_dir": str(run_dir),
        "dry_run": True,
        "inventory": inv,
        "development_devices": cfg.development_devices,
        "frozen_test_devices": cfg.frozen_test_devices,
        "claim_label": "DRY_RUN",
        "note": "Dry-run validated metadata only; no optical scores computed.",
    }
    (run_dir / "lifecycle_metrics.json").write_text(
        json.dumps({"dry_run": True, "inventory": inv}, indent=2), encoding="utf-8"
    )
    (run_dir / "thresholds_frozen.json").write_text(
        json.dumps({"dry_run": True}, indent=2), encoding="utf-8"
    )
    (run_dir / "frozen_test_results.json").write_text(
        json.dumps({"dry_run": True}, indent=2), encoding="utf-8"
    )
    pd.DataFrame(columns=["device_id"]).to_csv(run_dir / "lifecycle_events.csv", index=False)
    return summary


def exclusions_from_manifest(path: Path) -> set[tuple[str, str, str, str]]:
    """Recordings flagged ``EXCLUDE_FROM_INDEPENDENT_S1_ANALYSES`` in ``data_quality_exclusions.csv``."""
    df = pd.read_csv(path)
    drop = df[df["corrected_analysis_treatment"].eq("EXCLUDE_FROM_INDEPENDENT_S1_ANALYSES")]
    return {(r.device, r.state, r.round, r.challenge) for r in drop.itertuples(index=False)}


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Eight-challenge threshold-development analysis (Supplementary Note 7.1)")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=None, help="Zenodo directory holding data_quality_exclusions.csv")
    ap.add_argument("--exclusions", type=Path, default=None, help="Explicit path to data_quality_exclusions.csv")
    ap.add_argument("--no-exclusions", action="store_true", help="Reproduce the pre-correction (frozen) run instead")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = LifecycleConfig.from_yaml(args.config)
    cfg.dry_run = cfg.dry_run or args.dry_run
    excl: set[tuple[str, str, str, str]] = set()
    if not args.no_exclusions:
        manifest = args.exclusions or (args.data_root / "data_quality_exclusions.csv" if args.data_root else None)
        if manifest is None or not Path(manifest).is_file():
            ap.error("data_quality_exclusions.csv is required (pass --data-root or --exclusions, or --no-exclusions)")
        excl = exclusions_from_manifest(Path(manifest))
    summary = run_lifecycle_analysis(cfg, exclusions=excl or None)
    print("run_dir:", summary.get("run_dir"))
    print("excluded recordings:", len(excl))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
