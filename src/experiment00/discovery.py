"""End-to-end Experiment 00 analysis pipeline (English-only)."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from experiment00.authentication import (
    challenge_top1,
    device_top1,
    length_separation_metrics,
    summarize_top1,
)
from experiment00.bootstrap import hierarchical_bootstrap_length_metrics
from experiment00.cache import dump_json
from experiment00.config import Experiment00Config, save_frozen_config
from experiment00.leakage_audit import run_leakage_audit
from experiment00.length_selection import select_length
from experiment00.mask import build_global_valid_mask
from experiment00.metadata import parse_filename
from experiment00.sanity_check import run_post_run_sanity
from experiment00.score_construction import (
    block_repeatability_scores,
    construct_pair_scores,
    expected_pair_counts,
)
from experiment00.spatial_statistics import compute_spatial_metrics_table, summarize_spatial_by_length
from experiment00.statistics import red_pair_metrics
from experiment00.templates import build_green_detail_cm_group, group_green_recordings
from experiment00.validation import print_validation_summary, validate_dataset, write_validation_tables
from experiment00.video_processing import load_dark_mean, process_video_file

LOGGER = logging.getLogger("experiment00")


def _setup_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def _manifest_hash(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: str(x)):
        st = p.stat()
        h.update(p.name.encode())
        h.update(str(st.st_size).encode())
        h.update(str(int(st.st_mtime)).encode())
    return h.hexdigest()


def assert_pair_and_query_counts(
    pairs: pd.DataFrame,
    dtop: pd.DataFrame,
    ctop: pd.DataFrame,
    lengths: list[int],
) -> None:
    exp = expected_pair_counts(5, 8)
    errors = []
    for L in lengths:
        sub = pairs.loc[pairs.length_cm == L]
        for st, n_exp in exp.items():
            n = int((sub.score_type == st).sum())
            if n != n_exp:
                errors.append(f"length {L}cm {st}: got {n}, expected {n_exp}")
        nd = int((dtop.length_cm == L).sum()) if not dtop.empty else 0
        nc = int((ctop.length_cm == L).sum()) if not ctop.empty else 0
        if nd != 40:
            errors.append(f"length {L}cm device Top-1 queries: got {nd}, expected 40")
        if nc != 40:
            errors.append(f"length {L}cm challenge Top-1 queries: got {nc}, expected 40")
    # Totals across five lengths
    if len(lengths) == 5:
        totals = {
            "S_intra": int((pairs.score_type == "S_intra").sum()),
            "S_inter_challenge": int((pairs.score_type == "S_inter_challenge").sum()),
            "S_inter_device": int((pairs.score_type == "S_inter_device").sum()),
        }
        expect_tot = {"S_intra": 200, "S_inter_challenge": 700, "S_inter_device": 400}
        for k, v in expect_tot.items():
            if totals[k] != v:
                errors.append(f"total {k}: got {totals[k]}, expected {v}")
        if int(len(dtop)) != 200:
            errors.append(f"total device Top-1 queries: got {len(dtop)}, expected 200")
        if int(len(ctop)) != 200:
            errors.append(f"total challenge Top-1 queries: got {len(ctop)}, expected 200")
    if errors:
        raise AssertionError("Pair/query count assertion failed:\n- " + "\n- ".join(errors))


def print_final_summary(
    *,
    val,
    dark_prov: dict,
    mask_meta: dict,
    pairs: pd.DataFrame,
    length_metrics: pd.DataFrame,
    decision,
    boot_sel: pd.DataFrame,
    red: pd.DataFrame,
    run_dir: Path,
    figure_paths: dict,
) -> None:
    print("=" * 72)
    print("Experiment 00 — FORMAL RUN SUMMARY")
    print("=" * 72)
    print(f"validation_ok: {val.ok}  videos: {val.found_total}/{val.expected_total}")
    print(f"dark_field: {dark_prov}")
    print(f"mask_coverage: {mask_meta.get('coverage')}")
    if not pairs.empty:
        print("pair counts:", pairs.groupby("score_type").size().to_dict())
    if not length_metrics.empty:
        cols = [
            c
            for c in [
                "length_cm",
                "median_intra",
                "robust_gap_challenge",
                "robust_gap_device",
                "device_top1",
                "challenge_top1",
            ]
            if c in length_metrics.columns
        ]
        print(length_metrics[cols].to_string(index=False))
    if not red.empty:
        print("red NCC medians:", red.groupby("length_cm")["S_R_intra"].median().to_dict())
    print(
        f"selected_length_cm={decision.selected_length_cm} "
        f"decision_type={decision.decision_type} "
        f"bootstrap_p={decision.bootstrap_selection_probability}"
    )
    print(f"plateau/competing: {decision.plateau_lengths or decision.competing_lengths}")
    print(f"run_dir: {run_dir}")
    print(f"report_en: {run_dir / 'report_en.md'}")
    print(f"all_metrics.json: {run_dir / 'all_metrics.json'}")
    print(f"figures: {figure_paths}")
    print("=" * 72)


def run_analysis(
    cfg: Experiment00Config,
    *,
    smoke: bool = False,
    run_id: str | None = None,
    require_leakage_pass: bool = True,
    probe_on_validate: bool = False,
) -> Path:
    ts = run_id or datetime.now().strftime("%Y%m%d_%H%M%S") + "_length_optimization"
    run_dir = cfg.outputs_path() / ts
    _setup_logging(run_dir)
    save_frozen_config(cfg, run_dir / "config_frozen.yaml")

    LOGGER.info("Validating dataset in %s", cfg.videos_path())
    val = validate_dataset(cfg, probe_video=probe_on_validate)
    write_validation_tables(val, run_dir, cfg=cfg)
    print_validation_summary(val)
    if not val.ok:
        LOGGER.error("Validation failed under strict mode.")
        dump_json(
            run_dir / "analysis_manifest.json",
            {
                "status": "validation_failed",
                "expected_total": val.expected_total,
                "found_total": val.found_total,
                "missing": val.missing,
                "duplicates": val.duplicates,
                "exploratory": val.exploratory,
                "fix_list": val.fix_list,
            },
        )
        from experiment00.reporting import write_validation_failure_reports

        write_validation_failure_reports(cfg, val, run_dir)
        return run_dir

    # Leakage audit gate
    leak = run_leakage_audit(cfg, run_dir)
    if require_leakage_pass and not leak.get("ok", False):
        raise RuntimeError("Information-leakage audit FAILED; refusing formal analysis.")

    try:
        dark, dark_prov = load_dark_mean(cfg)
    except FileNotFoundError:
        if cfg.experiment.allow_missing_dark:
            dark, dark_prov = None, {"used": False, "warning": "missing_dark_exploratory"}
            LOGGER.warning("Continuing without dark field (exploratory).")
        else:
            raise

    inv = val.inventory
    inv_ok = inv.loc[inv["parse_status"] == "ok"].copy()
    paths = [Path(p) for p in inv_ok["path"].tolist()]
    dump_json(
        run_dir / "analysis_manifest.json",
        {
            "status": "running",
            "n_videos": len(paths),
            "manifest_hash": _manifest_hash(paths),
            "exploratory": val.exploratory,
            "smoke": smoke,
        },
    )

    LOGGER.info("Processing %d videos…", len(paths))
    green_recs = []
    red_recs = []
    qc_rows = []
    shapes = set()
    for _, row in tqdm(list(inv_ok.iterrows()), desc="Process videos", unit="vid"):
        parsed = parse_filename(
            row["filename"],
            assignment=cfg.acquisition.sequence_assignment,
            sequences=cfg.acquisition.challenge_sequences,
        )
        try:
            rec = process_video_file(
                path=Path(row["path"]),
                length_cm=parsed.length_cm,
                fiber_id=parsed.fiber_id,
                illumination=parsed.illumination,
                round_id=parsed.round,
                challenge=parsed.challenge_id or parsed.challenge,
                challenge_id=parsed.challenge_id,
                sequence_id=parsed.sequence_id,
                acquisition_position=parsed.acquisition_position,
                acquisition_position_zero_based=parsed.acquisition_position_zero_based,
                order_source=parsed.order_source,
                red_phase=parsed.red_phase,
                dark=dark,
                cfg=cfg,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed processing %s", row["filename"])
            qc_rows.append({"filename": row["filename"], "status": "fail", "error": str(exc)})
            if not cfg.experiment.allow_incomplete:
                raise
            continue
        shapes.add((rec.width, rec.height))
        qc_rows.append(
            {
                "filename": row["filename"],
                "status": "ok",
                "width": rec.width,
                "height": rec.height,
                "saturation_fraction": rec.saturation_fraction,
                "qc_flags": "|".join(rec.qc_flags),
                "challenge_id": rec.challenge_id,
                "sequence_id": rec.sequence_id,
                "acquisition_position": rec.acquisition_position,
                "order_source": rec.order_source,
            }
        )
        if rec.illumination == "green":
            green_recs.append(rec)
        else:
            red_recs.append(rec)

    pd.DataFrame(qc_rows).to_csv(run_dir / "video_qc.csv", index=False)
    if len(shapes) > 1 and cfg.quality_control.fail_on_resolution_mismatch:
        raise RuntimeError(f"Resolution mismatch across videos: {shapes}")

    LOGGER.info("Building global valid mask…")
    mask, mask_meta = build_global_valid_mask(green_recs, cfg, run_dir)
    dump_json(run_dir / "mask_qc.json", {"dark_provenance": dark_prov, "mask": mask_meta})

    LOGGER.info("Building detail_cm templates (grouped; no A/B leakage)…")
    template_map = {}
    groups = group_green_recordings(green_recs)
    for key, recs in tqdm(list(groups.items()), desc="detail_cm groups", unit="grp"):
        L, f, rnd = key
        built = build_green_detail_cm_group(recs, cfg)
        for ch, t in built.items():
            template_map[(L, f, rnd, ch)] = t

    pd.DataFrame(
        [
            {
                "length_cm": L,
                "fiber_id": f,
                "round": rnd,
                "challenge": ch,
                "challenge_id": t.challenge_id,
                "sequence_id": t.sequence_id,
                "acquisition_position": t.acquisition_position,
                "order_source": t.order_source,
                "shape": list(t.video_detail_cm.shape),
            }
            for (L, f, rnd, ch), t in template_map.items()
        ]
    ).to_csv(run_dir / "template_index.csv", index=False)

    lengths = cfg.dataset.expected_lengths_cm
    fibers = cfg.dataset.expected_fibers
    challenges = cfg.dataset.expected_challenges

    LOGGER.info("Constructing pair scores…")
    pairs = construct_pair_scores(template_map, mask, lengths, fibers, challenges, cfg=cfg)
    pairs.to_csv(run_dir / "pair_scores.csv", index=False)
    try:
        pairs.to_parquet(run_dir / "pair_scores.parquet", index=False)
    except Exception:  # noqa: BLE001
        LOGGER.warning("Parquet export unavailable.")

    blocks = block_repeatability_scores(template_map, mask, cfg=cfg)
    blocks.to_csv(run_dir / "block_repeatability.csv", index=False)

    LOGGER.info("Authentication metrics…")
    sep = length_separation_metrics(pairs)
    dtop = device_top1(template_map, mask, lengths, fibers, challenges)
    ctop = challenge_top1(template_map, mask, lengths, fibers, challenges)
    assert_pair_and_query_counts(pairs, dtop, ctop, lengths)
    preds = pd.concat([dtop, ctop], ignore_index=True)
    preds.to_csv(run_dir / "authentication_predictions.csv", index=False)
    top_sum = summarize_top1(preds)

    spatial = compute_spatial_metrics_table(green_recs, mask, cfg)
    spatial.to_csv(run_dir / "spatial_metrics_per_video.csv", index=False)
    spatial_len = summarize_spatial_by_length(spatial)
    spatial_len.to_csv(run_dir / "spatial_metrics_per_length.csv", index=False)
    if not spatial.empty:
        spatial.groupby(["length_cm", "fiber_id"], as_index=False).median(numeric_only=True).to_csv(
            run_dir / "spatial_metrics_per_fiber.csv", index=False
        )

    from experiment00.order_effect import compute_order_and_challenge_effects
    from experiment00.preflight import run_green_repeatability_preflight
    from experiment00.sequence_balance import run_sequence_balance_check

    run_sequence_balance_check(inv, cfg, run_dir)
    compute_order_and_challenge_effects(
        cfg=cfg,
        templates=template_map,
        mask=mask,
        spatial=spatial,
        block_scores=blocks,
        out_dir=run_dir,
    )
    run_green_repeatability_preflight(
        cfg=cfg, templates=template_map, mask=mask, out_dir=run_dir / "preflight"
    )

    red = red_pair_metrics(red_recs, mask)
    red.to_csv(run_dir / "red_reference_metrics.csv", index=False)

    length_metrics = sep.copy()
    if not top_sum.empty:
        for task, col in [("device_top1", "device_top1"), ("challenge_top1", "challenge_top1")]:
            t = top_sum.loc[top_sum.task == task, ["length_cm", "top1"]].rename(columns={"top1": col})
            length_metrics = length_metrics.merge(t, on="length_cm", how="left")
    if not spatial_len.empty and "n_eff_median" in spatial_len.columns:
        length_metrics = length_metrics.merge(
            spatial_len[["length_cm", "n_eff_median", "acf_width_px_median", "snr_median"]],
            on="length_cm",
            how="left",
        )
    length_metrics.to_csv(run_dir / "length_metrics.csv", index=False)

    n_boot = cfg.statistics.bootstrap_iterations_smoke if smoke else cfg.statistics.bootstrap_iterations
    LOGGER.info("Hierarchical bootstrap (%d iterations)…", n_boot)
    boot_sum, boot_sel, pairwise = hierarchical_bootstrap_length_metrics(
        pairs,
        n_iterations=n_boot,
        seed=cfg.experiment.seed,
        confidence_level=cfg.statistics.confidence_level,
    )
    boot_sum.to_csv(run_dir / "bootstrap_metrics.csv", index=False)
    boot_sel.to_csv(run_dir / "bootstrap_selection.csv", index=False)
    pairwise.to_csv(run_dir / "pairwise_length_probabilities.csv", index=False)

    decision = select_length(
        length_metrics,
        boot_sel,
        criterion=cfg.length_selection.criterion,
        never_force_unique_optimum=cfg.length_selection.never_force_unique_optimum,
        prefer_shorter_length_inside_plateau=cfg.length_selection.prefer_shorter_length_inside_plateau,
        selection_prob_unique_min=cfg.length_selection.selection_prob_unique_min,
        red_metrics=red,
        use_red_as_guardrail=cfg.length_selection.use_red_as_guardrail,
    )
    dump_json(run_dir / "selection_decision.json", decision.to_dict())

    figure_paths: dict = {}
    if not cfg.plotting.skip_figures:
        try:
            from experiment00.plotting import generate_all_figures

            figure_paths = generate_all_figures(
                cfg,
                run_dir,
                pairs,
                length_metrics,
                boot_sel,
                spatial,
                decision,
                templates=template_map,
                mask=mask,
            )
        except Exception:  # noqa: BLE001
            LOGGER.exception("Plotting failed")
    else:
        LOGGER.info("Skipping figure generation (plotting.skip_figures=true).")

    from experiment00.reporting import write_all_reports

    write_all_reports(
        cfg=cfg,
        run_dir=run_dir,
        validation=val,
        length_metrics=length_metrics,
        boot_sel=boot_sel,
        decision=decision,
        pair_counts=[],
        dark_prov=dark_prov,
        mask_meta=mask_meta,
    )

    sanity = run_post_run_sanity(
        run_dir=run_dir,
        pairs=pairs,
        length_metrics=length_metrics,
        boot_sel=boot_sel,
        decision=decision,
        red=red,
    )

    metrics_bundle = {
        "experiment_objective": (
            "How propagation length regulates spatial complexity vs authentication "
            "robustness for a side-polished polymer fiber PUF, and which operating "
            "window to freeze for later dual-channel experiments."
        ),
        "never_assumes_9cm_optimal": True,
        "dataset_summary": {
            "expected_total": val.expected_total,
            "found_total": val.found_total,
            "exploratory": val.exploratory,
        },
        "preprocessing": {
            "detail_sigma_px": cfg.preprocessing.detail_sigma_px,
            "detail_epsilon": cfg.preprocessing.detail_epsilon,
            "enrollment_round": "A",
            "query_round": "B",
            "dark": dark_prov,
            "mask": mask_meta,
        },
        "pair_counts_expected_per_length": expected_pair_counts(5, 8),
        "length_metrics": length_metrics.replace({np.nan: None}).to_dict(orient="records"),
        "bootstrap_selection": boot_sel.replace({np.nan: None}).to_dict(orient="records"),
        "selection_decision": decision.to_dict(),
        "leakage_audit_ok": leak.get("ok"),
        "post_run_sanity": sanity,
        "figure_paths": figure_paths,
        "figure_dir": str(run_dir / "figures"),
    }
    dump_json(run_dir / "all_metrics.json", metrics_bundle)
    dump_json(run_dir / "analysis_manifest.json", {"status": "completed", "smoke": smoke})

    print_final_summary(
        val=val,
        dark_prov=dark_prov,
        mask_meta=mask_meta,
        pairs=pairs,
        length_metrics=length_metrics,
        decision=decision,
        boot_sel=boot_sel,
        red=red,
        run_dir=run_dir,
        figure_paths=figure_paths,
    )
    LOGGER.info("Done → %s", run_dir)
    return run_dir
