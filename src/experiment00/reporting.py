"""Automatic English reports for the fiber-length optimization pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiment00.config import Experiment00Config
from experiment00.length_selection import SelectionDecision
from experiment00.validation import ValidationResult


def _decision_narrative_en(d: SelectionDecision) -> str:
    L = d.selected_length_cm
    if d.decision_type == "unique_optimum":
        return (
            f"Under the pre-registered maximin robust-gap criterion, {L} cm achieves the highest "
            f"worst-case separation across challenge and device impostors, with bootstrap selection "
            f"probability {d.bootstrap_selection_probability}."
        )
    if d.decision_type == "broad_plateau":
        return (
            f"Lengths {d.plateau_lengths} form a statistically comparable operating plateau. "
            f"{L} cm was selected from the plateau for compactness / lower loss without claiming "
            f"significant superiority over other plateau members."
        )
    if d.decision_type == "pareto_knee":
        return (
            f"Spatial complexity rises with length while remount robustness declines; "
            f"{L} cm is reported as the Pareto knee rather than maximum entropy."
        )
    return (
        f"No practically meaningful length dependence was detected across 7–15 cm. "
        f"{L} cm is an engineering selection for integration, not a statistical optimum."
    )


def write_validation_failure_reports(
    cfg: Experiment00Config, val: ValidationResult, run_dir: Path
) -> None:
    text = (
        f"# Fiber-length optimization: validation failed\n\n"
        f"Expected {val.expected_total} videos, found {val.found_total}.\n\n"
        f"Missing keys: {len(val.missing)}\n"
        f"Duplicates: {len(val.duplicates)}\n"
        f"Parse errors: {len(val.parse_errors)}\n\n"
        f"Messages:\n" + "\n".join(f"- {m}" for m in val.messages) + "\n"
    )
    (run_dir / "report_en.md").write_text(text, encoding="utf-8")
    (run_dir / "RESULTS_TEXT.md").write_text(
        "Dataset incomplete; formal length-selection statistics were not run. See missing_files.csv.\n",
        encoding="utf-8",
    )


def write_all_reports(
    *,
    cfg: Experiment00Config,
    run_dir: Path,
    validation: ValidationResult,
    length_metrics: pd.DataFrame,
    boot_sel: pd.DataFrame,
    decision: SelectionDecision,
    pair_counts: list[dict],
    dark_prov: dict,
    mask_meta: dict,
) -> None:
    en_narr = _decision_narrative_en(decision)
    metrics_md = length_metrics.to_markdown(index=False) if not length_metrics.empty else "(empty)"
    boot_md = boot_sel.to_markdown(index=False) if not boot_sel.empty else "(empty)"

    report_en = f"""# Fiber-length optimization report

## 1. Objective
How does propagation length regulate the spatial-complexity / authentication-robustness
trade-off of the side-polished polymer-fiber PUF, and which length defines the physical
operating window of the dual-channel system. The pipeline does **not** presuppose that 9 cm is optimal.

## 2. Dataset
- expected videos: {validation.expected_total}
- parsed successfully: {validation.found_total}
- exploratory: {validation.exploratory}
- challenge macro-pixel m = {cfg.dataset.challenge_macro_pixel} (frozen)

## 3. Preprocessing
- trim: {cfg.video.trim_start_s}s / {cfg.video.trim_end_s}s (start / end)
- temporal blocks: {cfg.video.temporal_blocks} (median)
- detail: sigma={cfg.preprocessing.detail_sigma_px}, epsilon={cfg.preprocessing.detail_epsilon}
- common-mode grouping: {cfg.preprocessing.common_mode_grouping}
- enrollment = A, query = B (common modes are never mixed)
- dark reference: {dark_prov}
- global mask coverage: {mask_meta.get('coverage')}

## 4. Definitions
- S_intra = NCC(detail_cm[L,f,A,c], detail_cm[L,f,B,c])
- robust_gap = Q05(S_intra) - Q95(S_impostor)
- selection: L* = argmax min(robust_gap_challenge, robust_gap_device)

## 5. Per-length metrics
{metrics_md}

## 6. Bootstrap selection probability
{boot_md}

## 7. Decision
- type: `{decision.decision_type}`
- selected_length_cm: `{decision.selected_length_cm}`
- criterion: {decision.criterion}
- evidence: {decision.evidence_summary}
- {en_narr}

## 8. Uncertainty and limitations
- Only five fibers per length; hierarchical bootstrap intervals are wide.
- Temporal blocks are never treated as independent devices.
- The red channel is a readout guardrail only and is never used to correct the green fingerprint.
- {decision.limitations}

## 9. Interpretation limits
The correlation-area approximation of N_eff must not be described as a strict optical mode count.
"""
    (run_dir / "report_en.md").write_text(report_en, encoding="utf-8")
    (run_dir / "RESULTS_TEXT.md").write_text(en_narr + "\n", encoding="utf-8")
