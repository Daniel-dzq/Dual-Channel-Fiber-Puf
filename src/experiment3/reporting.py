"""Markdown reporting for Experiment 3 runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from experiment3.validation import ValidationResult


PREPROCESSING_RULES = r"""
## Exact preprocessing rules

1. Videos are read with OpenCV in BGR color; grayscale conversion is forbidden.
2. Red uses BGR channel index 2; green uses BGR channel index 1.
3. Dark correction is applied when a valid dark video/artifact is available;
   otherwise dark correction is skipped (configured `allow_missing_dark`).
4. One fixed valid mask is built from **development enrollment (F01–F05, S0) only**
   and applied to all scores. Frozen devices do not enter mask estimation.
5. Temporal protocol (frozen): discard first 10 s and last 10 s; retain the
   middle segment; split into three non-overlapping blocks; pixelwise median
   per block. Insufficient length fails validation rather than changing the rule.
6. Green local detail uses sigma = 42, epsilon = 1.
7. Green common response = mean of eight challenge detail representatives,
   constructed separately for each device × state × round. Never across rounds,
   states, or devices.
8. Green official score \(S_G\) = zero-mean NCC of `detail_cm` templates.
9. Red does not use challenge common subtraction.
10. Red primary identity = low-dimensional spatial/spectral statistics with
    standardized Euclidean distance and no PCA.
11. Official red score \(S_R = -\mathrm{standardized\ Euclidean}\).
12. Red standardizer is fit on **development S0 enrollment only**.
13. Operational thresholds \(\tau_R, \tau_G\) are fixed on the development cohort.
14. Samples are never excluded because authentication scores are poor.
15. F02/F03 were not constructed through score-based mixing or post-hoc
    selection of old and new red videos.
"""


SCORE_DEFINITIONS = r"""
## Score definitions

- \(S_R\): negative standardized Euclidean distance of the frozen red feature vector.
  Larger means more similar.
- \(S_G\): zero-mean normalized cross-correlation of green `detail_cm` templates.
  Larger means more similar.
"""


def _metric_blob(summary: dict[str, Any], key: str) -> dict[str, Any]:
    return summary.get(key) or {}


def write_scoped_reports(
    run_dir: Path,
    summary: dict[str, Any],
    *,
    validation: ValidationResult | None = None,
    formal_allowed: bool = False,
) -> None:
    """Write development / frozen / descriptive reports with explicit claim labels."""
    del validation  # reserved for future inventory dumps
    audit = summary.get("leakage_audit") or {}

    def _write(name: str, title: str, claim: str, metrics_key: str, extra: list[str]) -> None:
        lines = [f"# {title}", "", f"> Claim label: **{claim}**", ""]
        lines.extend(extra)
        lines.append("")
        lines.append("## Metrics")
        lines.append("```json")
        lines.append(json.dumps(_metric_blob(summary, metrics_key), indent=2, default=str))
        lines.append("```")
        lines.append("")
        (run_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    _write(
        "development_report.md",
        "Experiment 3 development report",
        "DEVELOPMENT / EXPLORATORY / NOT AN INDEPENDENT TEST",
        "development_metrics",
        [
            "F01–F05 were used exclusively for method development, preprocessing",
            "calibration, standardization, and threshold selection.",
            "These results are **not** held-out formal claims.",
            f"- Development readiness gate: `{summary.get('development_readiness_gate')}`",
            f"- tau_R_dev={summary.get('tau_R_dev')} tau_G_dev={summary.get('tau_G_dev')}",
        ],
    )

    if formal_allowed and summary.get("frozen_test_metrics") is not None:
        _write(
            "frozen_test_report.md",
            "Experiment 3 frozen held-out report",
            "FROZEN TEST / FORMAL HELD-OUT EVALUATION",
            "frozen_test_metrics",
            [
                "1. F01–F05 were used exclusively for method development, preprocessing calibration, standardization, and threshold selection.",
                "2. F06–F15 were reserved for frozen held-out evaluation.",
                "3. All formal performance claims were computed exclusively on F06–F15 using parameters fixed before Frozen evaluation.",
                "4. Results pooled over F01–F15 are descriptive only and are not treated as held-out estimates.",
                "5. The valid mask was estimated exclusively from development enrollment data.",
                "6. The red feature standardizer was fitted exclusively on development enrollment (S0) data.",
                "7. Operational thresholds were fixed on the development cohort and were not re-estimated on the Frozen cohort.",
                "8. Frozen-set AUC/EER may be reported as descriptive separability summaries, but system FAR/FRR use development-fixed thresholds.",
                "9. No authentication-score-based sample exclusion was performed.",
                "10. F02/F03 were not constructed through score-based mixing or post-hoc selection of old and new red videos.",
                "",
                "No direct leakage from the Frozen cohort into threshold estimation or red-feature standardization was detected. The reporting pipeline was revised so that formal claims are based exclusively on the frozen F06–F15 cohort.",
                "",
                f"- Frozen confirmation: `{summary.get('frozen_confirmation')}`",
                f"- Leakage audit: {audit.get('overall_status')}",
            ],
        )
    else:
        (run_dir / "frozen_test_report.md").write_text(
            "\n".join(
                [
                    "# Experiment 3 frozen held-out report",
                    "",
                    "> **FORMAL REPORT BLOCKED**",
                    "",
                    f"- formal_allowed={formal_allowed}",
                    f"- leakage_audit={audit.get('overall_status')}",
                    f"- blocking_failures={audit.get('blocking_failures')}",
                    "",
                    "Formal held-out metrics were not emitted as a paper claim.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    _write(
        "all_cohort_descriptive_report.md",
        "Experiment 3 all-cohort descriptive report",
        "ALL-COHORT DESCRIPTIVE / NOT HELD-OUT / NOT USED FOR FORMAL CLAIMS",
        "all_cohort_descriptive_metrics",
        [
            "Pooled F01–F15 results are for cohort visualization and supplementary",
            "overview only. They must not be used for proceed, go/no-go, abstract",
            "performance, or main frozen claims.",
            f"- Descriptive diagnostics: `{summary.get('all_cohort_descriptive_diagnostics')}`",
            "- proceed: null (by design)",
        ],
    )


def write_full_report(
    run_dir: Path,
    summary: dict[str, Any],
    *,
    validation: ValidationResult | None = None,
    query_df: pd.DataFrame | None = None,
) -> Path:
    inv = summary.get("inventory", {})
    diag = summary.get("development_readiness_gate") or summary.get("diagnostics", {})
    lines: list[str] = []
    lines.append("# Experiment 3 full report")
    lines.append("")
    lines.append(f"- Run directory: `{run_dir}`")
    lines.append(f"- Dry run: {summary.get('dry_run')}")
    lines.append(f"- Claim label: {summary.get('claim_label', 'n/a')}")
    lines.append(
        f"- Leakage audit: {(summary.get('leakage_audit') or {}).get('overall_status')}"
    )
    lines.append("")

    lines.append("## Detected inventory")
    lines.append(f"- Devices: {inv.get('devices', [])}")
    lines.append(f"- States: {inv.get('states', [])}")
    lines.append(f"- Development devices: {summary.get('development_devices', [])}")
    lines.append(
        f"- Frozen-test devices present: {summary.get('frozen_test_devices_present', [])}"
    )
    lines.append("")

    if validation is not None:
        lines.append("## Missing or duplicated files")
        lines.append(f"- Missing ({len(validation.missing)}): {validation.missing[:50]}")
        lines.append(
            f"- Duplicates ({len(validation.duplicates)}): {validation.duplicates}"
        )
        lines.append("")

    lines.append(PREPROCESSING_RULES.strip())
    lines.append("")
    lines.append(SCORE_DEFINITIONS.strip())
    lines.append("")

    lines.append("## Metrics")
    lines.append("")
    lines.append(
        "Prefer `development_metrics.json`, `frozen_test_metrics.json`, and "
        "`all_cohort_descriptive_metrics.json`. Compat `system_metrics.json` is deprecated."
    )
    lines.append("")
    lines.append("### Frozen formal system metrics (if available)")
    lines.append("```json")
    frozen = summary.get("frozen_test_metrics") or {}
    lines.append(json.dumps(frozen.get("system") or {}, indent=2, default=str))
    lines.append("```")
    lines.append("")
    lines.append("### Development system metrics")
    lines.append("```json")
    devm = summary.get("development_metrics") or {}
    lines.append(json.dumps(devm.get("system") or {}, indent=2, default=str))
    lines.append("```")
    lines.append("")
    lines.append("### Compat / deprecated system_metrics")
    lines.append("```json")
    lines.append(json.dumps(summary.get("system_metrics", {}), indent=2, default=str))
    lines.append("```")
    lines.append("")

    if query_df is not None and not query_df.empty:
        lines.append("## Per-query identity ranks and margins")
        cols = [
            c
            for c in (
                "query_device",
                "query_state",
                "predicted_top1",
                "rank_correct",
                "identity_margin",
                "score_correct",
                "score_best_wrong",
            )
            if c in query_df.columns
        ]
        try:
            lines.append(query_df[cols].to_markdown(index=False))
        except Exception:  # noqa: BLE001
            lines.append(query_df[cols].to_string(index=False))
        lines.append("")

    lines.append("## Split / leakage")
    lines.append(
        "- F02/F03 were not constructed through score-based mixing or post-hoc "
        "selection of old and new red videos."
    )
    lines.append(
        "- Mask fit: development enrollment only. Standardizer fit: development S0 only."
    )
    lines.append("")

    lines.append("## Development readiness gate (Dev only)")
    for k, v in (diag or {}).items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Frozen confirmation (Frozen only; no retuning)")
    for k, v in (summary.get("frozen_confirmation") or {}).items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    path = run_dir / "full_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def print_terminal_summary(summary: dict[str, Any]) -> None:
    diag = summary.get("development_readiness_gate") or summary.get("diagnostics", {})
    inv = summary.get("inventory", {})
    print("")
    print("=" * 60)
    print("Experiment 3 summary")
    print("=" * 60)
    print(f"Claim: {summary.get('claim_label')}")
    print(f"Leakage audit: {(summary.get('leakage_audit') or {}).get('overall_status')}")
    if summary.get("is_pilot"):
        print("LABEL: PILOT / NOT FOR FINAL CLAIMS")
    print(f"Devices: {inv.get('devices')}  States: {inv.get('states')}")
    print(f"Run: {summary.get('run_dir')}")
    print("-" * 60)
    print("Development readiness gate:")
    for key in (
        "RED_IDENTITY",
        "GREEN_SAME_STATE_RELIABILITY",
        "GREEN_STATE_REVOCATION",
        "FULL_IDENTITY_STATE_STORY",
    ):
        print(f"  {key}: {diag.get(key, 'n/a')}")
    print("=" * 60)
