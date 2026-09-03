"""Lifecycle markdown reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_lifecycle_reports(run_dir: Path, summary: dict[str, Any]) -> None:
    full = [
        "# Experiment 4 lifecycle analysis",
        "",
        "Red channel = persistent device identity.",
        "Green channel = challenge- and mounting-state-bound credential.",
        "Remounting is expected to revoke the previous green credential.",
        "Re-enrollment acquires and independently verifies a new green template;",
        "it is not machine-learning training.",
        "",
        f"Run directory: `{run_dir}`",
        f"Dry-run: {summary.get('dry_run')}",
        f"Claim label: {summary.get('claim_label')}",
        "",
        "## Thresholds",
        "```json",
        json.dumps(summary.get("thresholds", {}), indent=2),
        "```",
        "",
        "## Lifecycle metrics",
        "```json",
        json.dumps(summary.get("lifecycle_metrics", summary.get("inventory", {})), indent=2),
        "```",
        "",
        "## Frozen-test results",
        "```json",
        json.dumps(summary.get("frozen_test_results", {}), indent=2),
        "```",
    ]
    (run_dir / "full_report.md").write_text("\n".join(full), encoding="utf-8")
    (run_dir / "paper_ready_methods.md").write_text(
        "\n".join(
            [
                "# Methods (lifecycle)",
                "",
                "Green responses use OpenCV BGR channel index 1, local-ratio detail",
                "(sigma=42, eps=1), and per device×state×round common subtraction.",
                "Official green score is zero-mean NCC of detail_cm templates.",
                "",
                "Red responses use OpenCV BGR channel index 2 and a frozen 9-D",
                "Fiber-ID statistic vector with development-set standardization.",
                "Official red score is the negative standardized Euclidean distance.",
                "Thresholds and the session pass rule are selected on F01–F05 only.",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "paper_ready_results.md").write_text(
        "\n".join(
            [
                "# Results (lifecycle)",
                "",
                "See `lifecycle_metrics.json` and `frozen_test_results.json`.",
                "Do not interpret dry-run inventories as optical performance.",
            ]
        ),
        encoding="utf-8",
    )
