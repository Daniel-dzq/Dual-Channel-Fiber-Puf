"""Automated leakage audit for Experiment 3 (fail-closed for formal claims)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _check(
    name: str,
    ok: bool,
    *,
    severity: str,
    evidence: str,
    affected: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": "PASS" if ok else "FAIL",
        "severity": severity,
        "evidence": evidence,
        "affected_outputs": affected or [],
    }


def run_leakage_audit(
    *,
    development_devices: list[str],
    frozen_devices: list[str],
    threshold_fit_devices: list[str],
    standardizer_fit_devices: list[str],
    standardizer_fit_states: list[str],
    mask_fit_devices: list[str],
    formal_metric_devices: list[str],
    development_metric_devices: list[str],
    pooled_metric_devices: list[str],
    readiness_gate_devices: list[str],
    frozen_confirmation_devices: list[str],
    gallery_states: list[str],
    query_states: list[str],
    gallery_keys: set[tuple[str, str]],
    query_keys: set[tuple[str, str]],
    gallery_video_ids: set[str],
    query_video_ids: set[str],
    common_groups: list[dict[str, Any]],
    qc_auth_score_used: bool,
    unsafe_fallback_triggered: bool,
    pooled_labeled_formal: bool,
    development_labeled_held_out: bool,
    frozen_eer_used_as_operational: bool,
    acquisition_has_false_mixing_claim: bool,
    mask_contains_frozen: bool,
    standardizer_contains_frozen: bool,
    pooled_proceed_is_none: bool,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    overlap = sorted(set(development_devices) & set(frozen_devices))
    checks.append(
        _check(
            "dev_and_frozen_disjoint",
            len(overlap) == 0,
            severity="high",
            evidence=f"overlap={overlap}",
            affected=["all"],
        )
    )
    checks.append(
        _check(
            "threshold_fit_dev_only",
            set(threshold_fit_devices).issubset(set(development_devices))
            and set(threshold_fit_devices).isdisjoint(set(frozen_devices))
            and len(threshold_fit_devices) > 0,
            severity="high",
            evidence=f"threshold_fit_devices={threshold_fit_devices}",
            affected=["tau_R", "tau_G"],
        )
    )
    checks.append(
        _check(
            "standardizer_fit_dev_only",
            set(standardizer_fit_devices).issubset(set(development_devices))
            and set(standardizer_fit_devices).isdisjoint(set(frozen_devices))
            and not standardizer_contains_frozen,
            severity="high",
            evidence=(
                f"fit_devices={standardizer_fit_devices}; "
                f"fit_states={standardizer_fit_states}"
            ),
            affected=["red_standardizer"],
        )
    )
    checks.append(
        _check(
            "standardizer_fit_s0_only",
            set(standardizer_fit_states) == {"S0"},
            severity="medium",
            evidence=f"fit_states={standardizer_fit_states}",
            affected=["red_standardizer", "development_metrics"],
        )
    )
    checks.append(
        _check(
            "mask_fit_dev_only",
            set(mask_fit_devices).issubset(set(development_devices))
            and set(mask_fit_devices).isdisjoint(set(frozen_devices))
            and not mask_contains_frozen,
            severity="high",
            evidence=f"mask_fit_devices={mask_fit_devices}",
            affected=["valid_mask"],
        )
    )
    checks.append(
        _check(
            "formal_metrics_frozen_only",
            set(formal_metric_devices) == set(frozen_devices),
            severity="high",
            evidence=f"formal={formal_metric_devices} frozen={frozen_devices}",
            affected=["frozen_test_metrics"],
        )
    )
    checks.append(
        _check(
            "development_metrics_dev_only",
            set(development_metric_devices) == set(development_devices),
            severity="high",
            evidence=f"dev_metrics={development_metric_devices}",
            affected=["development_metrics"],
        )
    )
    checks.append(
        _check(
            "pooled_metrics_all_cohort",
            set(pooled_metric_devices) == set(development_devices) | set(frozen_devices),
            severity="medium",
            evidence=f"pooled={pooled_metric_devices}",
            affected=["all_cohort_descriptive_metrics"],
        )
    )
    checks.append(
        _check(
            "proceed_uses_dev_only",
            set(readiness_gate_devices).issubset(set(development_devices))
            and set(readiness_gate_devices).isdisjoint(set(frozen_devices)),
            severity="high",
            evidence=f"readiness_gate_devices={readiness_gate_devices}",
            affected=["development_readiness_gate"],
        )
    )
    checks.append(
        _check(
            "frozen_confirmation_uses_frozen_only",
            set(frozen_confirmation_devices) == set(frozen_devices),
            severity="high",
            evidence=f"frozen_confirmation_devices={frozen_confirmation_devices}",
            affected=["frozen_confirmation"],
        )
    )
    checks.append(
        _check(
            "pooled_descriptive_has_no_proceed",
            pooled_proceed_is_none,
            severity="high",
            evidence=f"pooled_proceed_is_none={pooled_proceed_is_none}",
            affected=["all_cohort_descriptive"],
        )
    )
    checks.append(
        _check(
            "gallery_s0_only",
            set(gallery_states) == {"S0"},
            severity="high",
            evidence=f"gallery_states={gallery_states}",
        )
    )
    checks.append(
        _check(
            "query_s1_s2_only",
            set(query_states).issubset({"S1", "S2"}),
            severity="high",
            evidence=f"query_states={query_states}",
        )
    )
    checks.append(
        _check(
            "query_not_in_gallery_keys",
            gallery_keys.isdisjoint(query_keys),
            severity="high",
            evidence=(
                f"n_gallery_keys={len(gallery_keys)} n_query_keys={len(query_keys)} "
                f"overlap={sorted(gallery_keys & query_keys)[:5]}"
            ),
        )
    )
    checks.append(
        _check(
            "gallery_query_videos_disjoint",
            gallery_video_ids.isdisjoint(query_video_ids),
            severity="high",
            evidence=f"video_overlap={sorted(gallery_video_ids & query_video_ids)[:5]}",
        )
    )

    common_ok = True
    common_evidence = []
    for g in common_groups:
        if (
            g.get("n_devices", 1) != 1
            or g.get("n_states", 1) != 1
            or g.get("n_rounds", 1) != 1
            or g.get("n_roles", 1) != 1
        ):
            common_ok = False
            common_evidence.append(str(g))
    checks.append(
        _check(
            "common_subtraction_group_local",
            common_ok,
            severity="high",
            evidence="; ".join(common_evidence) if common_evidence else "all groups local",
            affected=["green_detail_cm"],
        )
    )
    checks.append(
        _check(
            "qc_does_not_use_auth_scores",
            not qc_auth_score_used,
            severity="high",
            evidence=f"auth_score_used_for_qc={qc_auth_score_used}",
        )
    )
    checks.append(
        _check(
            "no_unsafe_all_device_fallback",
            not unsafe_fallback_triggered,
            severity="high",
            evidence=f"unsafe_fallback_triggered={unsafe_fallback_triggered}",
        )
    )
    checks.append(
        _check(
            "pooled_not_labeled_formal",
            not pooled_labeled_formal,
            severity="high",
            evidence=f"pooled_labeled_formal={pooled_labeled_formal}",
        )
    )
    checks.append(
        _check(
            "development_not_labeled_held_out",
            not development_labeled_held_out,
            severity="high",
            evidence=f"development_labeled_held_out={development_labeled_held_out}",
        )
    )
    checks.append(
        _check(
            "frozen_eer_not_operational",
            not frozen_eer_used_as_operational,
            severity="high",
            evidence=f"frozen_eer_used_as_operational={frozen_eer_used_as_operational}",
        )
    )
    checks.append(
        _check(
            "acquisition_no_false_mixing_claim",
            not acquisition_has_false_mixing_claim,
            severity="medium",
            evidence=f"acquisition_has_false_mixing_claim={acquisition_has_false_mixing_claim}",
        )
    )

    blocking = {
        "threshold_fit_dev_only",
        "standardizer_fit_dev_only",
        "mask_fit_dev_only",
        "formal_metrics_frozen_only",
        "proceed_uses_dev_only",
        "query_not_in_gallery_keys",
        "gallery_query_videos_disjoint",
        "no_unsafe_all_device_fallback",
        "dev_and_frozen_disjoint",
    }
    failures = [c for c in checks if c["status"] == "FAIL"]
    blocking_failures = [c for c in failures if c["name"] in blocking]
    overall = "PASS" if not blocking_failures else "FAIL"
    return {
        "overall_status": overall,
        "n_checks": len(checks),
        "n_failures": len(failures),
        "n_blocking_failures": len(blocking_failures),
        "blocking_failures": [c["name"] for c in blocking_failures],
        "checks": checks,
        "allow_formal_report": overall == "PASS",
    }


def write_leakage_audit(run_dir: Path, audit: dict[str, Any]) -> None:
    (run_dir / "leakage_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    lines = [
        "# Leakage audit (Experiment 3)",
        "",
        f"- overall_status: **{audit['overall_status']}**",
        f"- allow_formal_report: {audit.get('allow_formal_report')}",
        f"- n_checks: {audit['n_checks']}",
        f"- n_failures: {audit['n_failures']}",
        f"- blocking_failures: {audit.get('blocking_failures', [])}",
        "",
        "## Checks",
        "",
    ]
    for c in audit["checks"]:
        lines.append(
            f"- `{c['name']}`: **{c['status']}** ({c['severity']}) — {c['evidence']}"
        )
    (run_dir / "leakage_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_common_groups_from_features(
    feature_keys: list[tuple[str, str, str, str]],
    role_by_device: dict[str, str],
) -> list[dict[str, Any]]:
    """Build common-subtraction group diagnostics from (device, state, round, challenge)."""
    groups: dict[tuple[str, str, str], list[str]] = {}
    for device, state, rnd, _cid in feature_keys:
        groups.setdefault((device, state, rnd), []).append(device)
    out = []
    for (device, state, rnd), devices in groups.items():
        roles = {role_by_device.get(device, "unknown")}
        out.append(
            {
                "device": device,
                "state": state,
                "round": rnd,
                "n_devices": 1,
                "n_states": 1,
                "n_rounds": 1,
                "n_roles": len(roles),
                "roles": sorted(roles),
            }
        )
    return out
