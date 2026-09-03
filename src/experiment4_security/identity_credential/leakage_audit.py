"""Leakage audit scaffolding for red→green side-information (formal when ready)."""

from __future__ import annotations

from typing import Any

from experiment4_security.identity_credential.gates import formal_metrics_allowed


def leakage_audit_structure(*, mode: str, data_status: str) -> dict[str, Any]:
    allowed = formal_metrics_allowed(mode, data_status)
    return {
        "audit_name": "red_to_green_side_information_leakage",
        "status": "PENDING_DATA" if not allowed else "READY_TO_RUN",
        "formal_results_emitted": False,
        "allowed_claim_when_pending": (
            "red is used as an identity anchor, while red-to-green credential "
            "leakage remains to be tested with synchronized data."
        ),
        "forbidden_claim": "red cannot leak green state",
    }
