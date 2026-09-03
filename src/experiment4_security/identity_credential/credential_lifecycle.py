"""Credential lifecycle vocabulary for remount / revoke / re-enroll."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CredentialRecord:
    device_id: str
    credential_state: str  # M0..M7
    database_status: str  # enrolled | revoked | pending_reenroll
    enrolled_challenge_bank: str = "C001-C128"


def mark_revoked(rec: CredentialRecord) -> CredentialRecord:
    return CredentialRecord(
        device_id=rec.device_id,
        credential_state=rec.credential_state,
        database_status="revoked",
        enrolled_challenge_bank=rec.enrolled_challenge_bank,
    )


def mark_enrolled(device_id: str, credential_state: str) -> CredentialRecord:
    return CredentialRecord(
        device_id=device_id,
        credential_state=credential_state,
        database_status="enrolled",
    )


def lifecycle_narrative() -> dict[str, Any]:
    return {
        "device_id": "unchanged across remount",
        "red_identity": "persistent physical identity anchor",
        "credential_state": "changes on mechanical remount (M0..M7)",
        "old_green_database": "revoked after remount",
        "new_green_database": "requires authorized re-enrollment",
        "forbidden_red_roles": [
            "green speckle corrector",
            "green key predictor",
            "mechanical state classifier",
            "green database auto-selector",
            "old green key restorer",
        ],
    }
