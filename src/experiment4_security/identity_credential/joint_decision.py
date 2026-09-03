"""Joint identity–credential decision table (Track J).

Formal conclusions are GATED until DATA_READY + formal run.
"""

from __future__ import annotations

from typing import Any

from experiment4_security.identity_credential.gates import formal_metrics_allowed
from experiment4_security.identity_credential.schemas import TRACK_J

# State machine (protocol) — labels only; do not emit as conclusions in prepare/
DECISION_TABLE = (
    ("red_fail", "green_any", "INVALID_IDENTITY"),
    ("red_pass", "old_green_pass", "AUTHENTICATED"),
    ("red_pass", "old_green_fail", "IDENTITY_RETAINED_OLD_CREDENTIAL_REVOKED"),
    ("red_pass", "new_green_pass", "NEW_CREDENTIAL_ENROLLED_AND_AUTHENTICATED"),
)


def classify_episode(
    *,
    red_pass: bool | None,
    old_green_pass: bool | None,
    new_green_pass: bool | None,
) -> str:
    if red_pass is None:
        return "PENDING_RED"
    if not red_pass:
        return "INVALID_IDENTITY"
    if old_green_pass is True:
        return "AUTHENTICATED"
    if new_green_pass is True:
        return "NEW_CREDENTIAL_ENROLLED_AND_AUTHENTICATED"
    if old_green_pass is False:
        return "IDENTITY_RETAINED_OLD_CREDENTIAL_REVOKED"
    return "INCOMPLETE_EPISODE"


def run_track_j_structure(*, mode: str, data_status: str) -> dict[str, Any]:
    allowed = formal_metrics_allowed(mode, data_status)
    return {
        "track": TRACK_J,
        "decision_table": [
            {"red": a, "green": b, "label": c} for a, b, c in DECISION_TABLE
        ],
        "formal_conclusions_emitted": False,
        "status": "STRUCTURE_ONLY" if not allowed else "READY_FOR_EPISODES",
        "note": (
            "Joint identity–credential conclusions withheld until synchronized "
            "DATA_READY formal run."
            if not allowed
            else "Episode generation may proceed."
        ),
    }
