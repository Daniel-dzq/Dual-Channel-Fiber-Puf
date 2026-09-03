"""Track E — red-conditioned attack scaffolding (gated; no conclusions today)."""

from __future__ import annotations

from typing import Any

from experiment4_security.identity_credential.gates import GateError, formal_metrics_allowed
from experiment4_security.identity_credential.schemas import TRACK_E


ATTACK_VARIANTS = (
    "challenge_only_clone",
    "red_only_to_green",
    "challenge_plus_red_clone",
)


def run_track_e_structure(*, mode: str, data_status: str) -> dict[str, Any]:
    allowed = formal_metrics_allowed(mode, data_status)
    return {
        "track": TRACK_E,
        "variants": list(ATTACK_VARIANTS),
        "metric_planned": "gain = perf(challenge+red) - perf(challenge_only)",
        "formal_conclusions_emitted": False,
        "status": "NOT_AVAILABLE" if not allowed else "READY_BUT_REQUIRES_EXPLICIT_EMIT",
        "note": (
            "Red-conditioned attack conclusions withheld. "
            "Do not fabricate red features or splice lifecycle red."
        ),
    }


def refuse_fake_red_conditioned_run() -> None:
    raise GateError(
        "Track E formal evaluation requires synchronized red+green DATA_READY. "
        "Synthetic/random red arrays are forbidden for formal results."
    )
