"""Attack evaluation helpers for Track E (gated)."""

from __future__ import annotations

from typing import Any

from experiment4_security.identity_credential.gates import GateError, formal_metrics_allowed


def evaluate_red_conditioning_gain(
    *,
    challenge_only_score: float | None,
    challenge_plus_red_score: float | None,
    mode: str,
    data_status: str,
) -> dict[str, Any]:
    if not formal_metrics_allowed(mode, data_status):
        return {
            "gain": None,
            "status": "WITHHELD",
            "note": "Gain withheld until formal DATA_READY run",
        }
    if challenge_only_score is None or challenge_plus_red_score is None:
        raise GateError("Missing scores for conditioning gain")
    return {
        "gain": float(challenge_plus_red_score - challenge_only_score),
        "status": "COMPUTED",
    }
