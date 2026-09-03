"""Credential-renewal lifecycle events S0→S1 and S1→S2."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


SYSTEM_STATES = (
    "ACTIVE",
    "STATE_CHANGE_DETECTED",
    "IDENTITY_CONFIRMED",
    "REENROLLMENT_PENDING",
    "NEW_CREDENTIAL_PENDING",
    "NEW_CREDENTIAL_ACTIVE",
    "REPLACEMENT_BLOCKED",
    "LOCKED",
)


def build_lifecycle_events(
    *,
    devices: list[str],
    red_scores: pd.DataFrame,
    green_scores: pd.DataFrame,
    tau_r: float,
    tau_g: float,
    k_of_8: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    transitions = [("S0", "S1"), ("S1", "S2")]
    for device in devices:
        for prev, curr in transitions:
            # Red: enroll S0 gallery vs query current state (primary protocol uses S0 gallery)
            r = red_scores[
                (red_scores["group"] == "same_device_diff_state")
                & (red_scores["device_id_a"] == device)
                & (
                    ((red_scores["state_id_a"] == "S0") & (red_scores["state_id_b"] == curr))
                    | ((red_scores["state_id_b"] == "S0") & (red_scores["state_id_a"] == curr))
                )
            ]
            s_r = float(r["score"].iloc[0]) if len(r) else float("nan")
            red_pass = bool(np.isfinite(s_r) and s_r >= tau_r)

            # Previous green credential: prev-state template vs curr-state same challenge
            g_cross = green_scores[
                (green_scores["group"] == "same_device_diff_state_same_challenge")
                & (green_scores["device_id_a"] == device)
                & (
                    (
                        (green_scores["state_id_a"] == prev)
                        & (green_scores["state_id_b"] == curr)
                    )
                    | (
                        (green_scores["state_id_a"] == curr)
                        & (green_scores["state_id_b"] == prev)
                    )
                )
            ]
            prev_green_fail = True
            if len(g_cross):
                # Credential revoked if median cross-state score fails threshold
                prev_green_fail = bool(float(g_cross["score"].median()) < tau_g)

            # Current-state A/B agreement
            g_ab = green_scores[
                (green_scores["group"] == "same_device_same_state_same_challenge")
                & (green_scores["device_id_a"] == device)
                & (green_scores["state_id_a"] == curr)
            ]
            n_pass = int((g_ab["score"] >= tau_g).sum()) if len(g_ab) else 0
            curr_cred_pass = n_pass >= k_of_8

            accepted = red_pass and prev_green_fail and curr_cred_pass
            if not red_pass:
                system = "LOCKED"
            elif prev_green_fail and curr_cred_pass:
                system = "NEW_CREDENTIAL_ACTIVE"
            elif prev_green_fail and not curr_cred_pass:
                system = "LOCKED"
            elif red_pass and not prev_green_fail:
                system = "ACTIVE"
            else:
                system = "STATE_CHANGE_DETECTED"

            rows.append(
                {
                    "device_id": device,
                    "from_state": prev,
                    "to_state": curr,
                    "red_score": s_r,
                    "red_pass": red_pass,
                    "previous_green_fail": prev_green_fail,
                    "n_green_pass": n_pass,
                    "k_of_8": k_of_8,
                    "current_credential_pass": curr_cred_pass,
                    "authenticated_reenrollment_success": accepted,
                    "system_state": system,
                }
            )
    return pd.DataFrame(rows)


def lifecycle_metrics(events: pd.DataFrame, frozen_devices: list[str]) -> dict[str, Any]:
    def _rate(mask: pd.Series, col: str) -> float:
        sub = events.loc[mask, col]
        return float(sub.mean()) if len(sub) else float("nan")

    all_m = np.ones(len(events), dtype=bool)
    froz = events["device_id"].isin(frozen_devices)
    return {
        "n_events": int(len(events)),
        "identity_continuity": _rate(all_m, "red_pass"),
        "revocation_rate": _rate(all_m, "previous_green_fail"),
        "reenrollment_success": _rate(all_m, "current_credential_pass"),
        "authenticated_reenrollment_success": _rate(
            all_m, "authenticated_reenrollment_success"
        ),
        "frozen_authenticated_reenrollment_success": _rate(
            froz, "authenticated_reenrollment_success"
        ),
        "replacement_rejection_note": (
            "Replacement blocking is demonstrated in the Demo "
            "(cross-device red failure); not counted as remount events here."
        ),
    }
