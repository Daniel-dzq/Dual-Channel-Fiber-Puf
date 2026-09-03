"""Build remount episodes from red_before/after + green A/B (formal when ready)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from experiment4_security.identity_credential.gates import formal_metrics_allowed
from experiment4_security.identity_credential.joint_decision import classify_episode


def build_episodes_from_manifests(
    *,
    red_rows: pd.DataFrame,
    green_ab_pairs: pd.DataFrame,
    mode: str,
    data_status: str,
) -> dict[str, Any]:
    """Create episode skeletons. Pass/fail scores filled only in formal mode."""
    allowed = formal_metrics_allowed(mode, data_status)
    episodes = []
    if red_rows is None or red_rows.empty:
        return {
            "n_episodes": 0,
            "episodes": [],
            "formal_conclusions_emitted": False,
            "status": "NO_RED_ROWS",
        }
    # One episode per device×state requiring before+after
    keys = red_rows.groupby(["device_id", "state_id"]).size().reset_index()
    for _, r in keys.iterrows():
        d, s = r["device_id"], r["state_id"]
        phases = set(
            red_rows[(red_rows["device_id"] == d) & (red_rows["state_id"] == s)]["phase"].tolist()
        )
        n_pairs = 0
        if green_ab_pairs is not None and not green_ab_pairs.empty:
            n_pairs = int(
                ((green_ab_pairs["device_id"] == d) & (green_ab_pairs["state_id"] == s)).sum()
            )
        ep = {
            "device_id": d,
            "credential_state": s,
            "has_red_before": "before" in phases,
            "has_red_after": "after" in phases,
            "n_green_ab_pairs": n_pairs,
            "decision_label": None if not allowed else classify_episode(
                red_pass=None, old_green_pass=None, new_green_pass=None
            ),
        }
        episodes.append(ep)
    return {
        "n_episodes": len(episodes),
        "episodes": episodes,
        "formal_conclusions_emitted": False,
        "status": "STRUCTURE_ONLY" if not allowed else "EPISODES_PENDING_SCORES",
    }
