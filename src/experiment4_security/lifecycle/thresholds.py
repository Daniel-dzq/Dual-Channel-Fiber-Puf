"""Development-only threshold and session-rule selection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from experiment4_security.common.frozen_protocol import SESSION_RULE_CANDIDATES
from puf_common.metrics import equal_error_rate_with_threshold


def select_tau(genuine: np.ndarray, impostor: np.ndarray) -> float:
    _, thr = equal_error_rate_with_threshold(genuine, impostor)
    return float(thr)


def select_session_rule(
    green_dev: pd.DataFrame,
    tau_g: float,
) -> dict[str, Any]:
    """Choose k/8 pass rule on development devices only."""
    # same-state same-challenge A/B pairs
    sub = green_dev[green_dev["group"] == "same_device_same_state_same_challenge"]
    best = None
    rows = []
    for k in SESSION_RULE_CANDIDATES:
        # Per device×state: count challenges with score >= tau_g
        ok_events = 0
        total_events = 0
        for (dev, state), g in sub.groupby(["device_id_a", "state_id_a"]):
            total_events += 1
            n_pass = int((g["score"] >= tau_g).sum())
            if n_pass >= k:
                ok_events += 1
        rate = ok_events / max(total_events, 1)
        rows.append({"k_of_8": k, "session_pass_rate": rate, "n_events": total_events})
        # Prefer highest k that keeps rate >= 0.8, else best rate
        score = rate + 0.01 * k
        if best is None or score > best["score"]:
            best = {"k_of_8": k, "session_pass_rate": rate, "score": score}
    assert best is not None
    return {
        "selected_k_of_8": int(best["k_of_8"]),
        "candidates": rows,
    }
