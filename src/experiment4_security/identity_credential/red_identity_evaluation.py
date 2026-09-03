"""Red identity Tracks R0–R1.

R0: within-state red_before vs red_after (same device, same M-state)
R1: cross-state same-device identity (gallery vs query across M-states)

Formal numeric AUC/EER/Top-1 are GATED — never auto-emitted by this module.
Pairwise S_R tables are allowed once real red vectors exist.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from experiment4_security.identity_credential.gates import formal_metrics_allowed
from experiment4_security.identity_credential.red_feature_adapter import red_score
from experiment4_security.identity_credential.schemas import TRACK_R0, TRACK_R1


def pair_score_table(
    z_by_key: dict[tuple[str, str, str], np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = list(z_by_key.keys())
    for i, ka in enumerate(keys):
        for kb in keys[i:]:
            if ka[0] != kb[0]:
                continue
            da, sa, pa = ka
            _db, sb, pb = kb
            score = red_score(z_by_key[ka], z_by_key[kb])
            within_state = sa == sb
            before_after = {pa, pb} == {"before", "after"} and within_state
            rows.append(
                {
                    "device_id": da,
                    "state_a": sa,
                    "phase_a": pa,
                    "state_b": sb,
                    "phase_b": pb,
                    "same_device": True,
                    "same_state": within_state,
                    "is_r0_before_after": before_after and ka != kb,
                    "is_r1_cross_state": (not within_state),
                    "score_type": "RED_IDENTITY",
                    "S_R": score,
                }
            )
    return pd.DataFrame(rows)


def run_track_r0_r1(
    z_by_key: dict[tuple[str, str, str], np.ndarray],
    *,
    mode: str,
    data_status: str,
) -> dict[str, Any]:
    pairs = pair_score_table(z_by_key) if z_by_key else pd.DataFrame()
    r0 = pairs[pairs["is_r0_before_after"]] if not pairs.empty else pairs
    r1 = pairs[pairs["is_r1_cross_state"]] if not pairs.empty else pairs

    out: dict[str, Any] = {
        "track_r0": TRACK_R0,
        "track_r1": TRACK_R1,
        "n_r0_pairs": int(len(r0)),
        "n_r1_pairs": int(len(r1)),
        "pairs": pairs,
        # Hard rule: this module never emits AUC/EER/Top-1 numeric metrics
        "formal_metrics_emitted": False,
        "withheld_red_summary_metrics": ["AUC", "EER", "Top-1"],
        "status": "PAIR_SCORES_ONLY" if formal_metrics_allowed(mode, data_status) else "STRUCTURE_ONLY",
        "note": (
            "S_R pair table may be written in formal mode; red AUC/EER/Top-1 remain withheld "
            "from automatic emission. Prepare mode writes neither pair scores nor summaries."
        ),
    }
    return out
