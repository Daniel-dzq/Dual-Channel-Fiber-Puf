"""Continuous green-sequence schedules with drift anchors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_round_schedule(
    *,
    challenge_ids: list[str],
    bank_by_id: dict[str, str],
    sha_by_id: dict[str, str],
    round_id: str,
    order_seed: int,
    settle_time_s: float = 0.5,
    effective_time_s: float = 2.5,
    anchor_ids: list[str] | None = None,
) -> pd.DataFrame:
    """128 unique presentations + 16 anchor repeats (begin + end of C001–C008)."""
    if anchor_ids is None:
        anchor_ids = [f"C{i:03d}" for i in range(1, 9)]
    rng = np.random.default_rng(order_seed)
    unique = list(challenge_ids)
    rng.shuffle(unique)

    sequence: list[tuple[str, bool]] = []
    # 8 begin anchors + 128 unique + 8 end anchors = 144
    for cid in anchor_ids:
        sequence.append((cid, True))
    for cid in unique:
        sequence.append((cid, False))
    for cid in anchor_ids:
        sequence.append((cid, True))

    dt = settle_time_s + effective_time_s
    rows = []
    t = 0.0
    for idx, (cid, is_anchor) in enumerate(sequence, start=1):
        onset = t
        settle_end = onset + settle_time_s
        rows.append(
            {
                "presentation_index": idx,
                "challenge_id": cid,
                "bank_id": bank_by_id[cid],
                "is_anchor_repeat": bool(is_anchor),
                "planned_onset_time_s": float(onset),
                "actual_onset_time_s": float("nan"),
                "settle_end_time_s": float(settle_end),
                "valid_start_time_s": float(settle_end),
                "valid_end_time_s": float(onset + dt),
                "round_id": round_id,
                "order_seed": int(order_seed),
                "pattern_sha256": sha_by_id[cid],
                "frames_recorded_in_valid_window": 0,
            }
        )
        t = onset + dt
    return pd.DataFrame(rows)


def schedule_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()


def write_schedule(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    h = schedule_hash(df)
    path.with_suffix(".hash.json").write_text(
        json.dumps({"schedule_hash": h, "n_rows": len(df)}, indent=2),
        encoding="utf-8",
    )
    return h
