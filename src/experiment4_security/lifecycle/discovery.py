"""Discover and adapt Experiment 3 metadata for lifecycle analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED = [
    "device_id",
    "state_id",
    "channel",
    "round_id",
    "challenge_id",
    "video_filename",
    "video_path",
]


def load_experiment3_metadata(path: Path, videos_root: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Metadata missing columns: {missing}")
    # Resolve relative video paths against videos_root / threshold_development root.
    resolved = []
    for _, row in df.iterrows():
        p = Path(str(row["video_path"]))
        if not p.is_absolute():
            cand = videos_root / Path(row["video_filename"]).name
            if cand.exists():
                p = cand
            else:
                p = (path.parent.parent / row["video_path"]).resolve()
        resolved.append(str(p))
    df = df.copy()
    df["video_path"] = resolved
    # Normalize challenge IDs: Exp3 uses C01; lifecycle accepts as-is.
    df["challenge_id"] = df["challenge_id"].fillna("").astype(str)
    df["round_id"] = df["round_id"].fillna("").astype(str)
    return df
