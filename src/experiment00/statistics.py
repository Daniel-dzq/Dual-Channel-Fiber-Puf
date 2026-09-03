"""Red reference analysis (guardrail / stability, not green correction)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiment00.video_processing import ProcessedVideo
from puf_common.ncc import zero_mean_ncc


def red_pair_metrics(
    red_videos: list[ProcessedVideo],
    mask: np.ndarray | None,
) -> pd.DataFrame:
    by_key: dict[tuple[int, str], dict[str, ProcessedVideo]] = {}
    for r in red_videos:
        by_key.setdefault((r.length_cm, r.fiber_id), {})[r.red_phase or ""] = r
    rows = []
    for (L, f), phases in sorted(by_key.items()):
        before = phases.get("before")
        after = phases.get("after")
        if before is None or after is None:
            rows.append(
                {
                    "length_cm": L,
                    "fiber_id": f,
                    "S_R_intra": np.nan,
                    "qc": "missing_before_or_after",
                }
            )
            continue
        s = float(zero_mean_ncc(before.representative_raw, after.representative_raw, mask=mask))
        rows.append(
            {
                "length_cm": L,
                "fiber_id": f,
                "S_R_intra": s,
                "mean_before": float(np.mean(before.representative_raw)),
                "mean_after": float(np.mean(after.representative_raw)),
                "sat_before": before.saturation_fraction,
                "sat_after": after.saturation_fraction,
                "qc": "ok" if s == s else "nan",
            }
        )
    return pd.DataFrame(rows)
