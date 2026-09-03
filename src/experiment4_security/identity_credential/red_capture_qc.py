"""Red capture QC summaries (no formal identity metrics)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from experiment4_security.identity_credential.red_video_processing import red_qc_flags


def summarize_red_qc(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "device_id",
                "state_id",
                "phase",
                "severe_saturation",
                "nearly_absent_signal",
                "qc_status",
            ]
        )
    df = pd.DataFrame(rows)
    if "qc_status" not in df.columns:
        df["qc_status"] = [
            "FAIL"
            if r.get("severe_saturation") or r.get("nearly_absent_signal")
            else "PASS"
            for _, r in df.iterrows()
        ]
    return df


def qc_row_from_recording(rec, *, phase: str) -> dict[str, Any]:
    flags = red_qc_flags(rec)
    flags["phase"] = phase
    flags["qc_status"] = (
        "FAIL" if flags["severe_saturation"] or flags["nearly_absent_signal"] else "PASS"
    )
    return flags
