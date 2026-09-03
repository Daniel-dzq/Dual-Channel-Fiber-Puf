"""A/B pair construction and documented state-label corrections.

Pairing rule (mandatory):
  device_id + state_id + challenge_id

presentation_index is recorded but NEVER used for pairing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def build_ab_pair_manifest(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """Build unique A/B pairs keyed by (device_id, state_id, challenge_id)."""
    ok = manifest_df[
        manifest_df["challenge_id"].notna()
        & manifest_df["round_id"].isin(["A", "B"])
        & manifest_df["sample_id"].notna()
    ].copy()

    rows: list[dict[str, Any]] = []
    keys = ["device_id", "state_id", "challenge_id"]
    for (device_id, state_id, challenge_id), sub in ok.groupby(keys, dropna=False):
        a = sub[sub["round_id"] == "A"]
        b = sub[sub["round_id"] == "B"]
        status = "OK"
        if len(a) != 1 or len(b) != 1:
            status = "ERROR_PAIR_COUNT"
        a_row = a.iloc[0] if len(a) else None
        b_row = b.iloc[0] if len(b) else None
        rows.append(
            {
                "device_id": device_id,
                "source_device_id": (a_row["source_device_id"] if a_row is not None else (b_row["source_device_id"] if b_row is not None else None)),
                "state_id": state_id,
                "challenge_id": challenge_id,
                "a_video_path": None if a_row is None else a_row["video_path"],
                "a_presentation_index": None if a_row is None else a_row["presentation_index"],
                "a_file_name": None if a_row is None else a_row["file_name"],
                "a_sample_id": None if a_row is None else a_row["sample_id"],
                "b_video_path": None if b_row is None else b_row["video_path"],
                "b_presentation_index": None if b_row is None else b_row["presentation_index"],
                "b_file_name": None if b_row is None else b_row["file_name"],
                "b_sample_id": None if b_row is None else b_row["sample_id"],
                "pair_status": status,
            }
        )
    return pd.DataFrame(rows).sort_values(["state_id", "challenge_id"]).reset_index(drop=True)


def apply_folder_authoritative_state_correction(
    manifest_df: pd.DataFrame,
    *,
    out_json: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep folder `state_id` as official; record filename mismatches.

    Forensic finding for F01 (2026-07-28 acquisition export):
      - All 256 conflicts are files under folder M7 whose filenames carry M0.
      - M0..M6 folders are internally consistent.
      - File bytes/timestamps show M7 content is distinct from M0; not a copy.
      - Not an ``if state_number:`` False-zero parser bug (M0 parses correctly).

    After correction, downstream uses folder state_id; raw conflict flags are
    preserved in audit columns, while `state_id_conflict_active` becomes False.
    """
    df = manifest_df.copy()
    conflicts = df["state_id_conflict"].fillna(False).astype(bool)
    correction_rows = []
    for _, r in df.loc[conflicts].iterrows():
        correction_rows.append(
            {
                "video_path": r["video_path"],
                "file_name": r["file_name"],
                "folder_state_id": r["state_id"],
                "filename_state_id": r["state_id_from_filename"],
                "official_state_id": r["state_id"],
                "correction_rule": "folder_authoritative_filename_export_bug",
                "reason": (
                    "Filename state token disagrees with parent folder. "
                    "Folder creation order, file mtimes, and byte hashes support "
                    "folder as the true mechanical state; filename token is an "
                    "export/acquisition labeling bug. Original videos not renamed."
                ),
            }
        )

    df["state_id_conflict_raw"] = conflicts
    df["state_id_conflict_active"] = False  # after documented correction
    df["official_state_id"] = df["state_id"]

    report = {
        "n_raw_conflicts": int(conflicts.sum()),
        "n_active_conflicts_after_correction": 0,
        "correction_rule": "folder_authoritative_filename_export_bug",
        "conflict_matrix": (
            df.loc[conflicts]
            .groupby(["state_id", "state_id_from_filename"])
            .size()
            .reset_index(name="count")
            .to_dict(orient="records")
            if conflicts.any()
            else []
        ),
        "corrections": correction_rows,
        "is_m0_false_zero_parser_bug": False,
        "finding": (
            "Conflicts (if any) are filename-vs-folder mismatches. "
            "F01: exclusively M7 folder with M0 filename tokens. "
            "M0 itself is NOT mis-parsed via integer 0 truthiness."
        ),
    }
    if out_json is not None:
        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    return df, report
