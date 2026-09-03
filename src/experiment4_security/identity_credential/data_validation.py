"""Dataset validation against frozen expected inventory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from experiment4_security.identity_credential.manifest import (
    ab_pair_table,
    build_expected_manifest_df,
    mark_expected_present,
    scan_videos,
)
from experiment4_security.identity_credential.schemas import (
    DEVICES,
    EXPECTED_INVENTORY,
    N_AB_PAIRS_EXPECTED,
    N_GREEN_VIDEOS_EXPECTED,
    N_RED_VIDEOS_EXPECTED,
    N_TOTAL_VIDEOS_EXPECTED,
    STATES,
    STATUS_DATA_BLOCKED,
    STATUS_DATA_PARTIAL,
    STATUS_DATA_READY,
    STATUS_DATA_WAITING,
    challenge_ids,
)


def validate_dataset(videos_root: Path) -> dict[str, Any]:
    root = Path(videos_root)
    found = scan_videos(root)
    expected = build_expected_manifest_df(root)
    expected = mark_expected_present(expected, found)

    n_green = int(((found.get("channel") == "green") & (found.get("parse_ok") == True)).sum()) if not found.empty else 0
    n_red = int(((found.get("channel") == "red") & (found.get("parse_ok") == True)).sum()) if not found.empty else 0
    n_unknown = int((found.get("channel") == "unknown").sum()) if not found.empty else 0
    n_conflicts = int(found["state_conflict"].sum()) if not found.empty and "state_conflict" in found.columns else 0
    n_dup = 0
    if not found.empty and "sample_id" in found.columns:
        sid = found.loc[found["parse_ok"] == True, "sample_id"]
        n_dup = int(sid.duplicated().sum())

    pairs = ab_pair_table(found)
    n_pairs = int(len(pairs))

    missing = expected.loc[~expected["present"]]
    n_missing = int(len(missing))
    n_present = int(expected["present"].sum())

    devices_present = sorted(found["device_id"].dropna().unique().tolist()) if not found.empty else []
    states_present = sorted(found["state_id"].dropna().unique().tolist()) if not found.empty else []

    if n_present == 0:
        status = STATUS_DATA_WAITING
    elif n_conflicts > 0 or n_dup > 0 or n_unknown > 0:
        status = STATUS_DATA_BLOCKED
    elif (
        n_green == N_GREEN_VIDEOS_EXPECTED
        and n_red == N_RED_VIDEOS_EXPECTED
        and n_pairs == N_AB_PAIRS_EXPECTED
        and n_missing == 0
        and set(devices_present) >= set(DEVICES)
    ):
        status = STATUS_DATA_READY
    else:
        status = STATUS_DATA_PARTIAL

    audit = {
        "data_status": status,
        "videos_root": str(root),
        "n_videos_found": int(len(found)),
        "n_green_found": n_green,
        "n_red_found": n_red,
        "n_unknown": n_unknown,
        "n_expected": int(len(expected)),
        "n_present": n_present,
        "n_missing": n_missing,
        "n_ab_pairs": n_pairs,
        "n_state_conflicts": n_conflicts,
        "n_duplicate_samples": n_dup,
        "devices_present": devices_present,
        "states_present": states_present,
        "expected_inventory": dict(EXPECTED_INVENTORY),
        "counts_match_expected": {
            "n_green_videos": n_green == N_GREEN_VIDEOS_EXPECTED,
            "n_red_videos": n_red == N_RED_VIDEOS_EXPECTED,
            "n_total_videos": (n_green + n_red) == N_TOTAL_VIDEOS_EXPECTED,
            "n_ab_pairs": n_pairs == N_AB_PAIRS_EXPECTED,
            "n_state_conflicts": n_conflicts == 0,
            "n_duplicate_samples": n_dup == 0,
        },
        "missing_examples": missing.head(20)[["channel", "device_id", "state_id", "relpath"]].to_dict(
            orient="records"
        )
        if n_missing
        else [],
        "challenge_ids": challenge_ids(),
        "note": (
            "DATA_READY requires complete synchronized F01–F10 red+green capture. "
            "Old lifecycle red must not be copied here."
        ),
    }
    return {"audit": audit, "found": found, "expected": expected, "ab_pairs": pairs}


def write_validation_artifacts(result: dict[str, Any], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result["expected"].to_csv(out_dir / "expected_dataset_manifest.csv", index=False)
    if result["found"] is not None and not result["found"].empty:
        result["found"].to_csv(out_dir / "found_videos_manifest.csv", index=False)
    if result["ab_pairs"] is not None and not result["ab_pairs"].empty:
        result["ab_pairs"].to_csv(out_dir / "ab_pair_manifest.csv", index=False)
    import json

    (out_dir / "data_validation_audit.json").write_text(
        json.dumps(result["audit"], indent=2, sort_keys=True) + "\n"
    )
