"""Challenge manifest helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def bank_id_for_index(one_based_index: int) -> str:
    bank = (one_based_index - 1) // 8 + 1
    return f"B{bank:02d}"


def write_manifests(out_root: Path, rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(out_root / "challenge_manifest.csv", index=False)
    payload = {
        "version": rows[0]["generation_version"] if rows else "",
        "n_challenges": len(rows),
        "challenges": rows,
    }
    (out_root / "challenge_manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def load_manifest(out_root: Path) -> pd.DataFrame:
    return pd.read_csv(out_root / "challenge_manifest.csv")
