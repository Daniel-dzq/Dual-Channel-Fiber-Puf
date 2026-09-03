"""Metadata loading, filename parsing, and inventory for Experiment 3."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = [
    "device_id",
    "state_id",
    "channel",
    "round_id",
    "challenge_id",
    "video_filename",
    "video_path",
]

FILENAME_RE = re.compile(
    r"^(?P<device>F\d{2})_(?P<state>S\d+)_(?:(?P<round>[AB])_(?P<challenge>C\d{2})|(?P<red>R))"
    r"\.(?P<ext>mp4|avi|mov)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedFilename:
    device_id: str
    state_id: str
    channel: str
    round_id: str
    challenge_id: str
    video_filename: str


def parse_filename(name: str) -> ParsedFilename:
    """Parse Experiment 3 video filenames.

    Examples:
      F01_S0_R.mp4
      F01_S0_A_C01.mp4
      F12_S2_B_C08.mp4
    """
    fname = Path(name).name
    m = FILENAME_RE.match(fname)
    if m is None:
        raise ValueError(f"Unrecognized Experiment 3 filename: {fname}")
    device = m.group("device").upper()
    state = m.group("state").upper()
    if m.group("red"):
        return ParsedFilename(
            device_id=device,
            state_id=state,
            channel="red",
            round_id="",
            challenge_id="",
            video_filename=fname,
        )
    return ParsedFilename(
        device_id=device,
        state_id=state,
        channel="green",
        round_id=m.group("round").upper(),
        challenge_id=m.group("challenge").upper(),
        video_filename=fname,
    )


def load_metadata_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Metadata missing required columns: {missing}")
    df = df.copy()
    for col in ("device_id", "state_id", "channel", "round_id", "challenge_id", "video_filename"):
        df[col] = df[col].fillna("").astype(str).str.strip()
    df["device_id"] = df["device_id"].str.upper()
    df["state_id"] = df["state_id"].str.upper()
    df["channel"] = df["channel"].str.lower()
    df["round_id"] = df["round_id"].str.upper().replace({"NAN": "", "NONE": ""})
    df["challenge_id"] = df["challenge_id"].str.upper().replace({"NAN": "", "NONE": ""})
    return df


def discover_from_directory(videos_root: Path) -> pd.DataFrame:
    """Auto-build metadata rows from filenames under videos_root."""
    rows: list[dict[str, Any]] = []
    if not videos_root.exists():
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    for path in sorted(videos_root.rglob("*")):
        if path.suffix.lower() not in {".mp4", ".avi", ".mov"}:
            continue
        try:
            parsed = parse_filename(path.name)
        except ValueError:
            continue
        rows.append(
            {
                "device_id": parsed.device_id,
                "state_id": parsed.state_id,
                "channel": parsed.channel,
                "round_id": parsed.round_id,
                "challenge_id": parsed.challenge_id,
                "video_filename": parsed.video_filename,
                "video_path": str(path.resolve()),
            }
        )
    return pd.DataFrame(rows)


def resolve_video_paths(df: pd.DataFrame, *, root: Path, videos_root: Path) -> pd.DataFrame:
    out = df.copy()
    resolved: list[str] = []
    for _, row in out.iterrows():
        raw = str(row.get("video_path", "") or "").strip()
        if raw:
            p = Path(raw)
            if not p.is_absolute():
                p = (root / p).resolve()
        else:
            p = (videos_root / str(row["video_filename"])).resolve()
        resolved.append(str(p))
    out["video_path"] = resolved
    return out


def inventory(df: pd.DataFrame) -> dict[str, Any]:
    devices = sorted(df["device_id"].unique().tolist())
    states = sorted(df["state_id"].unique().tolist())
    rounds = sorted([r for r in df["round_id"].unique().tolist() if r])
    challenges = sorted([c for c in df["challenge_id"].unique().tolist() if c])
    n_red = int(((df["channel"] == "red")).sum())
    n_green = int(((df["channel"] == "green")).sum())
    return {
        "n_rows": int(len(df)),
        "devices": devices,
        "states": states,
        "rounds": rounds,
        "challenges": challenges,
        "n_devices": len(devices),
        "n_states": len(states),
        "n_red": n_red,
        "n_green": n_green,
    }


def expected_key(device_id: str, state_id: str, channel: str, round_id: str, challenge_id: str) -> str:
    if channel == "red":
        return f"{device_id}_{state_id}_R"
    return f"{device_id}_{state_id}_{round_id}_{challenge_id}"


def build_expected_inventory(
    devices: list[str],
    states: list[str],
    rounds: list[str],
    challenges: list[str],
) -> list[str]:
    keys: list[str] = []
    for d in devices:
        for s in states:
            keys.append(expected_key(d, s, "red", "", ""))
            for r in rounds:
                for c in challenges:
                    keys.append(expected_key(d, s, "green", r, c))
    return keys


def find_missing_and_duplicates(
    df: pd.DataFrame,
    *,
    devices: list[str] | None = None,
    states: list[str] | None = None,
    rounds: list[str] | None = None,
    challenges: list[str] | None = None,
) -> dict[str, list[str]]:
    """Compare available rows to the cartesian product of detected (or provided) axes."""
    inv = inventory(df)
    devices = devices if devices is not None else inv["devices"]
    states = states if states is not None else inv["states"]
    rounds = rounds if rounds is not None else (inv["rounds"] or ["A", "B"])
    challenges = challenges if challenges is not None else (inv["challenges"] or [f"C{i:02d}" for i in range(1, 9)])

    present_keys: list[str] = []
    for _, row in df.iterrows():
        present_keys.append(
            expected_key(
                row["device_id"],
                row["state_id"],
                row["channel"],
                row["round_id"],
                row["challenge_id"],
            )
        )
    expected = build_expected_inventory(devices, states, rounds, challenges)
    present_set = set(present_keys)
    expected_set = set(expected)
    missing = sorted(expected_set - present_set)
    extras = sorted(present_set - expected_set)
    counts: dict[str, int] = {}
    for k in present_keys:
        counts[k] = counts.get(k, 0) + 1
    duplicates = sorted([k for k, n in counts.items() if n > 1])
    return {"missing": missing, "duplicates": duplicates, "extras": extras}
