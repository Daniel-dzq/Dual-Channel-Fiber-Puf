"""Scan identity_credential video tree into a manifest DataFrame."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from experiment4_security.identity_credential.dataset_layout import iter_expected_paths
from experiment4_security.identity_credential.schemas import (
    GREEN_FILENAME_RE,
    RED_FILENAME_RE,
    SAMPLE_ID_GREEN,
    SAMPLE_ID_RED,
    challenge_ids,
)


_GREEN_RE = re.compile(GREEN_FILENAME_RE)
_RED_RE = re.compile(RED_FILENAME_RE)


def parse_green_filename(name: str) -> dict[str, Any] | None:
    m = _GREEN_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    return {
        "channel": "green",
        "round_id": d["round"],
        "presentation_index": int(d["index"]),
        "challenge_id": f"C{int(d['cid']):03d}",
        "state_id_from_name": d["state"],
        "device_id_from_name": d["device"],
        "phase": "",
    }


def parse_red_filename(name: str) -> dict[str, Any] | None:
    m = _RED_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    return {
        "channel": "red",
        "round_id": "",
        "presentation_index": None,
        "challenge_id": "",
        "state_id_from_name": d["state"],
        "device_id_from_name": d["device"],
        "phase": d["phase"],
    }


def scan_videos(videos_root: Path) -> pd.DataFrame:
    """Walk videos_root; folder device/state are authoritative."""
    root = Path(videos_root)
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return pd.DataFrame(rows)

    for path in sorted(root.rglob("*.mp4")):
        rel = path.relative_to(root)
        parts = rel.parts
        device_folder = parts[0] if len(parts) >= 1 else ""
        state_folder = parts[1] if len(parts) >= 2 else ""
        parsed = parse_green_filename(path.name) or parse_red_filename(path.name)
        if parsed is None:
            rows.append(
                {
                    "abspath": str(path),
                    "relpath": str(rel),
                    "filename": path.name,
                    "channel": "unknown",
                    "device_id": device_folder,
                    "state_id": state_folder,
                    "parse_ok": False,
                    "state_conflict": False,
                    "sample_id": "",
                }
            )
            continue
        device_id = device_folder or parsed["device_id_from_name"]
        state_id = state_folder or parsed["state_id_from_name"]
        conflict = (
            (parsed["device_id_from_name"] != device_id)
            or (parsed["state_id_from_name"] != state_id)
        )
        if parsed["channel"] == "green":
            sample_id = SAMPLE_ID_GREEN.format(
                device=device_id,
                state=state_id,
                round=parsed["round_id"],
                cid=int(parsed["challenge_id"][1:]),
            )
        else:
            sample_id = SAMPLE_ID_RED.format(
                device=device_id, state=state_id, phase=parsed["phase"]
            )
        rows.append(
            {
                "abspath": str(path),
                "relpath": str(rel),
                "filename": path.name,
                "channel": parsed["channel"],
                "device_id": device_id,
                "state_id": state_id,
                "round_id": parsed["round_id"],
                "phase": parsed["phase"],
                "challenge_id": parsed["challenge_id"],
                "presentation_index": parsed["presentation_index"],
                "device_id_from_name": parsed["device_id_from_name"],
                "state_id_from_name": parsed["state_id_from_name"],
                "parse_ok": True,
                "state_conflict": bool(conflict),
                "sample_id": sample_id,
            }
        )
    return pd.DataFrame(rows)


def build_expected_manifest_df(videos_root: Path) -> pd.DataFrame:
    rows = iter_expected_paths(Path(videos_root))
    df = pd.DataFrame(rows)
    df["present"] = False
    return df


def mark_expected_present(expected: pd.DataFrame, found: pd.DataFrame) -> pd.DataFrame:
    out = expected.copy()
    if found.empty:
        out["present"] = False
        return out
    present = set(found["relpath"].astype(str))
    # Also accept green files whose presentation index differs but challenge/state/device/round match
    alt: set[str] = set()
    for _, r in found.iterrows():
        if r.get("channel") == "green" and r.get("parse_ok"):
            # canonicalize path key device/state/green/round/challenge
            alt.add(
                f"{r['device_id']}/{r['state_id']}/green/{r['round_id']}/{r['challenge_id']}"
            )
        elif r.get("channel") == "red" and r.get("parse_ok"):
            alt.add(f"{r['device_id']}/{r['state_id']}/red/{r['phase']}")
    flags = []
    for _, e in out.iterrows():
        if e["relpath"] in present:
            flags.append(True)
            continue
        if e["channel"] == "green":
            key = f"{e['device_id']}/{e['state_id']}/green/{e['round_id']}/{e['challenge_id']}"
            flags.append(key in alt)
        else:
            key = f"{e['device_id']}/{e['state_id']}/red/{e['phase']}"
            flags.append(key in alt)
    out["present"] = flags
    return out


def ab_pair_table(found: pd.DataFrame) -> pd.DataFrame:
    if found.empty:
        return pd.DataFrame()
    g = found[(found["channel"] == "green") & (found["parse_ok"])].copy()
    if g.empty:
        return pd.DataFrame()
    a = g[g["round_id"] == "A"][["device_id", "state_id", "challenge_id", "sample_id", "abspath"]].rename(
        columns={"sample_id": "sample_id_a", "abspath": "abspath_a"}
    )
    b = g[g["round_id"] == "B"][["device_id", "state_id", "challenge_id", "sample_id", "abspath"]].rename(
        columns={"sample_id": "sample_id_b", "abspath": "abspath_b"}
    )
    return a.merge(b, on=["device_id", "state_id", "challenge_id"], how="inner")
