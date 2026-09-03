"""Filename parsing and factorial inventory validation."""

from __future__ import annotations

import itertools
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

FILENAME_RE = re.compile(
    r"^(?P<repeat>M[0-2])_(?P<coupling>axial|lateral)_"
    r"(?P<color>green|red)_F(?P<fiber>[1-5])\.mp4$",
    re.IGNORECASE,
)

COLOR_TO_WAVELENGTH = {"green": 532, "red": 650}
REPEAT_TO_ID = {"M0": 0, "M1": 1, "M2": 2}
FIBERS = [1, 2, 3, 4, 5]
COLORS = ["green", "red"]
COUPLINGS = ["axial", "lateral"]
REPEATS = ["M0", "M1", "M2"]


@dataclass(frozen=True)
class ParsedVideo:
    filename: str
    source_path: str
    fiber_id: int
    raw_repeat_label: str
    repeat_id: int
    repeat_role: str
    color_label: str
    wavelength_nm: int
    excitation_geometry: str


def parse_filename(name: str) -> ParsedVideo | None:
    m = FILENAME_RE.match(Path(name).name)
    if not m:
        return None
    color = m.group("color").lower()
    coupling = m.group("coupling").lower()
    repeat = m.group("repeat").upper()
    fiber = int(m.group("fiber"))
    return ParsedVideo(
        filename=Path(name).name,
        source_path="",
        fiber_id=fiber,
        raw_repeat_label=repeat,
        repeat_id=REPEAT_TO_ID[repeat],
        repeat_role="technical_acquisition_repeat",
        color_label=color,
        wavelength_nm=COLOR_TO_WAVELENGTH[color],
        excitation_geometry=coupling,
    )


def expected_keys() -> list[tuple[int, str, str, str]]:
    """(fiber_id, color, coupling, repeat_label)."""
    return list(itertools.product(FIBERS, COLORS, COUPLINGS, REPEATS))


def scan_videos(videos_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    videos_root = Path(videos_root)
    rows: list[dict[str, Any]] = []
    unparsed: list[str] = []
    if not videos_root.is_dir():
        return pd.DataFrame(), {
            "ok": False,
            "error": f"videos_root does not exist: {videos_root}",
            "n_files": 0,
            "n_parsed": 0,
            "n_unparsed": 0,
            "missing_keys": [list(k) for k in expected_keys()],
            "duplicate_keys": [],
        }

    for p in sorted(videos_root.rglob("*.mp4")):
        parsed = parse_filename(p.name)
        if parsed is None:
            unparsed.append(str(p))
            continue
        d = asdict(parsed)
        d["source_path"] = str(p.resolve())
        rows.append(d)

    df = pd.DataFrame(rows)
    key_cols = ["fiber_id", "color_label", "excitation_geometry", "raw_repeat_label"]
    missing: list[tuple] = []
    duplicates: list[tuple] = []
    if df.empty:
        present = set()
    else:
        present = set(tuple(x) for x in df[key_cols].itertuples(index=False, name=None))
        vc = df.groupby(key_cols).size()
        duplicates = [tuple(k) if not isinstance(k, tuple) else k for k, n in vc.items() if n > 1]

    for key in expected_keys():
        # expected: fiber int, color, coupling, repeat
        if key not in present:
            missing.append(key)

    per_fiber = {}
    for f in FIBERS:
        n = 0 if df.empty else int((df["fiber_id"] == f).sum())
        per_fiber[f] = n

    ok = (
        len(missing) == 0
        and len(duplicates) == 0
        and len(unparsed) == 0
        and (len(df) == 60)
        and all(v == 12 for v in per_fiber.values())
    )
    audit = {
        "ok": ok,
        "n_files_mp4": int(len(list(videos_root.rglob("*.mp4")))) if videos_root.is_dir() else 0,
        "n_parsed": int(len(df)),
        "n_unparsed": int(len(unparsed)),
        "unparsed_files": unparsed,
        "missing_keys": [list(k) for k in missing],
        "duplicate_keys": [list(k) for k in duplicates],
        "per_fiber_video_count": per_fiber,
        "expected_n_videos": 60,
        "complete_cartesian": len(missing) == 0 and len(duplicates) == 0 and len(df) == 60,
    }
    return df, audit


def write_manifest(df: pd.DataFrame, out_csv: Path) -> Path:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "source_path",
        "filename",
        "fiber_id",
        "raw_repeat_label",
        "repeat_id",
        "repeat_role",
        "wavelength_nm",
        "color_label",
        "excitation_geometry",
    ]
    if df.empty:
        pd.DataFrame(columns=cols).to_csv(out_csv, index=False)
    else:
        df.loc[:, [c for c in cols if c in df.columns]].to_csv(out_csv, index=False)
    return out_csv
