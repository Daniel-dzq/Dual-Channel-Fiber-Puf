"""Filename parsing and dataset discovery for Experiment 00."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from experiment00.acquisition import ORDER_SOURCE, resolve_acquisition_meta

# Green challenge videos:
#   canonical `{L}cm_Fxx_G_A_C01.mp4`
#   aliases   `{L}cm_Fxx_S0_A_C01.mp4` (state-0) or `{L}cm_Fxx_A_C01.mp4` (channel omitted)
GREEN_RE = re.compile(
    r"^(?P<length>7|9|11|13|15)cm_(?P<fiber>F\d{2})_(?:(?:G|S0)_)?(?P<round>A|B)_C(?P<challenge>\d{2})\.mp4$",
    re.IGNORECASE,
)
RED_RE = re.compile(
    r"^(?P<length>7|9|11|13|15)cm_(?P<fiber>F\d{2})_(?:R|red)_(?P<phase>before|after)\.mp4$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedVideo:
    filename: str
    length_cm: int
    fiber_id: str
    illumination: str  # green | red
    round: str | None  # A | B | None
    challenge: str | None  # legacy alias of challenge_id
    challenge_id: str | None  # C01..C08 | None
    red_phase: str | None  # before | after | None
    sequence_id: str | None
    acquisition_position: int | None
    acquisition_position_zero_based: int | None
    order_source: str | None
    key: str

    def to_dict(self) -> dict:
        return asdict(self)


class FilenameParseError(ValueError):
    pass


def parse_filename(
    name: str,
    *,
    assignment: dict[str, str] | None = None,
    sequences: dict[str, list[str]] | None = None,
) -> ParsedVideo:
    """Parse a video filename (case-insensitive). Raises FilenameParseError.

    Acquisition chronology comes only from fiber_id -> sequence_id -> challenge_id.
    """
    base = Path(name).name
    g = GREEN_RE.match(base)
    if g:
        ch = int(g.group("challenge"))
        if ch < 1 or ch > 8:
            raise FilenameParseError(f"Invalid challenge number in {base}")
        challenge_id = f"C{ch:02d}"
        fiber = g.group("fiber").upper()
        rnd = g.group("round").upper()
        length = int(g.group("length"))
        meta = resolve_acquisition_meta(
            fiber_id=fiber,
            challenge_id=challenge_id,
            illumination="green",
            assignment=assignment,
            sequences=sequences,
        )
        key = f"{length}cm|{fiber}|G|{rnd}|{challenge_id}"
        return ParsedVideo(
            filename=base,
            length_cm=length,
            fiber_id=fiber,
            illumination="green",
            round=rnd,
            challenge=challenge_id,
            challenge_id=challenge_id,
            red_phase=None,
            sequence_id=meta.sequence_id,
            acquisition_position=meta.acquisition_position,
            acquisition_position_zero_based=meta.acquisition_position_zero_based,
            order_source=meta.order_source or ORDER_SOURCE,
            key=key,
        )
    r = RED_RE.match(base)
    if r:
        fiber = r.group("fiber").upper()
        length = int(r.group("length"))
        phase = r.group("phase").lower()
        key = f"{length}cm|{fiber}|R|{phase}"
        return ParsedVideo(
            filename=base,
            length_cm=length,
            fiber_id=fiber,
            illumination="red",
            round=None,
            challenge=None,
            challenge_id=None,
            red_phase=phase,
            sequence_id=None,
            acquisition_position=None,
            acquisition_position_zero_based=None,
            order_source=None,
            key=key,
        )
    raise FilenameParseError(f"Unrecognized filename: {base}")


def discover_videos(videos_dir: Path) -> list[Path]:
    """List MP4 paths.

    Sorting is only for deterministic inventory enumeration / hashing.
    It is NOT chronological acquisition order (see acquisition.order_source).
    """
    if not videos_dir.exists():
        return []
    files: list[Path] = []
    for p in videos_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".mp4":
            files.append(p)
    # Deterministic listing only — never treat this as acquisition chronology.
    return sorted(files, key=lambda x: x.name.lower())


def build_inventory_rows(
    videos_dir: Path,
    *,
    assignment: dict[str, str] | None = None,
    sequences: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Build inventory without OpenCV probing (fast parse stage)."""
    rows: list[dict] = []
    seen_keys: dict[str, list[str]] = {}
    for path in discover_videos(videos_dir):
        try:
            parsed = parse_filename(path.name, assignment=assignment, sequences=sequences)
            status = "ok"
            err = ""
        except FilenameParseError as exc:
            parsed = None
            status = "parse_error"
            err = str(exc)
        key = parsed.key if parsed else path.name.lower()
        seen_keys.setdefault(key, []).append(str(path))
        rows.append(
            {
                "path": str(path.resolve()),
                "filename": path.name,
                "length_cm": parsed.length_cm if parsed else None,
                "fiber_id": parsed.fiber_id if parsed else None,
                "illumination": parsed.illumination if parsed else None,
                "round": parsed.round if parsed else None,
                "challenge": parsed.challenge if parsed else None,
                "challenge_id": parsed.challenge_id if parsed else None,
                "sequence_id": parsed.sequence_id if parsed else None,
                "acquisition_position": parsed.acquisition_position if parsed else None,
                "acquisition_position_zero_based": (
                    parsed.acquisition_position_zero_based if parsed else None
                ),
                "order_source": parsed.order_source if parsed else None,
                "red_phase": parsed.red_phase if parsed else None,
                "file_size": path.stat().st_size,
                "frame_count": None,
                "fps": None,
                "duration_s": None,
                "width": None,
                "height": None,
                "parse_status": status,
                "parse_error": err,
                "logical_key": key,
                "duplicate_status": "unknown",
                "validation_status": "pending",
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    dup_keys = {k for k, v in seen_keys.items() if len(v) > 1}
    df["duplicate_status"] = df["logical_key"].map(
        lambda k: "duplicate" if k in dup_keys else "unique"
    )
    return df
