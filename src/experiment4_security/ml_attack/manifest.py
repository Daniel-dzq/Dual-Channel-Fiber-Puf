"""Recursive scan + filename parsing for the F01 per-clip video dataset.

Ground truth for challenge identity is `challenge_id` parsed from the file
name; `presentation_index` (the playback order used only in Round B, which is
shuffled) must never be used as, or confused with, challenge identity.

Files that cannot be parsed are marked ERROR and excluded from every
downstream training/evaluation step.
"""

from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

MANIFEST_COLUMNS = [
    "sample_id",
    "device_id",
    "source_device_id",
    "state_id",
    "state_id_from_filename",
    "state_id_conflict",
    "round_id",
    "presentation_index",
    "challenge_id",
    "challenge_number",
    "bank_id",
    "video_path",
    "file_name",
    "file_size_bytes",
    "duration_s",
    "fps",
    "frame_count_reported",
    "frame_count_decoded",
    "decode_status",
    "duplicate_status",
    "manifest_status",
    "parse_error",
]

# Strict pattern for the real acquisition naming convention actually used on
# disk: {round}_{presentation_index}_{challenge_id}_{state_id}_{device}.ext
# Examples: A_1_C001_M0_F1.mp4 ; B_83_C007_M1_F1.mp4
#
# CRITICAL: state_id must accept M0..M7 as strings. Never treat the integer
# state_number with truthiness (`if state_number:`) -- M0 yields 0 and would
# be silently dropped. Always use `if state_number is not None`.
STRICT_PATTERN = re.compile(
    r"^(?P<round_id>[AB])_(?P<presentation_index>\d+)_(?P<challenge_id>C\d{3})_"
    r"(?P<state_id>M[0-7])_(?P<source_device_id>F\d+)\.(?P<ext>mp4|avi|mov|MP4|AVI|MOV)$"
)
# Legacy numeric-capture form kept for older synthetic fixtures in unit tests.
STRICT_PATTERN_LEGACY = re.compile(
    r"^(?P<round>[A-Za-z])_(?P<pidx>\d+)_C(?P<chal>\d+)_M(?P<state>\d+)_(?P<device>[A-Za-z0-9]+)\.(?P<ext>mp4|avi|mov|MP4|AVI|MOV)$"
)

# Looser fallback patterns kept ONLY for robustness against alternative
# naming conventions explicitly enumerated by the protocol
# (F01_M0_A_C001.avi / F01/M4/Round_A/C037.avi style). These never guess
# challenge_id from sort order; if challenge_id truly cannot be recovered the
# file is ERROR.
FALLBACK_PATTERNS = [
    re.compile(
        r"^(?P<device>[A-Za-z0-9]+)_M(?P<state>\d+)_(?P<round>[A-Za-z])_C(?P<chal>\d+)\.(?P<ext>mp4|avi|mov|MP4|AVI|MOV)$"
    ),
]

DEVICE_ALIASES = {
    "F1": "F01",
    "FIBER1": "F01",
    "FIBER_1": "F01",
    "F01": "F01",
}

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".MP4", ".AVI", ".MOV")


def normalize_device_id(token: str) -> str:
    key = token.strip().upper().replace("-", "_")
    if key in DEVICE_ALIASES:
        return DEVICE_ALIASES[key]
    m = re.match(r"^F0*([0-9]+)$", key)
    if m:
        return f"F{int(m.group(1)):02d}"
    if key.startswith("FIBER"):
        digits = re.findall(r"\d+", key)
        if digits:
            return f"F{int(digits[0]):02d}"
    return key


@dataclass
class ParsedName:
    round_id: str
    presentation_index: int
    challenge_number: int
    state_number_from_filename: int
    source_device_token: str


def parse_filename(file_name: str) -> ParsedName | None:
    m = STRICT_PATTERN.match(file_name)
    if m is None:
        m = STRICT_PATTERN_LEGACY.match(file_name)
    if m is None:
        for pat in FALLBACK_PATTERNS:
            m = pat.match(file_name)
            if m is not None:
                break
    if m is None:
        return None
    gd = m.groupdict()
    # Prefer named groups from the primary STRICT_PATTERN; fall back to legacy.
    if "challenge_id" in gd and gd.get("challenge_id"):
        chal_num = int(gd["challenge_id"][1:])  # "C081" -> 81
        state_token = gd["state_id"]  # "M0".."M7"
        state_num = int(state_token[1:])
        # Guard against the classic zero-as-False bug class.
        if state_num is None:  # pragma: no cover
            raise RuntimeError("state_number unexpectedly None")
        return ParsedName(
            round_id=str(gd["round_id"]).strip().upper(),
            presentation_index=int(gd["presentation_index"]),
            challenge_number=chal_num,
            state_number_from_filename=state_num,
            source_device_token=gd["source_device_id"],
        )
    return ParsedName(
        round_id=gd["round"].strip().upper(),
        presentation_index=int(gd["pidx"]) if gd.get("pidx") is not None else -1,
        challenge_number=int(gd["chal"]),
        state_number_from_filename=int(gd["state"]),
        source_device_token=gd["device"],
    )


def bank_id_for_challenge_number(n: int, bank_size: int = 8) -> str:
    idx = (n - 1) // bank_size + 1
    return f"B{idx:02d}"


def _state_number_from_dir(state_dir_name: str) -> int | None:
    m = re.match(r"^M(\d+)$", state_dir_name.strip(), re.IGNORECASE)
    if m is None:
        return None
    # Must use `is not None` -- M0 yields integer 0.
    num = int(m.group(1))
    return num if num is not None else None  # noqa: SIM210 -- explicit zero-safe guard


def scan_videos(
    videos_root: Path,
    *,
    states: list[str],
    device_id: str,
    source_device_id: str,
    probe_decode: bool = True,
    n_workers: int = 1,
) -> pd.DataFrame:
    """Recursively scan `videos_root` for per-clip videos and parse identity.

    `videos_root` is expected to directly contain one subdirectory per
    mechanical state (e.g. `M0` .. `M7`), each holding that state's videos
    flat (Round A and Round B interleaved, one file per challenge per round).

    `n_workers` only controls HOW the (identical, per-file) decode probe in
    `probe_video_basic` is scheduled -- sequentially in-process (`<=1`, the
    default, used by all existing tests) or across a `ProcessPoolExecutor`
    (`>1`, for the real ~2000-video pilot run). It never changes what is
    computed or the resulting row order.
    """
    videos_root = Path(videos_root)
    rows: list[dict[str, Any]] = []
    # (row_index_in_rows, absolute_file_path) for every OK-parsed file that
    # still needs its decode probe filled in below.
    pending_probes: list[tuple[int, Path]] = []

    if not videos_root.exists():
        raise FileNotFoundError(f"videos_root does not exist: {videos_root}")

    state_dirs = sorted([d for d in videos_root.iterdir() if d.is_dir()])
    for state_dir in state_dirs:
        dir_state_num = _state_number_from_dir(state_dir.name)
        dir_state_id = f"M{dir_state_num}" if dir_state_num is not None else state_dir.name
        files = sorted([f for f in state_dir.iterdir() if f.is_file() and f.suffix in VIDEO_EXTS])
        for f in files:
            parsed = parse_filename(f.name)
            file_size = f.stat().st_size
            if parsed is None:
                rows.append(
                    {
                        "sample_id": None,
                        "device_id": device_id,
                        "source_device_id": None,
                        "state_id": dir_state_id if dir_state_id in states else None,
                        "state_id_from_filename": None,
                        "state_id_conflict": None,
                        "round_id": None,
                        "presentation_index": None,
                        "challenge_id": None,
                        "challenge_number": None,
                        "bank_id": None,
                        "video_path": str(f.resolve()),
                        "file_name": f.name,
                        "file_size_bytes": file_size,
                        "duration_s": None,
                        "fps": None,
                        "frame_count_reported": None,
                        "frame_count_decoded": None,
                        "decode_status": "NOT_ATTEMPTED",
                        "duplicate_status": None,
                        "manifest_status": "ERROR_UNPARSEABLE_FILENAME",
                        "parse_error": f"filename does not match any known pattern: {f.name}",
                    }
                )
                continue

            norm_device = normalize_device_id(parsed.source_device_token)
            filename_state_id = f"M{parsed.state_number_from_filename}"
            state_conflict = dir_state_id != filename_state_id
            challenge_id = f"C{parsed.challenge_number:03d}"

            status = "OK"
            if norm_device != device_id:
                status = "WARNING_DEVICE_ID_MISMATCH"
            if state_conflict:
                status = "WARNING_STATE_ID_CONFLICT"

            duration_s = fps = frame_count_reported = frame_count_decoded = None
            decode_status = "NOT_ATTEMPTED"
            if probe_decode and n_workers <= 1:
                probe = probe_video_basic(f)
                duration_s = probe["duration_s"]
                fps = probe["fps"]
                frame_count_reported = probe["frame_count_reported"]
                frame_count_decoded = probe["frame_count_decoded"]
                decode_status = probe["decode_status"]
                if decode_status != "OK":
                    status = "ERROR_DECODE_FAILED"
            elif probe_decode:
                pending_probes.append((len(rows), f))

            rows.append(
                {
                    "sample_id": f"{device_id}_{dir_state_id}_{parsed.round_id}_{challenge_id}",
                    "device_id": device_id,
                    "source_device_id": source_device_id if norm_device != device_id else norm_device,
                    "state_id": dir_state_id,
                    "state_id_from_filename": filename_state_id,
                    "state_id_conflict": bool(state_conflict),
                    "round_id": parsed.round_id,
                    "presentation_index": parsed.presentation_index,
                    "challenge_id": challenge_id,
                    "challenge_number": parsed.challenge_number,
                    "bank_id": bank_id_for_challenge_number(parsed.challenge_number),
                    "video_path": str(f.resolve()),
                    "file_name": f.name,
                    "file_size_bytes": file_size,
                    "duration_s": duration_s,
                    "fps": fps,
                    "frame_count_reported": frame_count_reported,
                    "frame_count_decoded": frame_count_decoded,
                    "decode_status": decode_status,
                    "duplicate_status": "UNIQUE",
                    "manifest_status": status,
                    "parse_error": None,
                }
            )

    if pending_probes:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            probe_results = list(ex.map(probe_video_basic, [p for _, p in pending_probes]))
        for (row_idx, _), probe in zip(pending_probes, probe_results):
            rows[row_idx]["duration_s"] = probe["duration_s"]
            rows[row_idx]["fps"] = probe["fps"]
            rows[row_idx]["frame_count_reported"] = probe["frame_count_reported"]
            rows[row_idx]["frame_count_decoded"] = probe["frame_count_decoded"]
            rows[row_idx]["decode_status"] = probe["decode_status"]
            if probe["decode_status"] != "OK" and rows[row_idx]["manifest_status"] == "OK":
                rows[row_idx]["manifest_status"] = "ERROR_DECODE_FAILED"

    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    df = _flag_duplicates(df)
    return df


def _flag_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    key_cols = ["device_id", "state_id", "round_id", "challenge_id"]
    valid = df["sample_id"].notna()
    counts = df.loc[valid].groupby(key_cols)["video_path"].transform("count")
    dup_mask = pd.Series(False, index=df.index)
    dup_mask.loc[valid] = counts.to_numpy() > 1
    df.loc[dup_mask, "duplicate_status"] = "DUPLICATE"
    df.loc[dup_mask & (df["manifest_status"] == "OK"), "manifest_status"] = "ERROR_DUPLICATE_CHALLENGE"
    return df


def probe_video_basic(path: Path) -> dict[str, Any]:
    """Cheap-ish decode probe: open, read every frame, report counts.

    OpenCV's reported `CAP_PROP_FRAME_COUNT` is unreliable for these H.264
    clips (observed to overcount vs. actually decodable frames), so
    `frame_count_decoded` -- obtained by actually reading every frame -- is
    the authoritative count used everywhere downstream.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {
            "duration_s": None,
            "fps": None,
            "frame_count_reported": None,
            "frame_count_decoded": 0,
            "decode_status": "ERROR_CANNOT_OPEN",
        }
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count_reported = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    n_decoded = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None or frame.ndim != 3 or frame.shape[2] < 3:
                cap.release()
                return {
                    "duration_s": None,
                    "fps": fps if fps > 0 else None,
                    "frame_count_reported": frame_count_reported,
                    "frame_count_decoded": n_decoded,
                    "decode_status": "ERROR_NON_COLOR_FRAME",
                }
            n_decoded += 1
    finally:
        cap.release()
    if n_decoded == 0:
        return {
            "duration_s": None,
            "fps": fps if fps > 0 else None,
            "frame_count_reported": frame_count_reported,
            "frame_count_decoded": 0,
            "decode_status": "ERROR_NO_FRAMES",
        }
    duration_s = n_decoded / fps if fps > 0 else None
    return {
        "duration_s": duration_s,
        "fps": fps if fps > 0 else None,
        "frame_count_reported": frame_count_reported,
        "frame_count_decoded": n_decoded,
        "decode_status": "OK",
    }


@dataclass
class CoverageReport:
    state_id: str
    round_id: str
    n_files: int
    unique_challenges: int
    missing_challenge_ids: list[str] = field(default_factory=list)
    duplicate_challenge_ids: list[str] = field(default_factory=list)
    covers_c001_c128: bool = False


def compute_coverage(
    df: pd.DataFrame, *, states: list[str], rounds: list[str], challenge_start: int, challenge_end: int
) -> list[CoverageReport]:
    expected = {f"C{i:03d}" for i in range(challenge_start, challenge_end + 1)}
    reports: list[CoverageReport] = []
    valid = df[df["challenge_id"].notna()]
    for state in states:
        for rnd in rounds:
            sub = valid[(valid["state_id"] == state) & (valid["round_id"] == rnd)]
            ids = sub["challenge_id"].tolist()
            uniq = sorted(set(ids))
            counts = pd.Series(ids).value_counts()
            dupes = sorted(counts[counts > 1].index.tolist())
            missing = sorted(expected - set(uniq))
            reports.append(
                CoverageReport(
                    state_id=state,
                    round_id=rnd,
                    n_files=len(sub),
                    unique_challenges=len(uniq),
                    missing_challenge_ids=missing,
                    duplicate_challenge_ids=dupes,
                    covers_c001_c128=(len(missing) == 0 and set(uniq) == expected),
                )
            )
    return reports
