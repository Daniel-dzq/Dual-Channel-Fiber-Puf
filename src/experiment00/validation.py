"""Dataset completeness validation for Experiment 00 (English-only code)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from experiment00.config import Experiment00Config
from experiment00.metadata import build_inventory_rows

# Files smaller than this are flagged (bytes).
SUSPICIOUS_MIN_BYTES = 50_000


@dataclass
class ValidationResult:
    ok: bool
    expected_total: int
    found_total: int
    inventory: pd.DataFrame
    missing: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    exploratory: bool = False
    dark_status: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    fix_list: list[str] = field(default_factory=list)


def expected_keys(cfg: Experiment00Config) -> list[str]:
    keys: list[str] = []
    for L in cfg.dataset.expected_lengths_cm:
        for f in cfg.dataset.expected_fibers:
            for rnd in cfg.dataset.expected_rounds:
                for ch in cfg.dataset.expected_challenges:
                    keys.append(f"{L}cm|{f}|G|{rnd}|{ch}")
            keys.append(f"{L}cm|{f}|R|before")
            keys.append(f"{L}cm|{f}|R|after")
    return keys


def expected_total(cfg: Experiment00Config) -> int:
    n_len = len(cfg.dataset.expected_lengths_cm)
    n_fib = len(cfg.dataset.expected_fibers)
    n_ch = len(cfg.dataset.expected_challenges)
    return n_len * n_fib * (2 * n_ch + 2)


def _check_dark(cfg: Experiment00Config) -> dict:
    """Validate dark availability. Prefer per-(length, fiber) bank under dark/."""
    from experiment00.video_processing import discover_dark_paths, _parse_dark_key

    info: dict = {
        "available": False,
        "compatible": False,
        "source": None,
        "warning": None,
        "shape": None,
        "mode": None,
        "n_mapped": 0,
        "missing_keys": [],
        "mapping": {},
        "note": "Per-fiber darks are matched 1:1 to (length_cm, fiber_id).",
    }
    paths = discover_dark_paths(cfg)
    if not paths:
        info["warning"] = (
            "No dark-field video/artifact found in dark/ or paths.dark_video. "
            "Formal analysis requires a real dark field unless allow_missing_dark=true "
            "(exploratory only)."
        )
        return info

    dark_root = cfg.dark_path()
    mapping: dict[str, str] = {}
    for p in paths:
        length, fiber = _parse_dark_key(p, dark_root)
        if length is None or fiber is None:
            continue
        key = f"{length}cm|{fiber}"
        mapping.setdefault(key, str(p.resolve()))

    expected = [
        f"{L}cm|{f}"
        for L in cfg.dataset.expected_lengths_cm
        for f in cfg.dataset.expected_fibers
    ]
    missing = [k for k in expected if k not in mapping]
    info["mapping"] = mapping
    info["n_mapped"] = len(mapping)
    info["missing_keys"] = missing
    info["available"] = len(mapping) > 0 or len(paths) > 0

    if mapping and not missing:
        info["mode"] = "per_length_fiber"
        info["source"] = "dark_bank"
        info["compatible"] = True
        # Probe one example
        sample = Path(next(iter(mapping.values())))
        if sample.suffix.lower() == ".mp4":
            try:
                from experiment00.video_processing import probe_video

                meta = probe_video(sample)
                info["shape"] = [int(meta["height"]), int(meta["width"])]
            except Exception as exc:  # noqa: BLE001
                info["compatible"] = False
                info["warning"] = f"Dark video unreadable: {exc}"
        return info

    if mapping and missing:
        info["mode"] = "per_length_fiber_incomplete"
        info["source"] = "dark_bank"
        info["compatible"] = bool(cfg.experiment.allow_missing_dark)
        info["warning"] = (
            f"Per-fiber dark bank incomplete: missing {len(missing)} / {len(expected)} "
            f"(e.g. {missing[:5]})"
        )
        # Formal runs treat incomplete bank as failure via available+compatible below
        if not cfg.experiment.allow_missing_dark:
            info["available"] = False
            info["compatible"] = False
        return info

    # Single shared dark fallback
    info["mode"] = "single_shared"
    info["source"] = str(paths[0].resolve())
    info["available"] = True
    try:
        from experiment00.video_processing import probe_video

        meta = probe_video(paths[0])
        info["shape"] = [int(meta["height"]), int(meta["width"])]
        info["compatible"] = True
        info["warning"] = (
            "Only unlabeled/shared dark found; prefer per-fiber darks under dark/."
        )
    except Exception as exc:  # noqa: BLE001
        info["compatible"] = False
        info["warning"] = f"Dark video unreadable: {exc}"
    return info


def validate_dataset(cfg: Experiment00Config, *, probe_video: bool = True) -> ValidationResult:
    videos_dir = cfg.videos_path()
    inv = build_inventory_rows(
        videos_dir,
        assignment=cfg.acquisition.sequence_assignment,
        sequences=cfg.acquisition.challenge_sequences,
    )
    exp_keys = expected_keys(cfg)
    exp_n = expected_total(cfg)
    messages: list[str] = []
    fix_list: list[str] = []

    parse_errors = (
        inv.loc[inv["parse_status"] != "ok", "filename"].astype(str).tolist() if not inv.empty else []
    )
    duplicates = (
        inv.loc[inv["duplicate_status"] == "duplicate", "filename"].astype(str).tolist()
        if not inv.empty
        else []
    )
    present = set()
    if not inv.empty:
        present = set(inv.loc[inv["parse_status"] == "ok", "logical_key"].astype(str))
    missing = [k for k in exp_keys if k not in present]

    # Zero-byte / small files
    small_files: list[str] = []
    if not inv.empty:
        for _, row in inv.iterrows():
            sz = int(row.get("file_size") or 0)
            if sz < SUSPICIOUS_MIN_BYTES:
                small_files.append(f"{row['filename']} ({sz} bytes)")
                inv.at[row.name, "validation_status"] = "suspicious_size"

    decode_failures: list[str] = []
    if probe_video and not inv.empty:
        from experiment00.video_processing import probe_video as _probe

        ok_rows = inv.index[inv["parse_status"] == "ok"].tolist()
        for i in tqdm(ok_rows, desc="Probe videos (OpenCV)", unit="vid"):
            row = inv.loc[i]
            try:
                meta = _probe(Path(row["path"]))
                inv.at[i, "frame_count"] = meta["n_frames"]
                inv.at[i, "fps"] = meta["fps"]
                inv.at[i, "width"] = meta["width"]
                inv.at[i, "height"] = meta["height"]
                inv.at[i, "duration_s"] = (
                    float(meta["n_frames"]) / float(meta["fps"]) if meta["fps"] else None
                )
                # Trim feasibility check
                fps = float(meta["fps"])
                n = int(meta["n_frames"])
                start = int(round(cfg.video.trim_start_s * fps))
                end = n - int(round(cfg.video.trim_end_s * fps))
                retained = max(0, end - start)
                inv.at[i, "retained_frames_est"] = retained
                if retained < cfg.video.temporal_blocks * cfg.video.minimum_valid_frames_per_block:
                    inv.at[i, "validation_status"] = "insufficient_after_trim"
                    decode_failures.append(f"{row['filename']}: insufficient frames after trim ({retained})")
                elif inv.at[i, "validation_status"] in (None, "pending", "ok", np.nan):
                    inv.at[i, "validation_status"] = "ok"
            except Exception as exc:  # noqa: BLE001
                inv.at[i, "validation_status"] = f"decode_fail:{exc}"
                decode_failures.append(f"{row['filename']}: {exc}")

    dark_status = _check_dark(cfg)

    found = int((inv["parse_status"] == "ok").sum()) if not inv.empty else 0
    exploratory = bool(cfg.experiment.allow_incomplete)
    ok = True

    def _fail(msg: str, fix: str | None = None) -> None:
        nonlocal ok
        ok = False
        messages.append(msg)
        if fix:
            fix_list.append(fix)

    if parse_errors:
        _fail(f"Unparsed files: {len(parse_errors)}", "Rename files to match green/red patterns; do not auto-rename.")
    if duplicates:
        _fail(
            f"Duplicate logical keys: {len(set(duplicates))}",
            "Remove or relocate duplicate videos that map to the same logical key.",
        )
    if missing:
        if cfg.experiment.strict_dataset and not exploratory:
            _fail(
                f"Missing files: {len(missing)} / expected {exp_n}",
                "Add the missing videos listed in missing_files.csv.",
            )
        else:
            messages.append(f"Missing files: {len(missing)} / expected {exp_n} (exploratory allowed)")
    if found == 0:
        _fail(f"No videos found in {videos_dir}", f"Place the 450-video dataset into {videos_dir}")
    if small_files:
        _fail(
            f"Zero-byte or suspiciously small files: {len(small_files)}",
            "Re-copy incomplete transfers; see validation summary for names.",
        )
    if decode_failures:
        _fail(
            f"Decode / trim failures: {len(decode_failures)}",
            "Fix corrupt videos or adjust acquisition length; never silently skip.",
        )
    if found == exp_n and not missing and not parse_errors and not duplicates:
        messages.append("Inventory structure: 450/450 logical keys present.")

    # Resolution mismatch
    if probe_video and not inv.empty and "width" in inv.columns:
        shapes = (
            inv.loc[inv["parse_status"] == "ok", ["width", "height"]]
            .dropna()
            .astype(int)
            .drop_duplicates()
        )
        if len(shapes) > 1 and cfg.quality_control.fail_on_resolution_mismatch:
            _fail(
                f"Resolution mismatch: {shapes.values.tolist()}",
                "Do not resize videos; re-acquire at matching resolution.",
            )

    if not dark_status.get("available"):
        if cfg.experiment.allow_missing_dark:
            messages.append("Dark field missing but allow_missing_dark=true (exploratory).")
        else:
            _fail(
                "Dark field unavailable.",
                "Place a dark MP4 in dark/ or set paths.dark_video "
                "(the public dark reference is calibration/dark_reference_2048x1536.npz).",
            )
    elif not dark_status.get("compatible", True):
        _fail("Dark field incompatible/unreadable.", str(dark_status.get("warning")))

    # Build summary tables
    summary = _build_summary(inv, cfg, found, exp_n, dark_status, small_files, decode_failures)

    if not inv.empty and "validation_status" in inv.columns:
        # Preserve decode/size flags; fill remaining ok parses
        mask_pending = inv["parse_status"].eq("ok") & inv["validation_status"].isin(
            ["pending", None, np.nan]
        )
        inv.loc[mask_pending, "validation_status"] = "ok"
        inv.loc[inv["parse_status"] != "ok", "validation_status"] = "parse_error"
        inv.loc[inv["duplicate_status"] == "duplicate", "validation_status"] = "duplicate"

    # Formal gate: must be exactly 450 ok videos
    if found != exp_n or missing or parse_errors or duplicates or decode_failures or small_files:
        if cfg.experiment.strict_dataset and not exploratory:
            ok = False

    from experiment00.sequence_balance import run_sequence_balance_check

    bal = run_sequence_balance_check(inv, cfg, out_dir=None)
    summary["sequence_balance_ok"] = bal["ok"]
    if not bal["ok"]:
        _fail(
            "Sequence balance FAILED (expected F01->P1 ... F05->P5 per length).",
            "Check fiber_id / sequence_assignment mapping; do not infer order from filenames.",
        )

    return ValidationResult(
        ok=ok,
        expected_total=exp_n,
        found_total=found,
        inventory=inv,
        missing=missing,
        duplicates=sorted(set(duplicates)),
        parse_errors=parse_errors,
        messages=messages,
        exploratory=exploratory and (bool(missing) or found != exp_n),
        dark_status=dark_status,
        summary=summary,
        fix_list=fix_list,
    )


def _build_summary(
    inv: pd.DataFrame,
    cfg: Experiment00Config,
    found: int,
    exp_n: int,
    dark_status: dict,
    small_files: list[str],
    decode_failures: list[str],
) -> dict:
    ok = inv.loc[inv["parse_status"] == "ok"] if not inv.empty else inv
    by_length = ok.groupby("length_cm").size().to_dict() if not ok.empty else {}
    by_fiber = ok.groupby("fiber_id").size().to_dict() if not ok.empty else {}
    by_illum = ok.groupby("illumination").size().to_dict() if not ok.empty else {}
    by_round = ok.loc[ok["illumination"] == "green"].groupby("round").size().to_dict() if not ok.empty else {}
    by_ch = (
        ok.loc[ok["illumination"] == "green"].groupby("challenge_id").size().to_dict()
        if not ok.empty and "challenge_id" in ok.columns
        else (
            ok.loc[ok["illumination"] == "green"].groupby("challenge").size().to_dict()
            if not ok.empty
            else {}
        )
    )
    by_seq = (
        ok.loc[ok["illumination"] == "green"].groupby("sequence_id").size().to_dict()
        if not ok.empty and "sequence_id" in ok.columns
        else {}
    )
    by_pos = (
        ok.loc[ok["illumination"] == "green"].groupby("acquisition_position").size().to_dict()
        if not ok.empty and "acquisition_position" in ok.columns
        else {}
    )
    red = ok.loc[ok["illumination"] == "red"] if not ok.empty else ok
    red_phase = red.groupby("red_phase").size().to_dict() if not red.empty else {}

    # Red completeness matrix
    red_missing = []
    for L in cfg.dataset.expected_lengths_cm:
        for f in cfg.dataset.expected_fibers:
            for phase in ("before", "after"):
                key = f"{L}cm|{f}|R|{phase}"
                if not ok.empty and key not in set(ok["logical_key"].astype(str)):
                    red_missing.append(key)

    res_dist = {}
    fps_dist = {}
    dur_stats = {}
    if not ok.empty and "width" in ok.columns and ok["width"].notna().any():
        shapes = ok.dropna(subset=["width", "height"]).apply(
            lambda r: f"{int(r.width)}x{int(r.height)}", axis=1
        )
        res_dist = dict(Counter(shapes.tolist()))
        fps_vals = pd.to_numeric(ok["fps"], errors="coerce").dropna().round(2)
        fps_dist = dict(Counter(fps_vals.tolist()))
        d = pd.to_numeric(ok["duration_s"], errors="coerce").dropna().astype(float)
        if len(d):
            dur_stats = {
                "min": float(d.min()),
                "median": float(d.median()),
                "max": float(d.max()),
            }

    return {
        "detected_total": found,
        "expected_total": exp_n,
        "by_length": {str(k): int(v) for k, v in by_length.items()},
        "by_fiber": {str(k): int(v) for k, v in by_fiber.items()},
        "by_illumination": {str(k): int(v) for k, v in by_illum.items()},
        "by_green_round": {str(k): int(v) for k, v in by_round.items()},
        "by_challenge": {str(k): int(v) for k, v in by_ch.items()},
        "by_sequence_id": {str(k): int(v) for k, v in by_seq.items()},
        "by_acquisition_position": {str(k): int(v) for k, v in by_pos.items()},
        "red_phase_counts": {str(k): int(v) for k, v in red_phase.items()},
        "red_missing_keys": red_missing,
        "small_files": small_files,
        "decode_failures": decode_failures,
        "resolution_distribution": res_dist,
        "fps_distribution": {str(k): int(v) for k, v in fps_dist.items()},
        "duration_s": dur_stats,
        "dark_status": dark_status,
        "theory": {
            "per_length": "5 fibers x 18 videos = 90",
            "per_fiber": "8 Round-A green + 8 Round-B green + 1 red before + 1 red after = 18",
            "complete": "5 x 90 = 450",
        },
    }


def print_validation_summary(result: ValidationResult) -> None:
    s = result.summary
    print("=" * 72)
    print("Experiment 00 — DATASET VALIDATION SUMMARY")
    print("=" * 72)
    print("Theory:")
    print("  Per length: 5 fibers x 18 videos = 90")
    print("  Per fiber:  8 Round-A green + 8 Round-B green + 1 red before + 1 red after = 18")
    print("  Complete:   5 x 90 = 450")
    print("-" * 72)
    print(f"Detected total: {result.found_total}")
    print(f"Expected total: {result.expected_total}")
    print(f"Validation OK:  {result.ok}")
    print(f"By length:      {s.get('by_length')}")
    print(f"By fiber:       {s.get('by_fiber')}")
    print(f"By illumination:{s.get('by_illumination')}")
    print(f"By green round: {s.get('by_green_round')}")
    print(f"By challenge:   {s.get('by_challenge')}")
    print(f"By sequence_id: {s.get('by_sequence_id')}")
    print(f"By acq. pos.:   {s.get('by_acquisition_position')}")
    print(f"Red phases:     {s.get('red_phase_counts')}")
    print(f"Red missing:    {len(s.get('red_missing_keys') or [])}")
    print(f"Missing keys:   {len(result.missing)}")
    print(f"Duplicates:     {len(result.duplicates)}")
    print(f"Unparsed:       {len(result.parse_errors)}")
    print(f"Small files:    {len(s.get('small_files') or [])}")
    print(f"Decode fails:   {len(s.get('decode_failures') or [])}")
    print(f"Resolutions:    {s.get('resolution_distribution')}")
    print(f"FPS:            {s.get('fps_distribution')}")
    print(f"Duration (s):   {s.get('duration_s')}")
    print(f"Dark field:     {s.get('dark_status')}")
    print("-" * 72)
    for m in result.messages:
        print(f"MSG: {m}")
    if result.fix_list:
        print("FIX LIST:")
        for i, f in enumerate(result.fix_list, 1):
            print(f"  {i}. {f}")
    print("=" * 72)


def write_validation_tables(
    result: ValidationResult,
    out_dir: Path,
    cfg: Experiment00Config | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.inventory.to_csv(out_dir / "dataset_inventory.csv", index=False)
    pd.DataFrame({"missing_key": result.missing}).to_csv(out_dir / "missing_files.csv", index=False)
    pd.DataFrame({"duplicate_filename": result.duplicates}).to_csv(
        out_dir / "duplicate_files.csv", index=False
    )
    pd.DataFrame({"parse_error_filename": result.parse_errors}).to_csv(
        out_dir / "unparsed_files.csv", index=False
    )
    from experiment00.cache import dump_json
    from experiment00.sequence_balance import run_sequence_balance_check

    dump_json(out_dir / "validation_summary.json", {
        "ok": result.ok,
        "found_total": result.found_total,
        "expected_total": result.expected_total,
        "messages": result.messages,
        "fix_list": result.fix_list,
        "summary": result.summary,
        "dark_status": result.dark_status,
    })
    if cfg is not None:
        run_sequence_balance_check(result.inventory, cfg, out_dir)
