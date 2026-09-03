"""Global valid mask (independent of length labels / authentication scores)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiment00.config import Experiment00Config
from experiment00.video_processing import ProcessedVideo
from puf_common.masks import build_valid_mask_from_refs, mask_hash, save_mask_png


def build_global_valid_mask(
    green_recordings: list[ProcessedVideo],
    cfg: Experiment00Config,
    out_dir: Path,
) -> tuple[np.ndarray, dict]:
    """Build one global mask from a balanced subset of green representatives.

    Selection is deterministic and does NOT depend on authentication scores or
    which length performs best.
    """
    if not green_recordings:
        raise ValueError("No green recordings for mask construction")
    # Deterministic: sort by (length, fiber, round, challenge), take one per fiber
    # across all lengths (up to N refs).
    sorted_recs = sorted(
        green_recordings,
        key=lambda r: (r.length_cm, r.fiber_id, r.round or "", r.challenge or ""),
    )
    refs: list[np.ndarray] = []
    used: list[str] = []
    seen_fiber_length: set[tuple[int, str]] = set()
    for rec in sorted_recs:
        key = (rec.length_cm, rec.fiber_id)
        if key in seen_fiber_length:
            continue
        # Prefer Round A C01 when available later in sort order — take first hit
        if rec.round == "A" and rec.challenge == "C01":
            refs.append(rec.representative_raw)
            used.append(rec.path.name)
            seen_fiber_length.add(key)
    # Fallback fill if some fiber-lengths lacked A/C01
    if len(refs) < 5:
        for rec in sorted_recs:
            key = (rec.length_cm, rec.fiber_id)
            if key in seen_fiber_length:
                continue
            refs.append(rec.representative_raw)
            used.append(rec.path.name)
            seen_fiber_length.add(key)
            if len(refs) >= 25:
                break

    mask, coverage, thr = build_valid_mask_from_refs(
        refs,
        abs_floor=cfg.preprocessing.mask_abs_floor,
        percentile=cfg.preprocessing.mask_percentile,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "global_valid_mask.npy", mask.astype(np.uint8))
    save_mask_png(mask, out_dir / "global_valid_mask.png")
    meta = {
        "coverage": float(coverage),
        "threshold": float(thr),
        "n_refs": len(refs),
        "ref_files": used,
        "mask_hash": mask_hash(mask),
        "shape": list(mask.shape),
        "note": "Mask independent of length-optimality scores",
    }
    return mask.astype(bool), meta
