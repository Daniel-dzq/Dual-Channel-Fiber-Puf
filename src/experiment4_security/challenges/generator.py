"""Generate fixed m=2 128-pattern challenge library."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from experiment4_security.common.config import ChallengeLibraryConfig, repo_root
from experiment4_security.common.frozen_protocol import GENERATION_VERSION, MASTER_SEED
from experiment4_security.common.progress_util import stage_tqdm
from experiment4_security.challenges.manifest import bank_id_for_index, write_manifests
from experiment4_security.challenges.plotting import render_contact_sheet
from experiment4_security.challenges.validator import validate_library


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def expand_macro_nearest(grid: np.ndarray, macro: int) -> np.ndarray:
    if grid.ndim != 2:
        raise ValueError("macro grid must be 2D")
    if not np.isin(grid, [0, 1]).all():
        raise ValueError("macro grid must be binary {0,1}")
    return np.repeat(np.repeat(grid.astype(np.uint8), macro, axis=0), macro, axis=1)


def place_on_canvas(
    active_gray: np.ndarray,
    *,
    canvas_w: int,
    canvas_h: int,
    ox: int,
    oy: int,
    closed: int,
) -> np.ndarray:
    canvas = np.full((canvas_h, canvas_w), int(closed), dtype=np.uint8)
    ah, aw = active_gray.shape
    canvas[oy : oy + ah, ox : ox + aw] = active_gray
    return canvas


def balanced_macro_grid(rng: np.random.Generator, grid_w: int, grid_h: int) -> np.ndarray:
    n = grid_w * grid_h
    if n % 2 != 0:
        raise ValueError("macro-cell count must be even for exact 50% open")
    vec = np.array([1] * (n // 2) + [0] * (n // 2), dtype=np.uint8)
    rng.shuffle(vec)
    return vec.reshape(grid_h, grid_w)


def generated_grid(master_seed: int, index: int, attempt: int, grid_w: int, grid_h: int) -> np.ndarray:
    """Candidate grid for challenge ``index`` (1-based) at rejection ``attempt`` under ``master_seed``.

    Challenges C009-C128 of ``m2_128_v1`` are the first candidates that pass the pairwise
    acceptance criteria (no duplicate / complement, |HD z-score| <= 3.5, |input NCC| <= 0.02).
    """
    rng = np.random.default_rng(np.random.SeedSequence([master_seed, index, attempt]))
    return balanced_macro_grid(rng, grid_w, grid_h)


def grid_from_canvas(
    canvas: np.ndarray,
    *,
    ox: int,
    oy: int,
    aw: int,
    ah: int,
    macro: int,
    open_level: int,
) -> np.ndarray:
    active = canvas[oy : oy + ah, ox : ox + aw]
    # Sample top-left of each macro cell.
    return (active[::macro, ::macro] >= (open_level // 2)).astype(np.uint8)


def _find_canonical(cfg: ChallengeLibraryConfig, idx: int) -> Path | None:
    rel = cfg.canonical_source_glob.format(idx=idx)
    p = repo_root() / rel
    return p if p.exists() else None


def _write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path, format="PNG", compress_level=0)


def generate_library(cfg: ChallengeLibraryConfig) -> dict[str, Any]:
    out = cfg.output_root
    patterns = out / "patterns"
    if out.exists() and any(out.iterdir()) and not cfg.overwrite:
        raise FileExistsError(
            f"Challenge library already exists at {out}. "
            "Pass overwrite=true or request a new version."
        )
    out.mkdir(parents=True, exist_ok=True)
    patterns.mkdir(parents=True, exist_ok=True)

    grid_w = cfg.active_width_px // cfg.macro_pixel_size
    grid_h = cfg.active_height_px // cfg.macro_pixel_size
    if grid_w * cfg.macro_pixel_size != cfg.active_width_px:
        raise ValueError("active_width_px must be divisible by macro_pixel_size")
    if grid_h * cfg.macro_pixel_size != cfg.active_height_px:
        raise ValueError("active_height_px must be divisible by macro_pixel_size")

    ss = np.random.SeedSequence(cfg.master_seed)
    child_seeds = ss.spawn(cfg.n_challenges + 8)

    rows: list[dict[str, Any]] = []
    grids: list[np.ndarray] = []
    preserved = 0
    newly = 0

    # Special patterns
    closed = np.full(
        (cfg.canvas_height_px, cfg.canvas_width_px), cfg.closed_level, dtype=np.uint8
    )
    open_active = np.full(
        (cfg.active_height_px, cfg.active_width_px), cfg.open_level, dtype=np.uint8
    )
    open_canvas = place_on_canvas(
        open_active,
        canvas_w=cfg.canvas_width_px,
        canvas_h=cfg.canvas_height_px,
        ox=cfg.active_offset_x_px,
        oy=cfg.active_offset_y_px,
        closed=cfg.closed_level,
    )
    _write_png(patterns / "uniform_closed.png", closed)
    _write_png(patterns / "uniform_open.png", open_canvas)

    # Control: half-open vertical stripe in active region
    control_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    control_grid[:, : grid_w // 2] = 1
    control_active = expand_macro_nearest(control_grid, cfg.macro_pixel_size)
    control_active = np.where(control_active > 0, cfg.open_level, cfg.closed_level).astype(
        np.uint8
    )
    control = place_on_canvas(
        control_active,
        canvas_w=cfg.canvas_width_px,
        canvas_h=cfg.canvas_height_px,
        ox=cfg.active_offset_x_px,
        oy=cfg.active_offset_y_px,
        closed=cfg.closed_level,
    )
    _write_png(patterns / "C000_control.png", control)

    for i in stage_tqdm(range(1, cfg.n_challenges + 1), desc="Challenges", unit="pat"):
        cid = f"C{i:03d}"
        dest = patterns / f"{cid}.png"
        canonical_source = ""
        seed_int = int(child_seeds[i - 1].generate_state(1)[0])
        status = "newly_generated"

        if i <= 8:
            src = _find_canonical(cfg, i)
            if src is not None:
                # Preserve exact historical bytes (Exp1 mp002 C01-C08).
                shutil.copyfile(src, dest)
                canvas = np.asarray(Image.open(dest).convert("L"))
                if canvas.shape != (cfg.canvas_height_px, cfg.canvas_width_px):
                    raise ValueError(
                        f"Canonical {src} shape {canvas.shape} != "
                        f"({cfg.canvas_height_px}, {cfg.canvas_width_px})"
                    )
                grid = grid_from_canvas(
                    canvas,
                    ox=cfg.active_offset_x_px,
                    oy=cfg.active_offset_y_px,
                    aw=cfg.active_width_px,
                    ah=cfg.active_height_px,
                    macro=cfg.macro_pixel_size,
                    open_level=cfg.open_level,
                )
                canonical_source = str(src.relative_to(repo_root()))
                status = "preserved"
                preserved += 1
            else:
                rng = np.random.default_rng(child_seeds[i - 1])
                grid = balanced_macro_grid(rng, grid_w, grid_h)
                active = expand_macro_nearest(grid, cfg.macro_pixel_size)
                active = np.where(active > 0, cfg.open_level, cfg.closed_level).astype(
                    np.uint8
                )
                canvas = place_on_canvas(
                    active,
                    canvas_w=cfg.canvas_width_px,
                    canvas_h=cfg.canvas_height_px,
                    ox=cfg.active_offset_x_px,
                    oy=cfg.active_offset_y_px,
                    closed=cfg.closed_level,
                )
                _write_png(dest, canvas)
                newly += 1
        else:
            # Reject on duplicates/complements and pairwise HD-z / input NCC.
            n_cells = grid_w * grid_h
            denom = np.sqrt(0.25 / n_cells)
            attempt = 0
            while True:
                grid = generated_grid(cfg.master_seed, i, attempt, grid_w, grid_h)
                ok = True
                if any(np.array_equal(grid, g) for g in grids):
                    ok = False
                if ok and any(np.array_equal(1 - grid, g) for g in grids):
                    ok = False
                if ok:
                    for g in grids:
                        hd = float(np.mean(grid != g))
                        z = abs((hd - 0.5) / denom)
                        if z > cfg.max_abs_hd_z:
                            ok = False
                            break
                        a = grid.astype(np.float64)
                        b = g.astype(np.float64)
                        a = a - a.mean()
                        b = b - b.mean()
                        ncc = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                        if abs(ncc) > cfg.max_abs_input_ncc:
                            ok = False
                            break
                if ok:
                    break
                attempt += 1
                if attempt > 20000:
                    raise RuntimeError(f"Failed to sample valid pattern for {cid}")
            active = expand_macro_nearest(grid, cfg.macro_pixel_size)
            active = np.where(active > 0, cfg.open_level, cfg.closed_level).astype(np.uint8)
            canvas = place_on_canvas(
                active,
                canvas_w=cfg.canvas_width_px,
                canvas_h=cfg.canvas_height_px,
                ox=cfg.active_offset_x_px,
                oy=cfg.active_offset_y_px,
                closed=cfg.closed_level,
            )
            _write_png(dest, canvas)
            newly += 1

        sha = _sha256_file(dest)
        n_open = int(grid.sum())
        rows.append(
            {
                "challenge_id": cid,
                "bank_id": bank_id_for_index(i),
                "bank_local_index": ((i - 1) % 8) + 1,
                "macro_pixel_size": cfg.macro_pixel_size,
                "canvas_width_px": cfg.canvas_width_px,
                "canvas_height_px": cfg.canvas_height_px,
                "active_width_px": cfg.active_width_px,
                "active_height_px": cfg.active_height_px,
                "active_offset_x_px": cfg.active_offset_x_px,
                "active_offset_y_px": cfg.active_offset_y_px,
                "grid_width": grid_w,
                "grid_height": grid_h,
                "n_macro_cells": int(grid_w * grid_h),
                "n_open_cells": n_open,
                "open_fraction": float(n_open / (grid_w * grid_h)),
                "seed": seed_int,
                "sha256": sha,
                "pattern_path": f"patterns/{cid}.png",
                "canonical_source": canonical_source,
                "generation_version": cfg.version or GENERATION_VERSION,
                "canonical_anchor_status": status,
            }
        )
        grids.append(grid)

    if preserved == 8:
        anchor_status = "preserved"
    elif preserved == 0:
        anchor_status = "newly_generated"
    else:
        anchor_status = "partial_preserved"

    write_manifests(out, rows)
    render_contact_sheet(patterns, out / "challenge_contact_sheet.png", out / "challenge_contact_sheet.pdf")

    report = validate_library(
        out,
        grids=grids,
        max_abs_hd_z=cfg.max_abs_hd_z,
        max_abs_input_ncc=cfg.max_abs_input_ncc,
    )
    report["canonical_anchor_status"] = anchor_status
    report["n_preserved_canonical"] = preserved
    report["n_newly_generated"] = newly
    report["master_seed"] = cfg.master_seed
    report["polarity_warning"] = (
        "Confirm physical amplitude-SLM polarity before acquisition "
        f"(closed_level={cfg.closed_level}, open_level={cfg.open_level})."
    )
    if anchor_status != "preserved":
        report["continuity_note"] = (
            "C001-C008 are not byte-identical to historical Experiment 1/3 patterns."
        )
    else:
        report["continuity_note"] = (
            "C001-C008 preserve exact Experiment 1 mp002 C01-C08 PNG bytes."
        )

    (out / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_readme(out, report, cfg)

    if not report.get("passed", False):
        raise RuntimeError(f"Challenge library validation failed: {report.get('failures')}")
    return report


def _write_readme(out: Path, report: dict[str, Any], cfg: ChallengeLibraryConfig) -> None:
    text = f"""# Challenge library `{cfg.version}`

Fixed m=2 binary amplitude challenge library for Experiment 4.

## Geometry

- Canvas: {cfg.canvas_width_px} × {cfg.canvas_height_px}
- Active: {cfg.active_width_px} × {cfg.active_height_px} at offset ({cfg.active_offset_x_px}, {cfg.active_offset_y_px})
- Macro-pixel size: {cfg.macro_pixel_size}
- Closed / open levels: {cfg.closed_level} / {cfg.open_level}

**Warning:** Confirm physical amplitude-SLM polarity before acquisition.

## Canonical anchors

- Status: `{report.get('canonical_anchor_status')}`
- {report.get('continuity_note')}

## Validation

- Passed: {report.get('passed')}
- Master seed: {cfg.master_seed}

Banks B01–B16 group C001–C128 in blocks of eight for analysis only (same acquisition timing).
"""
    (out / "README.md").write_text(text, encoding="utf-8")
