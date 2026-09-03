"""Generate C01-C08 screening challenges for macro-pixel size selection.

Uses the fixed 512x512 active region on a 1024x768 SLM canvas.
All challenges have exact 50% white duty at the macro-cell level and
pairwise Hamming z-scores constrained to |z| <= max_abs_z.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from e01.config import AppConfig
from e01.naming import PathLayout, challenge_grid_size, mp_tag
from e01.patterns.canvas import blank_canvas, expand_binary_nearest, map_binary_to_gray, place_active_region
from e01.patterns.challenges import (
    duty_cycle,
    generate_balanced_binary,
    normalized_hamming_distance,
)
from e01.utils.hashing import sha256_array, sha256_file
from e01.utils.io_utils import assert_writable, ensure_dir

logger = logging.getLogger(__name__)

DEFAULT_CHALLENGE_IDS = [f"C{i:02d}" for i in range(1, 9)]
DEFAULT_MAX_ABS_Z = 2.5


def hamming_z(hd: float, n_cells: int) -> float:
    """Standardized Hamming distance relative to random 0.5 baseline."""
    if n_cells <= 0:
        raise ValueError("n_cells must be positive")
    denom = np.sqrt(0.25 / float(n_cells))
    return float((hd - 0.5) / denom)


def generate_independent_set(
    grid: int,
    n_challenges: int,
    rng: np.random.Generator,
    max_abs_z: float = DEFAULT_MAX_ABS_Z,
    max_tries: int = 200000,
) -> tuple[list[np.ndarray], list[tuple[int, int, float, float]]]:
    """Generate n balanced binaries with pairwise |z| <= max_abs_z."""
    n_cells = grid * grid
    accepted: list[np.ndarray] = []
    pair_stats: list[tuple[int, int, float, float]] = []
    tries = 0
    while len(accepted) < n_challenges:
        tries += 1
        if tries > max_tries:
            raise RuntimeError(
                f"Failed to build {n_challenges} independent challenges "
                f"for grid={grid} with |z|<={max_abs_z} after {max_tries} tries"
            )
        cand = generate_balanced_binary(rng, grid)
        ok = True
        provisional: list[tuple[int, int, float, float]] = []
        for i, prev in enumerate(accepted):
            hd = normalized_hamming_distance(prev, cand)
            z = hamming_z(hd, n_cells)
            if abs(z) > max_abs_z:
                ok = False
                break
            provisional.append((i, len(accepted), hd, z))
        if not ok:
            continue
        accepted.append(cand)
        pair_stats.extend(provisional)
    return accepted, pair_stats


def _save_png(path: Path, arr: np.ndarray, force: bool) -> None:
    ensure_dir(path.parent)
    assert_writable(path, force=force)
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)


def _save_npy(path: Path, arr: np.ndarray, force: bool) -> None:
    ensure_dir(path.parent)
    assert_writable(path, force=force)
    np.save(path, arr)


def generate_crosstalk_slm_patterns(cfg: AppConfig, force: bool = False) -> dict[str, Path]:
    """Fixed SLM images for calibration (not used in ranking)."""
    layout = PathLayout(cfg.root)
    cal = layout.patterns_calibration
    ensure_dir(cal)
    slm = cfg.slm
    paths: dict[str, Path] = {}

    # Full black canvas
    p = cal / "slm_full_black.png"
    _save_png(p, blank_canvas(slm, fill=slm.gray_off), force)
    paths["full_black"] = p

    # Full white active region (optional bright-field)
    active_white = np.full((slm.active_height, slm.active_width), slm.gray_on, dtype=np.uint8)
    p = cal / "slm_active_white.png"
    _save_png(p, place_active_region(slm, active_white), force)
    paths["active_white"] = p

    # Fixed checker at m=8 for green-only / both-on crosstalk (locked choice)
    s = 8
    gy = slm.active_height // s
    gx = slm.active_width // s
    yy, xx = np.indices((gy, gx))
    checker = ((yy + xx) % 2).astype(np.uint8)
    expanded = np.repeat(np.repeat(checker, s, axis=0), s, axis=1)
    gray = np.where(expanded == 1, slm.gray_on, slm.gray_off).astype(np.uint8)
    p = cal / "slm_crosstalk_fixed_checker_mp008.png"
    _save_png(p, place_active_region(slm, gray), force)
    paths["crosstalk_fixed"] = p

    meta = {
        "purpose": "Calibration SLM images for dark-field and crosstalk only",
        "crosstalk_fixed_rule": "Always use slm_crosstalk_fixed_checker_mp008.png for green_only and both_on",
        "full_black_rule": "Use slm_full_black.png for dark and red_only",
        "do_not_sweep_C01_C08_during_crosstalk": True,
    }
    meta_path = cal / "calibration_slm_readme.json"
    assert_writable(meta_path, force=force)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return paths


def generate_screening_challenges(
    cfg: AppConfig,
    challenge_ids: list[str] | None = None,
    max_abs_z: float = DEFAULT_MAX_ABS_Z,
    force: bool = False,
) -> Path:
    """Generate all screening challenges and write challenge_manifest.csv."""
    challenge_ids = challenge_ids or list(DEFAULT_CHALLENGE_IDS)
    layout = PathLayout(cfg.root)
    ensure_dir(layout.patterns)
    ensure_dir(layout.manifests)
    ensure_dir(layout.metadata)

    if cfg.slm.active_width != cfg.slm.active_height:
        raise ValueError("Screening generator assumes square active region")

    manifest_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []

    for m in cfg.experiment.macro_pixel_sizes:
        grid = challenge_grid_size(cfg.slm.active_width, m)
        seed = int(cfg.experiment.global_seed) * 1000 + int(m) * 97
        rng = np.random.default_rng(seed)
        binaries, pair_stats = generate_independent_set(
            grid=grid,
            n_challenges=len(challenge_ids),
            rng=rng,
            max_abs_z=max_abs_z,
        )
        logger.info(
            "macro=%d grid=%d seed=%d n=%d pair_checks=%d",
            m,
            grid,
            seed,
            len(binaries),
            len(pair_stats),
        )

        hds = [p[2] for p in pair_stats] if pair_stats else [0.5]
        zs = [abs(p[3]) for p in pair_stats] if pair_stats else [0.0]
        min_hd = float(min(hds))
        max_abs_z_obs = float(max(zs)) if zs else 0.0

        for idx, (cid, binary) in enumerate(zip(challenge_ids, binaries)):
            out_dir = layout.patterns / mp_tag(m) / cid
            ensure_dir(out_dir)
            expanded_bin = expand_binary_nearest(binary, m)
            expanded = map_binary_to_gray(expanded_bin, cfg.slm.gray_off, cfg.slm.gray_on)
            full = place_active_region(cfg.slm, expanded)

            stem = f"{mp_tag(m)}_{cid}"
            binary_npy = out_dir / "binary.npy"
            binary_png = out_dir / "binary.png"
            expanded_png = out_dir / "expanded_512.png"
            full_png = out_dir / f"{stem}.png"
            full_npy = out_dir / f"{stem}.npy"
            alias_png = layout.patterns / mp_tag(m) / f"{stem}.png"

            _save_npy(binary_npy, binary.astype(np.uint8), force)
            _save_png(binary_png, (binary * 255).astype(np.uint8), force)
            _save_png(expanded_png, expanded, force)
            _save_npy(full_npy, full, force)
            _save_png(full_png, full, force)
            _save_png(alias_png, full, force)

            meta = {
                "macro_pixel": m,
                "challenge_id": cid,
                "challenge_index": idx + 1,
                "challenge_grid_size": grid,
                "seed_macro": seed,
                "global_seed": cfg.experiment.global_seed,
                "white_fraction": duty_cycle(binary),
                "gray_off": cfg.slm.gray_off,
                "gray_on": cfg.slm.gray_on,
                "canvas_width": cfg.slm.canvas_width,
                "canvas_height": cfg.slm.canvas_height,
                "active_width": cfg.slm.active_width,
                "active_height": cfg.slm.active_height,
                "center_x": cfg.slm.center_x,
                "center_y": cfg.slm.center_y,
                "x_start": cfg.slm.x_start,
                "x_end": cfg.slm.x_end,
                "y_start": cfg.slm.y_start,
                "y_end": cfg.slm.y_end,
                "sha256_binary": sha256_array(binary.astype(np.uint8)),
                "sha256_full": sha256_array(full),
                "sha256_full_png": sha256_file(full_png),
                "max_abs_z_constraint": max_abs_z,
                "set_min_pairwise_hd": min_hd,
                "set_max_abs_z": max_abs_z_obs,
            }
            meta_path = out_dir / "meta.json"
            assert_writable(meta_path, force=force)
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            manifest_rows.append(
                {
                    "macro_pixel": m,
                    "challenge_id": cid,
                    "grid_size": grid,
                    "random_seed": seed,
                    "white_fraction": meta["white_fraction"],
                    "minimum_pairwise_hd": min_hd,
                    "maximum_abs_z": max_abs_z_obs,
                    "sha256": meta["sha256_full_png"],
                    "pattern_path": str(full_png.relative_to(cfg.root)).replace("\\", "/"),
                    "alias_path": str(alias_png.relative_to(cfg.root)).replace("\\", "/"),
                }
            )

        for i, j, hd, z in pair_stats:
            pair_rows.append(
                {
                    "macro_pixel": m,
                    "challenge_i": challenge_ids[i],
                    "challenge_j": challenge_ids[j],
                    "hd": hd,
                    "z": z,
                    "abs_z": abs(z),
                }
            )

    manifest_path = layout.manifests / "challenge_manifest.csv"
    assert_writable(manifest_path, force=force)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    pair_path = layout.manifests / "challenge_pairwise_hd.csv"
    assert_writable(pair_path, force=force)
    if pair_rows:
        with pair_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0].keys()))
            writer.writeheader()
            writer.writerows(pair_rows)

    logger.info("Wrote %s (%d rows)", manifest_path, len(manifest_rows))
    return manifest_path


def write_randomized_acquisition_order(
    cfg: AppConfig,
    challenge_ids: list[str] | None = None,
    control_every: int = 10,
    force: bool = False,
) -> Path:
    """Write a randomized 56-task acquisition table with periodic control slots."""
    challenge_ids = challenge_ids or list(DEFAULT_CHALLENGE_IDS)
    layout = PathLayout(cfg.root)
    ensure_dir(layout.manifests)

    tasks: list[tuple[int, str]] = [
        (m, cid) for m in cfg.experiment.macro_pixel_sizes for cid in challenge_ids
    ]
    rng = np.random.default_rng(int(cfg.experiment.global_seed) + 777)
    order = list(range(len(tasks)))
    rng.shuffle(order)

    rows: list[dict[str, object]] = []
    order_index = 0
    control_idx = 0
    for k, task_i in enumerate(order, start=1):
        if control_every > 0 and k > 1 and (k - 1) % control_every == 0:
            order_index += 1
            control_idx += 1
            rows.append(
                {
                    "order_index": order_index,
                    "macro_pixel": "",
                    "challenge_id": f"CTRL_{control_idx:02d}",
                    "is_control": 1,
                    "pattern_path": "patterns/calibration/slm_crosstalk_fixed_checker_mp008.png",
                    "planned_video_path": f"videos/control/CTRL_{control_idx:02d}.mp4",
                    "notes": "Drift control; not used for macro ranking",
                }
            )
        m, cid = tasks[task_i]
        order_index += 1
        stem = f"{mp_tag(m)}_{cid}"
        rows.append(
            {
                "order_index": order_index,
                "macro_pixel": m,
                "challenge_id": cid,
                "is_control": 0,
                "pattern_path": f"patterns/{mp_tag(m)}/{cid}/{stem}.png",
                "planned_video_path": f"videos/{mp_tag(m)}/{cid}.mp4",
                "notes": "",
            }
        )

    out = layout.manifests / "acquisition_order.csv"
    assert_writable(out, force=force)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote randomized acquisition order (%d rows): %s", len(rows), out)
    return out
