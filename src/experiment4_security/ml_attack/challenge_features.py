"""Challenge-side (input) feature construction.

All features here depend ONLY on the challenge pattern itself (its binary
macro-pixel grid), never on any measured response. This module also
contains a read-only-safe repair path for the case where the challenge
library's raster `patterns/*.png` files are missing from disk but the
manifest (with per-challenge generation seed) is intact: it regenerates them
byte-for-byte via the existing, unmodified
`experiment4_security.challenges.generator.generate_library` recipe into a
cache directory OUTSIDE `data/challenges/m2_128_v1/`, and verifies every
regenerated file's sha256 against the checked-in `challenge_manifest.json`
before trusting it. `data/challenges/m2_128_v1/` itself is never written to.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from experiment4_security.challenges.generator import generate_library, grid_from_canvas
from experiment4_security.common.config import ChallengeLibraryConfig


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_challenge_manifest(challenge_root: Path) -> pd.DataFrame:
    """`challenge_root` is expected to be the `.../m2_128_v1/patterns` dir."""
    lib_root = Path(challenge_root).parent
    manifest_csv = lib_root / "challenge_manifest.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Challenge manifest not found: {manifest_csv}")
    return pd.read_csv(manifest_csv)


def ensure_patterns_available(
    manifest: pd.DataFrame,
    *,
    challenge_root: Path,
    pattern_cache_dir: Path,
    challenge_library_yaml: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Return {challenge_id: resolved_png_path}, regenerating into a cache
    directory (never into `challenge_root` itself) if any raster file is
    missing. Every path returned (whether original or cache) has a verified
    sha256 matching the manifest.
    """
    challenge_root = Path(challenge_root)
    resolved: dict[str, Path] = {}
    missing_ids: list[str] = []
    for row in manifest.itertuples(index=False):
        p = challenge_root / f"{row.challenge_id}.png"
        if p.exists() and _sha256_file(p) == row.sha256:
            resolved[row.challenge_id] = p
        else:
            missing_ids.append(row.challenge_id)

    audit: dict[str, Any] = {
        "n_challenges": len(manifest),
        "n_found_in_challenge_root": len(manifest) - len(missing_ids),
        "n_missing_or_hash_mismatch": len(missing_ids),
        "regenerated": False,
    }

    if not missing_ids:
        return {"paths": resolved, "audit": audit}

    pattern_cache_dir = Path(pattern_cache_dir)
    exact_npz = challenge_root.parent / "challenges_exact.npz"
    if exact_npz.exists():
        return _materialize_from_exact_archive(manifest, exact_npz, pattern_cache_dir, resolved, missing_ids, audit)
    cache_manifest_csv = pattern_cache_dir / "challenge_manifest.csv"
    need_regenerate = True
    if cache_manifest_csv.exists():
        all_cached_ok = True
        for row in manifest.itertuples(index=False):
            p = pattern_cache_dir / "patterns" / f"{row.challenge_id}.png"
            if not p.exists() or _sha256_file(p) != row.sha256:
                all_cached_ok = False
                break
        need_regenerate = not all_cached_ok

    if need_regenerate:
        base_cfg = ChallengeLibraryConfig.from_yaml(challenge_library_yaml)
        cfg = dataclasses.replace(base_cfg, output_root=pattern_cache_dir, overwrite=True)
        generate_library(cfg)

    verified_mismatches: list[str] = []
    for cid in missing_ids:
        row = manifest[manifest["challenge_id"] == cid].iloc[0]
        p = pattern_cache_dir / "patterns" / f"{cid}.png"
        if not p.exists():
            verified_mismatches.append(f"{cid}: not produced by regeneration")
            continue
        actual = _sha256_file(p)
        if actual != row["sha256"]:
            verified_mismatches.append(f"{cid}: sha256 mismatch (expected {row['sha256']}, got {actual})")
            continue
        resolved[cid] = p

    audit["regenerated"] = True
    audit["regeneration_cache_dir"] = str(pattern_cache_dir)
    audit["regeneration_verification_mismatches"] = verified_mismatches
    audit["regeneration_note"] = (
        "patterns/*.png were missing (or hash-mismatched) under challenge_root "
        "on disk in this workspace. They were regenerated via the unmodified "
        "formal.challenges.generator.generate_library recipe "
        "into a cache directory OUTSIDE data/challenges/m2_128_v1/, and every "
        "regenerated file's sha256 was verified against the checked-in "
        "challenge_manifest.json before use. data/challenges/m2_128_v1/ was "
        "never written to."
    )
    if verified_mismatches:
        raise RuntimeError(
            "Challenge pattern regeneration did not reproduce byte-identical "
            f"files for: {verified_mismatches}. Refusing to proceed with "
            "unverified challenge patterns."
        )
    return {"paths": resolved, "audit": audit}


def _materialize_from_exact_archive(
    manifest: pd.DataFrame,
    exact_npz: Path,
    pattern_cache_dir: Path,
    resolved: dict[str, Path],
    missing_ids: list[str],
    audit: dict[str, Any],
) -> dict[str, Any]:
    """Write missing challenge PNGs from the public ``challenges_exact.npz`` archive.

    The public dataset ships the exact uint8 SLM canvases (``canvases_128``) and the
    exact 256x256 binary grids (``binary_256_128``) instead of the raster files. Each
    materialized canvas is verified against the archived binary grid before use.
    """
    with np.load(exact_npz) as z:
        ids = [str(x) for x in z["ids_128"]]
        canvases = z["canvases_128"]
        grids = z["binary_256_128"]
    index = {cid: i for i, cid in enumerate(ids)}
    mismatches: list[str] = []
    for cid in missing_ids:
        row = manifest[manifest["challenge_id"] == cid].iloc[0]
        if cid not in index:
            mismatches.append(f"{cid}: not present in {exact_npz.name}")
            continue
        canvas = canvases[index[cid]]
        grid = grid_from_canvas(
            canvas,
            ox=int(row["active_offset_x_px"]),
            oy=int(row["active_offset_y_px"]),
            aw=int(row["active_width_px"]),
            ah=int(row["active_height_px"]),
            macro=int(row["macro_pixel_size"]),
            open_level=255,
        )
        if not np.array_equal(grid.astype(np.uint8), grids[index[cid]].astype(np.uint8)):
            mismatches.append(f"{cid}: canvas/grid disagreement inside {exact_npz.name}")
            continue
        dest = pattern_cache_dir / "patterns" / f"{cid}.png"
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(canvas.astype(np.uint8), mode="L").save(dest, format="PNG", compress_level=0)
        resolved[cid] = dest
    audit["regenerated"] = True
    audit["regeneration_cache_dir"] = str(pattern_cache_dir)
    audit["regeneration_source"] = str(exact_npz)
    audit["regeneration_verification"] = "exact binary grid equality (PNG bytes are not hash-compared)"
    audit["regeneration_verification_mismatches"] = mismatches
    if mismatches:
        raise RuntimeError(f"Exact challenge archive verification failed: {mismatches}")
    return {"paths": resolved, "audit": audit}


def binary_grid_from_png(png_path: Path, manifest_row: pd.Series, *, open_level: int) -> np.ndarray:
    canvas = np.asarray(Image.open(png_path).convert("L"))
    expected_shape = (int(manifest_row["canvas_height_px"]), int(manifest_row["canvas_width_px"]))
    if canvas.shape != expected_shape:
        raise ValueError(f"{png_path}: canvas shape {canvas.shape} != manifest {expected_shape}")
    grid = grid_from_canvas(
        canvas,
        ox=int(manifest_row["active_offset_x_px"]),
        oy=int(manifest_row["active_offset_y_px"]),
        aw=int(manifest_row["active_width_px"]),
        ah=int(manifest_row["active_height_px"]),
        macro=int(manifest_row["macro_pixel_size"]),
        open_level=open_level,
    )
    return grid.astype(np.uint8)


def downsample_bitmap(grid: np.ndarray, *, size: int) -> np.ndarray:
    """Average-pool a binary macro-cell grid down to `size x size`."""
    h, w = grid.shape
    if h % size != 0 or w % size != 0:
        raise ValueError(f"grid shape {grid.shape} not divisible by target size {size}")
    bh, bw = h // size, w // size
    reshaped = grid.reshape(size, bh, size, bw).astype(np.float64)
    return reshaped.mean(axis=(1, 3))


@dataclasses.dataclass
class ChallengeFeatureSet:
    challenge_id: str
    source_path: str
    sha256: str
    full_binary_vector: np.ndarray  # flattened active-region macro grid, {0,1}
    bitmap_32: np.ndarray  # 32x32 average-pooled, flattened to 1024-dim


def build_challenge_features(
    challenge_ids: list[str],
    *,
    manifest: pd.DataFrame,
    resolved_paths: dict[str, Path],
    bitmap_size: int = 32,
    open_level: int = 255,
) -> dict[str, ChallengeFeatureSet]:
    out: dict[str, ChallengeFeatureSet] = {}
    manifest_idx = manifest.set_index("challenge_id")
    for cid in challenge_ids:
        if cid not in resolved_paths:
            continue
        row = manifest_idx.loc[cid]
        path = resolved_paths[cid]
        grid = binary_grid_from_png(path, row, open_level=open_level)
        bitmap = downsample_bitmap(grid, size=bitmap_size)
        out[cid] = ChallengeFeatureSet(
            challenge_id=cid,
            source_path=str(path),
            sha256=str(row["sha256"]),
            full_binary_vector=grid.astype(np.float32).ravel(),
            bitmap_32=bitmap.astype(np.float32).ravel(),
        )
    return out


def write_feature_manifest(features: dict[str, ChallengeFeatureSet], out_csv: Path) -> pd.DataFrame:
    rows = []
    for cid, fs in features.items():
        rows.append(
            {
                "challenge_id": cid,
                "source_path": fs.source_path,
                "sha256": fs.sha256,
                "feature_type": "full_binary_vector",
                "feature_dimension": int(fs.full_binary_vector.size),
            }
        )
        rows.append(
            {
                "challenge_id": cid,
                "source_path": fs.source_path,
                "sha256": fs.sha256,
                "feature_type": "bitmap_32x32",
                "feature_dimension": int(fs.bitmap_32.size),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df
