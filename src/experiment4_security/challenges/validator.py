"""Input-space validation for the m=2 challenge library."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from experiment4_security.common.progress_util import stage_tqdm
from puf_common.ncc import zero_mean_ncc


def _load_grid(
    path: Path,
    *,
    ox: int,
    oy: int,
    aw: int,
    ah: int,
    macro: int,
    open_level: int,
) -> np.ndarray:
    canvas = np.asarray(Image.open(path).convert("L"))
    active = canvas[oy : oy + ah, ox : ox + aw]
    return (active[::macro, ::macro] >= (open_level // 2)).astype(np.uint8)


def validate_library(
    root: Path,
    *,
    grids: list[np.ndarray] | None = None,
    max_abs_hd_z: float = 3.5,
    max_abs_input_ncc: float = 0.02,
) -> dict[str, Any]:
    manifest = pd.read_csv(root / "challenge_manifest.csv")
    if grids is None:
        grids = []
        for _, row in stage_tqdm(
            list(manifest.iterrows()), desc="Load grids", unit="pat"
        ):
            p = root / row["pattern_path"]
            grids.append(
                _load_grid(
                    p,
                    ox=int(row["active_offset_x_px"]),
                    oy=int(row["active_offset_y_px"]),
                    aw=int(row["active_width_px"]),
                    ah=int(row["active_height_px"]),
                    macro=int(row["macro_pixel_size"]),
                    open_level=255,
                )
            )

    n = len(grids)
    n_cells = int(grids[0].size)
    failures: list[str] = []

    for i, g in enumerate(grids):
        frac = float(g.mean())
        if abs(frac - 0.5) > 1e-12:
            failures.append(f"{manifest.iloc[i]['challenge_id']} open_fraction={frac}")

    # Duplicates / complements
    for i in range(n):
        for j in range(i + 1, n):
            if np.array_equal(grids[i], grids[j]):
                failures.append(
                    f"duplicate {manifest.iloc[i]['challenge_id']}-{manifest.iloc[j]['challenge_id']}"
                )
            if np.array_equal(grids[i], 1 - grids[j]):
                failures.append(
                    f"complement {manifest.iloc[i]['challenge_id']}-{manifest.iloc[j]['challenge_id']}"
                )

    hd_rows = []
    ncc_rows = []
    max_abs_z = 0.0
    max_abs_ncc = 0.0
    denom = np.sqrt(0.25 / n_cells)

    for i in stage_tqdm(range(n), desc="Pairwise input", unit="i"):
        gi = grids[i].astype(np.float64)
        for j in range(i + 1, n):
            gj = grids[j].astype(np.float64)
            hd = float(np.mean(grids[i] != grids[j]))
            z = (hd - 0.5) / denom
            ncc = float(zero_mean_ncc(gi, gj))
            max_abs_z = max(max_abs_z, abs(z))
            max_abs_ncc = max(max_abs_ncc, abs(ncc))
            a = manifest.iloc[i]["challenge_id"]
            b = manifest.iloc[j]["challenge_id"]
            hd_rows.append(
                {
                    "challenge_id_a": a,
                    "challenge_id_b": b,
                    "normalized_hamming": hd,
                    "hd_z": z,
                }
            )
            ncc_rows.append(
                {
                    "challenge_id_a": a,
                    "challenge_id_b": b,
                    "zero_mean_input_ncc": ncc,
                }
            )
            if abs(z) > max_abs_hd_z:
                failures.append(f"|z|={abs(z):.3f} > {max_abs_hd_z} for {a}-{b}")
            if abs(ncc) > max_abs_input_ncc:
                failures.append(f"|NCC|={abs(ncc):.4f} > {max_abs_input_ncc} for {a}-{b}")

    pd.DataFrame(hd_rows).to_csv(root / "pairwise_hamming_distance.csv", index=False)
    pd.DataFrame(ncc_rows).to_csv(root / "pairwise_input_ncc.csv", index=False)

    # Deduplicate failure messages (pairwise can be long)
    failures = sorted(set(failures))
    report = {
        "passed": len(failures) == 0,
        "n_challenges": n,
        "n_macro_cells": n_cells,
        "max_abs_hd_z": max_abs_z,
        "max_abs_input_ncc": max_abs_ncc,
        "thresholds": {
            "max_abs_hd_z": max_abs_hd_z,
            "max_abs_input_ncc": max_abs_input_ncc,
        },
        "n_failures": len(failures),
        "failures": failures[:50],
    }
    return report


def validate_existing(root: Path) -> dict[str, Any]:
    return validate_library(Path(root))
