#!/usr/bin/env python3
"""Verify a local copy of the Zenodo dataset (DOI 10.5281/zenodo.22267156) before reproduction.

Checks, without decoding any video:

1. presence of the top-level files and archives (raw archives are reported, not required);
2. the challenge libraries: exact 50% occupancy, C001-C008 identical to the eight-challenge
   bank C01-C08 and to the canvases shipped in ``data/challenge_patterns/mp002``, deterministic
   regeneration of C01-C08 (global seed 20260711) and C009-C128 (master seed 20260719 with
   rejection sampling), pairwise acceptance criteria |HD z| <= 3.5 and |input NCC| <= 0.02;
3. the calibration dark reference and the four valid-pixel masks against ``MASKS.json``;
4. the metadata tables and the data-quality exclusion manifest (64 retained S0 reuses,
   16 excluded S1 copies, one non-independent formal Round A/B pair);
5. the principal scientific-authority values stored in the frozen tables.

``--checksums`` additionally verifies the SHA-256 of the lightweight files listed in
``final_release_checksums.sha256``; ``--checksums-all`` also hashes the raw archives (slow).
Nothing is downloaded and nothing is written into the dataset directory.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image

from e01.patterns.screening import DEFAULT_MAX_ABS_Z, generate_independent_set
from experiment4_security.challenges.generator import generated_grid
from puf_common.checks import Check, check, report, reproduction_parser
from puf_common.ncc import zero_mean_ncc
from puf_common.public_data import LIGHTWEIGHT_ARCHIVES, RAW_ARCHIVES, ROOT_FILES, PublicDataset, resolve_data_root

REPO = Path(__file__).resolve().parents[1]
CH = "challenges_calibration_masks/challenges/"
MASKS = "challenges_calibration_masks/masks/"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inventory_checks(ds: PublicDataset) -> list[Check]:
    present = ds.inventory()
    checks = [check(f"present: {n}", present[n], True) for n in list(LIGHTWEIGHT_ARCHIVES) + list(ROOT_FILES)]
    n_raw = sum(present[n] for n in RAW_ARCHIVES)
    print(f"raw archives available locally: {n_raw}/{len(RAW_ARCHIVES)} (only needed for raw mode)")
    return checks


def challenge_checks(ds: PublicDataset) -> list[Check]:
    z = ds.read_npz(CH + "challenges_exact.npz")
    grids = (z["binary_256_128"] > 0).astype(np.uint8)
    eight = (z["binary_256_8"] > 0).astype(np.uint8)
    canvases = z["canvases_128"]
    manifest = ds.read_csv(CH + "challenge_manifest.csv")
    r0 = manifest.iloc[0]
    oy, ox, ah, aw, m = (int(r0[k]) for k in ("active_offset_y_px", "active_offset_x_px", "active_height_px", "active_width_px", "macro_pixel_size"))
    from_canvas = (canvases[:, oy : oy + ah, ox : ox + aw][:, ::m, ::m] > 127).astype(np.uint8)

    local = np.stack([(np.asarray(Image.open(REPO / f"data/challenge_patterns/mp002/mp002_C{i:02d}.png").convert("L"))) for i in range(1, 9)])
    local_grids = (local[:, oy : oy + ah, ox : ox + aw][:, ::m, ::m] > 127).astype(np.uint8)

    seed8 = int(z["global_seed_8"]) * 1000 + 2 * 97
    regen8, _ = generate_independent_set(grid=256, n_challenges=8, rng=np.random.default_rng(seed8), max_abs_z=DEFAULT_MAX_ABS_Z)
    master = int(z["master_seed_128"])
    regen128 = 0
    for i in range(9, 129):
        if any(np.array_equal(generated_grid(master, i, a, 256, 256), grids[i - 1]) for a in range(10)):
            regen128 += 1

    flat = grids.reshape(128, -1).astype(np.int16)
    hd = (flat[:, None, :] != flat[None, :, :]).mean(-1)
    iu = np.triu_indices(128, 1)
    hd_z = np.abs((hd[iu] - 0.5) / np.sqrt(0.25 / flat.shape[1]))
    ncc = np.array([zero_mean_ncc(flat[i].astype(float), flat[j].astype(float)) for i, j in zip(*iu)])
    frozen_hd = ds.read_csv(CH + "pairwise_hamming_distance.csv")
    frozen_ncc = ds.read_csv(CH + "pairwise_input_ncc.csv")
    return [
        check("128-challenge bank: exact 50% open cells for every challenge", bool(np.all(grids.reshape(128, -1).mean(1) == 0.5)), True),
        check("128-challenge bank: canvases consistent with binary grids", bool(np.array_equal(from_canvas, grids)), True),
        check("C001-C008 identical to the eight-challenge bank C01-C08", bool(np.array_equal(grids[:8], eight)), True),
        check("C001-C008 identical to data/challenge_patterns/mp002 canvases", bool(np.array_equal(local_grids, grids[:8])) and bool(np.array_equal(local, canvases[:8])), True),
        check(f"C01-C08 regenerate from global seed {int(z['global_seed_8'])} (m = 2 seed {seed8})", bool(all(np.array_equal(a, b) for a, b in zip(regen8, eight))), True),
        check(f"C009-C128 regenerate from master seed {master} (rejection sampling)", regen128, 120, 0),
        check("pairwise |HD z-score| maximum <= 3.5", float(hd_z.max()) <= 3.5, True),
        check("pairwise |input NCC| maximum <= 0.02", float(np.abs(ncc).max()) <= 0.02, True),
        check("pairwise HD table equals frozen pairwise_hamming_distance.csv", float(np.abs(hd[iu] - frozen_hd.normalized_hamming.to_numpy()).max()), 0.0, 1e-12),
        check("pairwise NCC table equals frozen pairwise_input_ncc.csv", float(np.abs(ncc - frozen_ncc.zero_mean_input_ncc.to_numpy()).max()), 0.0, 1e-9),
        check("manifest sha256 of C001 equals shipped mp002_C01.png", str(manifest.loc[0, "sha256"]), sha256((REPO / "data/challenge_patterns/mp002/mp002_C01.png").read_bytes())),
    ]


def calibration_checks(ds: PublicDataset) -> list[Check]:
    checks = []
    dark_meta = ds.read_json("challenges_calibration_masks/calibration/dark_reference.json")
    dark_bytes = ds.read_bytes("challenges_calibration_masks/calibration/dark_reference_2048x1536.npz")
    dark = ds.read_npz("challenges_calibration_masks/calibration/dark_reference_2048x1536.npz")
    checks.append(check("dark reference sha256", sha256(dark_bytes), dark_meta["sha256"]))
    checks.append(check("dark reference shape (1536, 2048)", list(dark["dark_green"].shape), [1536, 2048]))
    for entry in ds.read_json(MASKS + "MASKS.json"):
        raw = ds.read_bytes("challenges_calibration_masks/" + entry["file"])
        arr = np.asarray(Image.open(io.BytesIO(raw))) if entry["file"].endswith(".png") else next(iter(np.load(io.BytesIO(raw)).values()))
        checks.append(check(f"{entry['id']}: sha256", sha256(raw), entry["sha256_file"]))
        checks.append(check(f"{entry['id']}: valid pixels", int(np.count_nonzero(arr)), int(entry["valid_pixel_count"]), 0))
    return checks


def metadata_checks(ds: PublicDataset) -> list[Check]:
    ex = ds.read_csv("data_quality_exclusions.csv")
    e2 = ds.read_csv("analysis_ready_data/metadata/experiment2_metadata.csv")
    e3 = ds.read_csv("analysis_ready_data/metadata/experiment3_metadata.csv")
    raw = ds.read_csv("raw_file_manifest.csv")
    return [
        check("fixed-state metadata rows (15 x (8 green + 2 red))", len(e2), 150, 0),
        check("threshold-development metadata rows", len(e3), 765, 0),
        check("raw file manifest entries", len(raw), 22147, 0),
        check("exclusion manifest: intentional S0 reuses retained", int((ex.classification == "INTENTIONAL_REUSE_RETAINED").sum()), 64, 0),
        check("exclusion manifest: invalid S1 copies excluded", int((ex.classification == "INVALID_COPY_NO_REPLACEMENT").sum()), 16, 0),
        check("exclusion manifest: non-independent formal A/B pair (F02/M1/C099)", int((ex.classification == "FILE_COPY").sum()), 1, 0),
        check("exclusion manifest: raw files retained for provenance", bool((ex.raw_files_retained_for_provenance == "YES").all()), True),
    ]


def authority_checks(ds: PublicDataset) -> list[Check]:
    fig4 = ds.read_csv("Source_Data/Fig4/Fig4c_pair_NCC_scores.csv")
    nine = fig4[fig4.L_cm == 9]
    g, c, d = (nine.loc[nine.score_class == k, "NCC"].to_numpy() for k in ("S_intra", "S_inter_c", "S_inter_d"))
    boot = ds.read_csv("Source_Data/Fig4/Fig4f_bootstrap_selected_length.csv")
    lifecycle = ds.read_json("analysis_ready_data/fiber_id_9d/lifecycle/threshold_development.json")
    track_a = ds.read_csv("analysis_ready_data/formal/track_a_database_authentication_summary.csv")
    fl = ds.read_csv("analysis_ready_data/fiber_length/bootstrap_selection.csv")
    checks = [
        check("Fig. 4: G_min(9 cm) from Source Data pair scores", float(min(np.quantile(g, 0.05) - np.quantile(c, 0.95), np.quantile(g, 0.05) - np.quantile(d, 0.95))), 0.3897924079274487),
        check("Fig. 4: bootstrap replicates selecting 9 cm", int((boot.selected_L_cm == 9).sum()), 4991, 0),
        check("Fig. 4: bootstrap replicates total", len(boot), 5000, 0),
        check("Supplementary Note 7.1: T_G frozen", float(lifecycle["tau_G_frozen"]), 0.12899641700197656, 0),
        check("Supplementary Note 7.1: n_req frozen", int(lifecycle["n_req_frozen"]), 7, 0),
        check("Fig. 7: device-state units", len(track_a), 80, 0),
        check("Fig. 7: Valid units", int(track_a.state_conclusion.str.endswith("VALID").sum()), 77, 0),
        check("Fig. 7: Partial units", int(track_a.state_conclusion.str.endswith("PARTIAL").sum()), 3, 0),
    ]
    # Known dataset caveat: the analysis-ready fiber_length/ folder is the frozen pre-v6 run.
    n9 = int(fl.loc[fl.length_cm == 9, "n_selected"].item()) if "n_selected" in fl else -1
    if n9 != 4991:
        print(f"note: analysis_ready_data/fiber_length/ holds the earlier frozen fiber-length run ({n9}/5000 for 9 cm); the Fig. 4 authority is Source_Data/Fig4 (4991/5000).")
    return checks


def checksum_checks(ds: PublicDataset, include_raw: bool) -> list[Check]:
    checks = []
    for line in ds.read_text("final_release_checksums.sha256").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.strip().lstrip("*")
        if name.startswith("raw_") and not include_raw:
            continue
        path = ds.root / name
        if not path.is_file():
            checks.append(check(f"sha256 {name}", "missing", digest))
            continue
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        checks.append(check(f"sha256 {name}", h.hexdigest(), digest))
    return checks


def main() -> int:
    p = reproduction_parser(__doc__.splitlines()[0])
    p.add_argument("--checksums", action="store_true", help="Verify SHA-256 of the lightweight files")
    p.add_argument("--checksums-all", action="store_true", help="Verify SHA-256 of every file including the raw archives (~19 GiB)")
    args = p.parse_args()
    ds = PublicDataset(resolve_data_root(args.data_root))
    out = Path(args.output_root).expanduser().resolve() / "verify_public_data"

    checks = inventory_checks(ds)
    lightweight_ok = all(c.passed for c in checks)
    if lightweight_ok:
        checks += challenge_checks(ds) + calibration_checks(ds) + metadata_checks(ds) + authority_checks(ds)
    if args.checksums or args.checksums_all:
        checks += checksum_checks(ds, include_raw=args.checksums_all)
    return 0 if report(checks, out, f"Public dataset verification ({ds.root})") else 1


if __name__ == "__main__":
    raise SystemExit(main())
