#!/usr/bin/env python3
"""Stage the Zenodo dataset for raw-video mode.

Nothing is written into the dataset directory. The script extracts, under the repository's
git-ignored ``data/`` folder,

* the lightweight archives into ``data/zenodo/<archive-name>/`` (challenge library,
  calibration dark reference, valid masks, Source Data, analysis-ready tables) and copies
  the top-level CSV manifests there;
* the requested raw recording archives into ``data/raw/<experiment>/`` using the layout
  expected by the corresponding config in ``configs/``;
* for the fixed-state and threshold-development datasets, a local copy of the public
  metadata CSV whose ``video_path`` column points at the extracted files.

Examples
--------
    python scripts/prepare_raw_workspace.py --data-root /path/to/zenodo --lightweight-only
    python scripts/prepare_raw_workspace.py --data-root /path/to/zenodo --experiment fiber_length
    python scripts/prepare_raw_workspace.py --data-root /path/to/zenodo --experiment formal --device F01 --device F02
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import pandas as pd

from puf_common.public_data import LIGHTWEIGHT_ARCHIVES, ROOT_FILES, resolve_data_root

REPO = Path(__file__).resolve().parents[1]

# experiment -> (raw archives, metadata CSV inside analysis_ready_data.zip or None, extraction subdirectory)
RAW_EXPERIMENTS = {
    "macro_pixel": (["raw_macro_pixel.zip"], None, "videos"),
    "fiber_length": (["raw_fiber_length.zip"], None, ""),
    "fixed_state": (["raw_fixed_state.zip"], "metadata/experiment2_metadata.csv", ""),
    "wavelength_pathway": (["raw_wavelength_pathway.zip"], None, ""),
    "threshold_development": (["raw_threshold_development.zip"], "metadata/experiment3_metadata.csv", ""),
    "formal": ([f"raw_formal_F{i:02d}.zip" for i in range(1, 11)], None, ""),
}


def extract(zip_path: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if not m.endswith("/")]
        todo = [m for m in members if not (dest / m).exists()]
        if todo:
            zf.extractall(dest, members=todo)
    print(f"  {zip_path.name}: {len(members)} files -> {dest} ({len(todo)} newly extracted)")
    return len(members)


def stage_lightweight(root: Path, zenodo_dir: Path) -> None:
    print("Lightweight archives:")
    for name in LIGHTWEIGHT_ARCHIVES:
        src = root / name
        if src.is_file():
            # Source_Data.zip already prefixes members with Source_Data/; the other two
            # archives are prefix-free and are extracted into a named subdirectory.
            dest = zenodo_dir if LIGHTWEIGHT_ARCHIVES[name] else zenodo_dir / name[: -len(".zip")]
            extract(src, dest)
        elif (root / name[: -len(".zip")]).is_dir():
            print(f"  {name}: already extracted in the dataset directory")
        else:
            print(f"  {name}: MISSING")
    for name in ROOT_FILES:
        if (root / name).is_file():
            shutil.copy2(root / name, zenodo_dir / name)


def write_local_metadata(zenodo_dir: Path, rel_csv: str, raw_dir: Path) -> None:
    meta = pd.read_csv(zenodo_dir / "analysis_ready_data" / rel_csv)
    meta["video_path"] = [str((raw_dir / p).resolve()) for p in meta["internal_path"]]
    missing = [p for p in meta["video_path"] if not Path(p).is_file()]
    out = raw_dir / Path(rel_csv).name
    meta.to_csv(out, index=False)
    print(f"  wrote {out} ({len(meta)} recordings, {len(missing)} missing on disk)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", default=None, help="Zenodo download directory (or $PUF_DATA_ROOT)")
    p.add_argument("--experiment", action="append", choices=sorted(RAW_EXPERIMENTS), default=None, help="Raw archive group to extract (repeatable)")
    p.add_argument("--device", action="append", default=None, help="Restrict --experiment formal to the given devices (repeatable)")
    p.add_argument("--lightweight-only", action="store_true", help="Only stage the lightweight archives")
    p.add_argument("--workspace", default=str(REPO / "data"), help="Destination (default: <repo>/data)")
    args = p.parse_args()

    root = resolve_data_root(args.data_root)
    workspace = Path(args.workspace).expanduser().resolve()
    zenodo_dir, raw_root = workspace / "zenodo", workspace / "raw"
    zenodo_dir.mkdir(parents=True, exist_ok=True)
    stage_lightweight(root, zenodo_dir)
    if args.lightweight_only or not args.experiment:
        return 0

    for exp in args.experiment:
        archives, meta_csv, subdir = RAW_EXPERIMENTS[exp]
        if exp == "formal" and args.device:
            archives = [f"raw_formal_{d}.zip" for d in args.device]
        print(f"Raw archives for {exp}:")
        raw_dir = raw_root / exp
        for name in archives:
            src = root / name
            if not src.is_file():
                print(f"  {name}: MISSING in {root}")
                continue
            extract(src, raw_dir / subdir if subdir else raw_dir)
        if meta_csv:
            write_local_metadata(zenodo_dir, meta_csv, raw_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
