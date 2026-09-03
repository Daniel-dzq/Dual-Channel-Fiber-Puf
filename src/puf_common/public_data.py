"""Read-only access to the public Zenodo dataset (DOI 10.5281/zenodo.22267156).

``data_root`` is the directory into which the Zenodo record was downloaded. The
lightweight archives (``Source_Data.zip``, ``analysis_ready_data.zip``,
``challenges_calibration_masks.zip``) may be left zipped or extracted next to the
zip files; both layouts are read transparently. Nothing is ever written under
``data_root``.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATASET_DOI = "10.5281/zenodo.22267156"
DATA_ROOT_ENV = "PUF_DATA_ROOT"

# Archive name -> directory prefix used by the archive members.
LIGHTWEIGHT_ARCHIVES = {
    "Source_Data.zip": "Source_Data/",
    "analysis_ready_data.zip": "",
    "challenges_calibration_masks.zip": "",
}
ROOT_FILES = (
    "README.md",
    "README_Source_Data.txt",
    "Source_Data.xlsx",
    "data_quality_exclusions.csv",
    "file_manifest.csv",
    "raw_file_manifest.csv",
    "final_release_checksums.sha256",
    "representative_workflows.csv",
)
RAW_ARCHIVES = (
    "raw_macro_pixel.zip",
    "raw_fiber_length.zip",
    "raw_fixed_state.zip",
    "raw_wavelength_pathway.zip",
    "raw_threshold_development.zip",
    *(f"raw_formal_F{i:02d}.zip" for i in range(1, 11)),
)


def resolve_data_root(explicit: str | os.PathLike[str] | None) -> Path:
    """Command-line value first, then ``$PUF_DATA_ROOT``."""
    value = explicit or os.environ.get(DATA_ROOT_ENV)
    if not value:
        raise SystemExit(f"Dataset root not given: pass --data-root or set {DATA_ROOT_ENV}.")
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {root}")
    return root


class PublicDataset:
    """Locate and read files of the public dataset by their canonical relative path.

    Relative paths use the extracted layout, e.g. ``Source_Data/Fig4/Fig4c_pair_NCC_scores.csv``,
    ``analysis_ready_data/formal/metrics_summary.json`` or
    ``challenges_calibration_masks/masks/exp04_formal_valid_mask.npz``.
    """

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)
        self._zips: dict[str, zipfile.ZipFile] = {}

    # -- location -------------------------------------------------------------
    def _archive_for(self, rel: str) -> tuple[str, str] | None:
        head, _, tail = rel.partition("/")
        if head == "Source_Data":
            return "Source_Data.zip", rel
        if head == "analysis_ready_data":
            return "analysis_ready_data.zip", tail
        if head == "challenges_calibration_masks":
            return "challenges_calibration_masks.zip", tail
        return None

    def _zip(self, name: str) -> zipfile.ZipFile | None:
        if name not in self._zips:
            path = self.root / name
            self._zips[name] = zipfile.ZipFile(path) if path.is_file() else None
        return self._zips[name]

    def exists(self, rel: str) -> bool:
        if (self.root / rel).exists():
            return True
        hit = self._archive_for(rel)
        if hit is None:
            return False
        zf = self._zip(hit[0])
        return zf is not None and hit[1] in zf.namelist()

    def local_path(self, rel: str) -> Path | None:
        """Extracted on-disk path, or ``None`` if only available inside a zip."""
        p = self.root / rel
        return p if p.exists() else None

    # -- reading --------------------------------------------------------------
    def read_bytes(self, rel: str) -> bytes:
        p = self.root / rel
        if p.is_file():
            return p.read_bytes()
        hit = self._archive_for(rel)
        zf = self._zip(hit[0]) if hit else None
        if zf is None or hit[1] not in zf.namelist():
            raise FileNotFoundError(f"{rel} not found under {self.root} (extracted or zipped)")
        return zf.read(hit[1])

    def read_text(self, rel: str) -> str:
        return self.read_bytes(rel).decode("utf-8")

    def read_json(self, rel: str) -> Any:
        return json.loads(self.read_text(rel))

    def read_csv(self, rel: str, **kwargs: Any) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(self.read_bytes(rel)), **kwargs)

    def read_npz(self, rel: str) -> dict[str, np.ndarray]:
        with np.load(io.BytesIO(self.read_bytes(rel)), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def read_npy(self, rel: str) -> np.ndarray:
        return np.load(io.BytesIO(self.read_bytes(rel)), allow_pickle=False)

    # -- inventory ------------------------------------------------------------
    def inventory(self) -> dict[str, bool]:
        """Presence of every top-level dataset component (zipped or extracted)."""
        present = {}
        for name, prefix in LIGHTWEIGHT_ARCHIVES.items():
            present[name] = (self.root / name).is_file() or (self.root / name[: -len(".zip")]).is_dir()
        for name in ROOT_FILES:
            present[name] = (self.root / name).is_file()
        for name in RAW_ARCHIVES:
            present[name] = (self.root / name).is_file()
        return present
