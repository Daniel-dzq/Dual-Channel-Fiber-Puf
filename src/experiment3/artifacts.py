"""Frozen artifact bundle: mask, standardizer, thresholds, hashes."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return sha256_bytes(raw)


def save_mask_artifacts(
    mask: np.ndarray,
    run_dir: Path,
    *,
    metadata: dict[str, Any],
) -> dict[str, str]:
    """Write valid_mask.npy/png + metadata + hash. Returns path map."""
    from puf_common.masks import save_mask_png

    npy_path = run_dir / "valid_mask.npy"
    png_path = run_dir / "valid_mask.png"
    meta_path = run_dir / "valid_mask_metadata.json"
    hash_path = run_dir / "valid_mask_hash.txt"

    np.save(npy_path, mask.astype(bool))
    save_mask_png(mask, png_path)
    mask_bytes = npy_path.read_bytes()
    digest = sha256_bytes(mask_bytes)
    meta = {
        **metadata,
        "sha256": digest,
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "contains_frozen_data": False,
        "shape": list(mask.shape),
        "coverage": float(np.mean(mask.astype(bool))),
    }
    if meta.get("contains_frozen_data") is not False:
        raise RuntimeError("valid_mask metadata must set contains_frozen_data=false")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    hash_path.write_text(digest + "\n", encoding="utf-8")
    return {
        "valid_mask.npy": str(npy_path),
        "valid_mask.png": str(png_path),
        "valid_mask_metadata.json": str(meta_path),
        "valid_mask_hash.txt": str(hash_path),
        "sha256": digest,
    }


def save_standardizer_artifacts(
    run_dir: Path,
    *,
    mu: np.ndarray,
    sd: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, str]:
    payload = {
        "mean": np.asarray(mu, dtype=float).tolist(),
        "scale": np.asarray(sd, dtype=float).tolist(),
        **{k: v for k, v in metadata.items() if k not in {"mean", "scale"}},
    }
    payload["contains_frozen_data"] = False
    digest = sha256_json(payload)
    payload["sha256"] = digest
    json_path = run_dir / "red_standardizer.json"
    meta_path = run_dir / "red_standardizer_metadata.json"
    hash_path = run_dir / "red_standardizer_hash.txt"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    hash_path.write_text(digest + "\n", encoding="utf-8")
    return {
        "red_standardizer.json": str(json_path),
        "red_standardizer_metadata.json": str(meta_path),
        "red_standardizer_hash.txt": str(hash_path),
        "sha256": digest,
    }


def save_threshold_artifacts(
    run_dir: Path,
    *,
    tau_r: float,
    tau_g: float,
    source: dict[str, Any],
) -> dict[str, str]:
    tau_r_obj = {
        "tau_R": float(tau_r),
        "operational": True,
        "source": "development",
        **source,
    }
    tau_g_obj = {
        "tau_G": float(tau_g),
        "operational": True,
        "source": "development",
        **source,
    }
    (run_dir / "tau_R.json").write_text(json.dumps(tau_r_obj, indent=2), encoding="utf-8")
    (run_dir / "tau_G.json").write_text(json.dumps(tau_g_obj, indent=2), encoding="utf-8")
    return {
        "tau_R.json": str(run_dir / "tau_R.json"),
        "tau_G.json": str(run_dir / "tau_G.json"),
        "sha256_tau_R": sha256_json(tau_r_obj),
        "sha256_tau_G": sha256_json(tau_g_obj),
    }


def write_frozen_protocol_bundle(
    run_dir: Path,
    *,
    protocol: dict[str, Any],
    artifact_hashes: dict[str, str],
) -> None:
    (run_dir / "frozen_protocol.yaml").write_text(
        yaml.safe_dump(protocol, sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "artifact_hashes.json").write_text(
        json.dumps(artifact_hashes, indent=2),
        encoding="utf-8",
    )
    (run_dir / "frozen_artifact_manifest.json").write_text(
        json.dumps(
            {
                "bundle_dir": str(run_dir),
                "artifacts": sorted(artifact_hashes.keys()),
                "protocol_keys": sorted(protocol.keys()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
