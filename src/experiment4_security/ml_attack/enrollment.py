"""State-specific enrollment common templates (detail_cm construction).

Formal definition (per protocol):

  common_s_A = mean_c detail(A_s,c)   for c in C001..C128
  T_s,c      = detail(A_s,c) - common_s_A
  Q_t,c|s    = detail(B_t,c) - common_s_A

The enrollment common for source state Ms is fit ONLY on that state's
Round A registered responses. Target Round B never enters the common.
Cross-state tests always subtract the *source* enrollment common.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


def _sha256_arr(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@dataclass
class EnrollmentCommon:
    state_id: str
    common_vector: np.ndarray  # float32, masked detail space
    fit_sample_ids: list[str]
    fit_challenge_ids: list[str]
    common_hash: str

    def to_detail_cm(self, detail_vector: np.ndarray) -> np.ndarray:
        return (detail_vector.astype(np.float64) - self.common_vector.astype(np.float64)).astype(np.float32)


def fit_enrollment_common(
    state_id: str,
    challenge_ids: list[str],
    *,
    detail_lookup: Callable[[str, str, str], np.ndarray],
    device_id: str = "F01",
) -> EnrollmentCommon:
    """Fit common_s_A from Round A only for one mechanical state."""
    if not challenge_ids:
        raise ValueError(f"No challenges for enrollment common of {state_id}")
    sample_ids: list[str] = []
    stack = []
    for cid in challenge_ids:
        sid = f"{device_id}_{state_id}_A_{cid}"
        sample_ids.append(sid)
        stack.append(detail_lookup(state_id, "A", cid))
    arr = np.stack(stack, axis=0).astype(np.float64)
    common = arr.mean(axis=0).astype(np.float32)
    return EnrollmentCommon(
        state_id=state_id,
        common_vector=common,
        fit_sample_ids=sample_ids,
        fit_challenge_ids=list(challenge_ids),
        common_hash=_sha256_arr(common),
    )


def build_all_enrollment_commons(
    states: list[str],
    challenge_ids: list[str],
    *,
    detail_lookup: Callable[[str, str, str], np.ndarray],
    device_id: str = "F01",
) -> dict[str, EnrollmentCommon]:
    out: dict[str, EnrollmentCommon] = {}
    for state in states:
        out[state] = fit_enrollment_common(
            state, challenge_ids, detail_lookup=detail_lookup, device_id=device_id
        )
    return out


def enrollment_templates(
    common: EnrollmentCommon,
    challenge_ids: list[str],
    *,
    detail_lookup: Callable[[str, str, str], np.ndarray],
) -> dict[str, np.ndarray]:
    """T_s,c = detail(A_s,c) - common_s_A for every registered challenge."""
    return {
        cid: common.to_detail_cm(detail_lookup(common.state_id, "A", cid))
        for cid in challenge_ids
    }


def query_vectors(
    source_common: EnrollmentCommon,
    target_state: str,
    challenge_ids: list[str],
    *,
    detail_lookup: Callable[[str, str, str], np.ndarray],
) -> dict[str, np.ndarray]:
    """Q_t,c|s = detail(B_t,c) - common_s_A (source common, target Round B)."""
    return {
        cid: source_common.to_detail_cm(detail_lookup(target_state, "B", cid))
        for cid in challenge_ids
    }


def write_enrollment_artifacts(
    commons: dict[str, EnrollmentCommon],
    out_dir: Path,
) -> dict[str, Any]:
    """Persist commons + audit tables. Never writes into lifecycle."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmpl_dir = out_dir / "state_specific_common_templates"
    tmpl_dir.mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = {}
    fit_rows: list[dict[str, Any]] = []
    hash_map: dict[str, str] = {}

    for state, common in commons.items():
        npy_path = tmpl_dir / f"{state}_common_A.npy"
        np.save(npy_path, common.common_vector)
        meta[state] = {
            "state_id": state,
            "n_fit_samples": len(common.fit_sample_ids),
            "fit_round": "A",
            "common_hash": common.common_hash,
            "vector_path": str(npy_path),
            "vector_shape": list(common.common_vector.shape),
            "dtype": str(common.common_vector.dtype),
            "note": "Fit exclusively on Round A enrollment details; Round B never enters.",
        }
        hash_map[state] = common.common_hash
        for sid, cid in zip(common.fit_sample_ids, common.fit_challenge_ids):
            fit_rows.append(
                {
                    "state_id": state,
                    "round_id": "A",
                    "challenge_id": cid,
                    "sample_id": sid,
                    "used_in_common": True,
                }
            )

    import json

    with (out_dir / "state_specific_common_templates.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    with (out_dir / "common_template_hashes.json").open("w", encoding="utf-8") as fh:
        json.dump(hash_map, fh, indent=2)
    pd.DataFrame(fit_rows).to_csv(out_dir / "state_specific_common_fit_samples.csv", index=False)
    return meta
