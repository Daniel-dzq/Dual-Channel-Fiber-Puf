"""Bank-balanced, nested partial-leakage challenge splits (Track C-PL / D-PL).

Design (frozen, see configs/partial_disclosure.yaml):

- The 128-challenge bank is partitioned into 16 analysis banks (B01..B16) of
  8 challenges each (already recorded in ``challenge_manifest.csv``).
- For a fixed split-repetition seed, each bank is independently shuffled
  with a seed derived deterministically from (seed, bank_id) — never from
  challenge content or from any measured response.
- Leak sizes 16/32/64/96 take the first 1/2/4/6 shuffled challenges of every
  bank, so leak sets are (a) always bank-balanced (all 16 banks represented
  at every leak size) and (b) strictly nested across leak sizes (leak=16 is
  a subset of leak=32, which is a subset of leak=64, which is a subset of
  leak=96).
- ``hidden`` is always the complement within the full 128-challenge bank —
  leaked and hidden are disjoint and their union is always C001..C128.

Nothing in this module ever looks at measured responses (Round A or Round
B); it only permutes challenge IDs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_LEAK_SIZES = (16, 32, 64, 96)
DEFAULT_PER_BANK_COUNTS = (1, 2, 4, 6)
DEFAULT_SEEDS = (20260801, 20260802, 20260803, 20260804, 20260805)


def load_bank_map(manifest_csv: Path | str) -> dict[str, list[str]]:
    """bank_id -> challenge_ids ordered by bank_local_index (1..8)."""
    df = pd.read_csv(manifest_csv)
    required = {"challenge_id", "bank_id", "bank_local_index"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"challenge manifest missing columns: {missing}")
    out: dict[str, list[str]] = {}
    for bank_id, grp in df.groupby("bank_id"):
        ordered = grp.sort_values("bank_local_index")["challenge_id"].tolist()
        out[str(bank_id)] = [str(c) for c in ordered]
    return dict(sorted(out.items()))


def _bank_seed(seed: int, bank_id: str) -> int:
    """Deterministic, independent-looking per-bank sub-seed."""
    h = hashlib.sha256(f"partial_leakage_bank_shuffle|{seed}|{bank_id}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % (2**32 - 1)


def shuffle_bank(seed: int, bank_id: str, challenge_ids: list[str]) -> list[str]:
    rng = np.random.default_rng(_bank_seed(seed, bank_id))
    idx = rng.permutation(len(challenge_ids))
    return [challenge_ids[i] for i in idx]


@dataclass(frozen=True)
class LeakSplit:
    seed: int
    leak_size: int
    leaked: tuple[str, ...]
    hidden: tuple[str, ...]

    def __post_init__(self) -> None:
        leaked_set = set(self.leaked)
        hidden_set = set(self.hidden)
        if leaked_set & hidden_set:
            raise AssertionError("leaked and hidden challenge sets must be disjoint")
        if len(self.leaked) != self.leak_size:
            raise AssertionError(f"leak_size={self.leak_size} but got {len(self.leaked)} leaked ids")


def generate_splits_for_seed(
    seed: int,
    bank_map: dict[str, list[str]],
    *,
    leak_sizes: tuple[int, ...] = DEFAULT_LEAK_SIZES,
    per_bank_counts: tuple[int, ...] = DEFAULT_PER_BANK_COUNTS,
) -> dict[int, LeakSplit]:
    """One nested family of leak/hidden splits for a single repetition seed."""
    if len(leak_sizes) != len(per_bank_counts):
        raise ValueError("leak_sizes and per_bank_counts must have equal length")

    shuffled_by_bank = {
        bank_id: shuffle_bank(seed, bank_id, challenges) for bank_id, challenges in bank_map.items()
    }
    all_challenge_ids = {cid for challenges in bank_map.values() for cid in challenges}

    out: dict[int, LeakSplit] = {}
    for leak_size, per_bank in zip(leak_sizes, per_bank_counts):
        leaked: list[str] = []
        for bank_id in sorted(shuffled_by_bank.keys()):
            leaked.extend(shuffled_by_bank[bank_id][:per_bank])
        leaked_set = set(leaked)
        if len(leaked_set) != leak_size:
            raise AssertionError(
                f"seed={seed} leak_size={leak_size}: expected {leak_size} unique leaked "
                f"challenges, got {len(leaked_set)}"
            )
        hidden = sorted(all_challenge_ids - leaked_set)
        out[leak_size] = LeakSplit(seed=seed, leak_size=leak_size, leaked=tuple(sorted(leaked)), hidden=tuple(hidden))
    return out


def assert_nested(splits_by_leak_size: dict[int, LeakSplit]) -> None:
    ordered = sorted(splits_by_leak_size.keys())
    for smaller, larger in zip(ordered, ordered[1:]):
        s_leaked = set(splits_by_leak_size[smaller].leaked)
        l_leaked = set(splits_by_leak_size[larger].leaked)
        if not s_leaked.issubset(l_leaked):
            raise AssertionError(f"leak_size={smaller} is not nested inside leak_size={larger}")


def build_split_manifest(
    seeds: tuple[int, ...],
    bank_map: dict[str, list[str]],
    *,
    leak_sizes: tuple[int, ...] = DEFAULT_LEAK_SIZES,
    per_bank_counts: tuple[int, ...] = DEFAULT_PER_BANK_COUNTS,
) -> pd.DataFrame:
    """One row per (repetition, leak_size, challenge_id)."""
    challenge_to_bank = {cid: bank_id for bank_id, cids in bank_map.items() for cid in cids}
    rows: list[dict[str, Any]] = []
    for rep, seed in enumerate(seeds):
        splits = generate_splits_for_seed(seed, bank_map, leak_sizes=leak_sizes, per_bank_counts=per_bank_counts)
        assert_nested(splits)
        for leak_size, split in splits.items():
            leaked_set = set(split.leaked)
            for bank_id, cids in bank_map.items():
                for cid in cids:
                    rows.append(
                        {
                            "repetition": rep,
                            "seed": seed,
                            "bank_id": bank_id,
                            "challenge_id": cid,
                            "leak_size": leak_size,
                            "role": "leaked" if cid in leaked_set else "hidden",
                        }
                    )
    df = pd.DataFrame(rows).sort_values(["repetition", "leak_size", "bank_id", "challenge_id"]).reset_index(drop=True)
    # sanity: challenge_to_bank must agree with bank_map grouping used above
    assert all(challenge_to_bank[c] for c in challenge_to_bank)
    return df


def splits_from_manifest(manifest: pd.DataFrame, *, repetition: int, leak_size: int) -> LeakSplit:
    sub = manifest[(manifest["repetition"] == repetition) & (manifest["leak_size"] == leak_size)]
    seed = int(sub["seed"].iloc[0])
    leaked = tuple(sorted(sub.loc[sub["role"] == "leaked", "challenge_id"]))
    hidden = tuple(sorted(sub.loc[sub["role"] == "hidden", "challenge_id"]))
    return LeakSplit(seed=seed, leak_size=leak_size, leaked=leaked, hidden=hidden)


def all_splits_by_repetition(
    manifest: pd.DataFrame, *, leak_sizes: tuple[int, ...] = DEFAULT_LEAK_SIZES
) -> dict[int, dict[int, LeakSplit]]:
    reps = sorted(manifest["repetition"].unique().tolist())
    return {
        rep: {leak_size: splits_from_manifest(manifest, repetition=rep, leak_size=leak_size) for leak_size in leak_sizes}
        for rep in reps
    }
