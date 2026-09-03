"""Explicit development / frozen / descriptive evaluation scopes."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

ROLE_DEVELOPMENT = "development"
ROLE_FROZEN_TEST = "frozen_test"

SCOPE_DEVELOPMENT_ONLY = "development_only"
SCOPE_FROZEN_FORMAL = "frozen_formal"
SCOPE_ALL_COHORT_DESCRIPTIVE = "all_cohort_descriptive"


def device_role(device_id: str, *, dev: Iterable[str], frozen: Iterable[str]) -> str:
    if device_id in set(dev):
        return ROLE_DEVELOPMENT
    if device_id in set(frozen):
        return ROLE_FROZEN_TEST
    raise ValueError(f"Device {device_id} is not in development or frozen splits")


def annotate_metadata_roles(
    metadata: pd.DataFrame,
    *,
    development_devices: list[str],
    frozen_devices: list[str],
) -> pd.DataFrame:
    df = metadata.copy()
    role_map = {d: ROLE_DEVELOPMENT for d in development_devices}
    role_map.update({d: ROLE_FROZEN_TEST for d in frozen_devices})
    unknown = sorted(set(df["device_id"]) - set(role_map))
    if unknown:
        raise RuntimeError(
            f"Devices missing from configured splits: {unknown}. "
            "Formal evaluation cannot continue."
        )
    df["dataset_role"] = df["device_id"].map(role_map)
    return df


def filter_score_pairs_by_devices(
    scores: pd.DataFrame,
    devices: list[str],
) -> pd.DataFrame:
    """Keep pairs where both endpoints are in the device set."""
    if scores.empty:
        return scores.copy()
    devices_set = set(devices)
    out = scores[
        scores["device_id_a"].isin(devices_set) & scores["device_id_b"].isin(devices_set)
    ].copy()
    return out


def filter_queries_by_devices(query_df: pd.DataFrame, devices: list[str]) -> pd.DataFrame:
    if query_df.empty:
        return query_df.copy()
    return query_df[query_df["query_device"].isin(devices)].copy()


def assert_disjoint(dev: list[str], frozen: list[str]) -> None:
    overlap = sorted(set(dev) & set(frozen))
    if overlap:
        raise RuntimeError(f"Development and frozen device sets overlap: {overlap}")


def require_nonempty_dev(dev: list[str]) -> None:
    if not dev:
        raise RuntimeError(
            "Development device set is empty. Formal evaluation cannot continue."
        )


def require_nonempty_frozen(frozen: list[str], *, require_for_formal: bool) -> None:
    if require_for_formal and not frozen:
        raise RuntimeError(
            "Frozen test device set is empty. Formal held-out metrics cannot be generated."
        )
