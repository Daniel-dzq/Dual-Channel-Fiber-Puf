"""Manuscript-level summaries of the frozen attack tables (Fig. 8, Supplementary Figs. S3-S4)."""

from __future__ import annotations

import pandas as pd

STATES = [f"M{i}" for i in range(8)]

ATTACK_LABELS = {
    "exact_template_replay": "Exact replay",
    "mean_response": "Mean-response baseline",
    "ridge_clone": "Ridge",
    "kernel_ridge_clone": "Kernel ridge",
    "random_fourier_ridge_clone": "RFF ridge",
    "small_mlp_clone": "Small MLP",
    "mean_leaked_response": "Mean-response baseline",
    "nearest_leaked_challenge": "Nearest-challenge baseline",
}


def state_matrix(table: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """8 x 8 source-state x target-state matrix of the median unit-level score across devices."""
    m = table.groupby(["source_state", "target_state"])[value_col].median().unstack()
    return m.reindex(index=STATES, columns=STATES)


def same_minus_cross_delta(table: pd.DataFrame, value_col: str = "median_S_A") -> pd.Series:
    """Delta S_A = S_A(same state) - S_A(cross state) for every ordered (device, source, target != source) pair.

    The manuscript reports the median of these 560 pair-level differences per attack method.
    """
    same = table[table.is_diagonal].set_index(["device_id", "source_state"])[value_col]
    cross = table[~table.is_diagonal]
    keys = pd.MultiIndex.from_frame(cross[["device_id", "source_state"]])
    return pd.Series(same.reindex(keys).to_numpy() - cross[value_col].to_numpy(), index=cross.index, name="delta_S_A")
