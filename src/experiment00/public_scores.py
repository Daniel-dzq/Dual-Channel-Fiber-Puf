"""Frozen Fig. 4 cross-round pair scores from the public Source Data, in pipeline column names."""

from __future__ import annotations

import pandas as pd

SCORE_CLASS = {"S_intra": "S_intra", "S_inter_c": "S_inter_challenge", "S_inter_d": "S_inter_device"}


def load_fig4_pair_scores(ds) -> pd.DataFrame:
    src = ds.read_csv("Source_Data/Fig4/Fig4c_pair_NCC_scores.csv")
    return pd.DataFrame(
        {
            "length_cm": src["L_cm"].astype(int),
            "fiber_id": src["device_id"],
            "challenge": src["challenge_id"],
            "fiber_id_b": src["device_id_b"],
            "challenge_b": src["challenge_id_b"],
            "score_type": src["score_class"].map(SCORE_CLASS),
            "score": src["NCC"].astype(float),
        }
    )
