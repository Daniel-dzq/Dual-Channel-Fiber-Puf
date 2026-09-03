#!/usr/bin/env python3
"""Minimal end-to-end example on the public dataset (no video decoding).

1. open the Zenodo download,
2. recompute the Fig. 4 length-selection statistic G_min(L) from the frozen cross-round scores,
3. score two challenge patterns of the eight-challenge bank with the zero-mean NCC,
4. evaluate the Fig. 6 red identity scores with EER / AUC / robust gap.

    python examples/minimal_workflow.py --data-root /path/to/zenodo_dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from puf_common.metrics import auc_roc, equal_error_rate, robust_gap
from puf_common.ncc import zero_mean_ncc
from puf_common.public_data import PublicDataset, resolve_data_root

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", default=None)
    ds = PublicDataset(resolve_data_root(p.parse_args().data_root))

    pairs = ds.read_csv("Source_Data/Fig4/Fig4c_pair_NCC_scores.csv")
    print("Fig. 4  G_min(L) = min[Q05(S_intra) - Q95(S_inter,c), Q05(S_intra) - Q95(S_inter,d)]")
    for L, sub in pairs.groupby("L_cm"):
        q05 = sub.loc[sub.score_class == "S_intra", "NCC"].quantile(0.05)
        gap_c = q05 - sub.loc[sub.score_class == "S_inter_c", "NCC"].quantile(0.95)
        gap_d = q05 - sub.loc[sub.score_class == "S_inter_d", "NCC"].quantile(0.95)
        print(f"  L = {L:2d} cm  G_min = {min(gap_c, gap_d):.4f}")

    # active 512 x 512 region of the 1024 x 768 SLM canvas, at offset (256, 128)
    c01, c02 = (np.asarray(Image.open(REPO / f"data/challenge_patterns/mp002/mp002_C{i:02d}.png").convert("L"), float)[128:640, 256:768] for i in (1, 2))
    print(f"\nzero-mean NCC of challenge patterns  C01 vs C01: {zero_mean_ncc(c01, c01):+.4f}   C01 vs C02: {zero_mean_ncc(c01, c02):+.4f}")

    red = ds.read_csv("Source_Data/Fig6/Fig6e_red_identity_scores.csv")
    genuine = red.loc[red.comparison_class == "same device", "q_R"].to_numpy()
    impostor = red.loc[red.comparison_class == "different device", "q_R"].to_numpy()
    print(f"\nFig. 6e red identity: {len(genuine)} same-device and {len(impostor)} different-device comparisons")
    print(f"  AUC = {auc_roc(genuine, impostor):.6f}  EER = {equal_error_rate(genuine, impostor):.6f}  robust gap = {robust_gap(genuine, impostor):+.6f}")


if __name__ == "__main__":
    main()
