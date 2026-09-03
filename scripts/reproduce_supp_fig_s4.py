#!/usr/bin/env python3
"""Supplementary Fig. S4 - hidden-response prediction and model audit under partial CRP disclosure.

* (a) source-to-target state matrix for ridge regression with N_L = 96 disclosed CRPs
  (median attack score S_A over the 32 undisclosed challenges, five partitions and ten devices);
* (b) retained PCA dimension d_PCA = max(1, min(64, N_L - 1));
* (c) median residual NCC after removing the common predicted-response component;
* (d) fraction of predicted responses dominated by the common component.

Inputs: ``analysis_ready_data/attacks/track_d_pl_partial_leakage_transfer_summary.csv``
and ``Source_Data/Fig8/Fig8c_partial_disclosure_unit_evaluations.csv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiment4_security.ml_attack.summaries import state_matrix
from puf_common.checks import check, open_dataset, report, reproduction_parser

LEARNING_METHODS = ["Ridge", "Kernel Ridge", "RFF Ridge", "Small MLP"]
COMMON_DOMINATED = "HIGH_NCC_DOMINATED_BY_COMMON_RESPONSE"
EXPECTED = {"d_PCA": {16: 15, 32: 31, 64: 63, 96: 64}, "ridge96_diag_range": (-0.08, -0.06), "hidden_reference_median_NL96": 0.929}


def prescribed_pca_dimension(n_leaked: int, cap: int = 64) -> int:
    return max(1, min(cap, n_leaked - 1))


def plot(matrix, audit: pd.DataFrame, out) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    im = axes[0].imshow(matrix.to_numpy(float), cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set(xticks=range(8), yticks=range(8), xticklabels=matrix.columns, yticklabels=matrix.index, xlabel="Target state", ylabel="Source state", title="(a) ridge, $N_L$ = 96")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    for ax, col, title in ((axes[1], "d_PCA", "(b) retained PCA dimension"), (axes[2], "residual_NCC", "(c) median residual NCC"), (axes[3], "common_dominated_fraction", "(d) common-dominated fraction")):
        for m in LEARNING_METHODS:
            s = audit[audit.attack_method == m].set_index("N_L")[col]
            ax.plot(s.index, s.values, "o-", label=m)
        ax.set(xlabel="Disclosed CRPs $N_L$", xticks=[16, 32, 64, 96], title=title)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "supp_fig_s4.png", dpi=200)
    plt.close(fig)


def main() -> int:
    args = reproduction_parser(__doc__.splitlines()[0]).parse_args()
    ds, out = open_dataset(args, "supp_fig_s4")

    transfer = ds.read_csv("analysis_ready_data/attacks/track_d_pl_partial_leakage_transfer_summary.csv")
    units = ds.read_csv("Source_Data/Fig8/Fig8c_partial_disclosure_unit_evaluations.csv")
    frozen = ds.read_csv("Source_Data/FigS4/FigS4a_ridge_NL96_state_matrix.csv", index_col=0)

    ridge96 = state_matrix(transfer[(transfer.attack_method == "ridge_clone") & (transfer.leak_size == 96)], "median_S_A_partial")
    learning = units[units.attack_method.isin(LEARNING_METHODS)].copy()
    learning["common_dominated"] = learning.common_dominance_flag.eq(COMMON_DOMINATED).astype(float)
    audit = learning.groupby(["attack_method", "N_L"], as_index=False).agg(d_PCA=("d_PCA", "first"), residual_NCC=("residual_NCC", "median"), common_dominated_fraction=("common_dominated", "mean"))

    ridge96.to_csv(out / "s4a_ridge_NL96_state_matrix.csv")
    audit.to_csv(out / "s4bcd_model_audit.csv", index=False)
    if not args.no_figures:
        plot(ridge96, audit, out)

    diag = np.diag(ridge96.to_numpy(float))
    flags = set(units.common_dominance_flag.unique())
    checks = [
        check("S4(a) matrix equals frozen Source Data", float(np.abs(ridge96.to_numpy(float) - frozen.to_numpy(float)).max()), 0.0, 1e-12),
        check("S4(a) diagonal within [-0.08, -0.06] (display)", bool(round(diag.min(), 2) >= EXPECTED["ridge96_diag_range"][0] and round(diag.max(), 2) <= EXPECTED["ridge96_diag_range"][1]), True),
        check("S4(a) hidden-response reference median at N_L = 96", round(float(units.loc[units.N_L == 96, "genuine_hidden_reference"].median()), 3), EXPECTED["hidden_reference_median_NL96"], 0),
        *[check(f"d_PCA at N_L = {n} (prescribed {prescribed_pca_dimension(n)})", int(audit.loc[audit.N_L == n, "d_PCA"].iloc[0]), EXPECTED["d_PCA"][n], 0) for n in (16, 32, 64, 96)],
        check("d_PCA identical across learning methods", bool((audit.groupby("N_L").d_PCA.nunique() == 1).all()), True),
        check("common-dominance flag present in audit table", COMMON_DOMINATED in flags, True),
        check("common-dominated fraction non-increasing in N_L for every method", bool(all(np.all(np.diff(g.sort_values("N_L").common_dominated_fraction.to_numpy()) <= 1e-12) for _, g in audit.groupby("attack_method"))), True),
    ]
    return 0 if report(checks, out, "Supplementary Fig. S4 authority checks") else 1


if __name__ == "__main__":
    raise SystemExit(main())
