#!/usr/bin/env python3
"""Supplementary Fig. S3 - mechanical-state specificity of attack-generated responses (complete disclosure).

Source-to-target state matrices for exact replay, ridge, kernel ridge, RFF ridge and the
small MLP. Each element is the median unit-level attack score S_A across the ten devices;
templates / model parameters fitted on the complete 128-CRP bank of the source state are
evaluated on independent Round B responses of the target state without retraining
(``analysis_ready_data/formal/track_d_clone_transfer_matrix_summary.csv``).
"""

from __future__ import annotations

import numpy as np

from experiment4_security.ml_attack.summaries import ATTACK_LABELS, state_matrix
from puf_common.checks import check, open_dataset, report, reproduction_parser

PANELS = {"exact_template_replay": "FigS3_exact_replay_state_matrix.csv", "ridge_clone": "FigS3_ridge_state_matrix.csv", "kernel_ridge_clone": "FigS3_kernel_ridge_state_matrix.csv", "random_fourier_ridge_clone": "FigS3_rff_ridge_state_matrix.csv", "small_mlp_clone": "FigS3_small_mlp_state_matrix.csv"}
EXPECTED_DIAGONAL_RANGE = {"exact_template_replay": (0.92, 0.94), "ridge_clone": (0.76, 0.81), "kernel_ridge_clone": (0.75, 0.80), "random_fourier_ridge_clone": (0.61, 0.65), "small_mlp_clone": (0.24, 0.27)}
EXPECTED_REPLAY_OFFDIAG = (-0.02, -0.01)


def plot(matrices, out) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6))
    for ax, (method, m) in zip(axes, matrices.items()):
        im = ax.imshow(m.to_numpy(float), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set(xticks=range(8), yticks=range(8), xticklabels=m.columns, yticklabels=m.index, title=ATTACK_LABELS[method], xlabel="Target state")
    axes[0].set_ylabel("Source state")
    fig.colorbar(im, ax=axes, fraction=0.01, label="median $S_A$")
    fig.savefig(out / "supp_fig_s3.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = reproduction_parser(__doc__.splitlines()[0]).parse_args()
    ds, out = open_dataset(args, "supp_fig_s3")
    d = ds.read_csv("analysis_ready_data/formal/track_d_clone_transfer_matrix_summary.csv")

    matrices = {m: state_matrix(d[d.attack_method == m], "median_S_A") for m in PANELS}
    for m, mat in matrices.items():
        mat.to_csv(out / f"s3_{m}_state_matrix.csv")
    if not args.no_figures:
        plot(matrices, out)

    off = ~np.eye(8, dtype=bool)
    checks = []
    for m, fname in PANELS.items():
        frozen = ds.read_csv(f"Source_Data/FigS3/{fname}", index_col=0)
        arr = matrices[m].to_numpy(float)
        lo, hi = EXPECTED_DIAGONAL_RANGE[m]
        checks.append(check(f"{ATTACK_LABELS[m]}: matrix equals frozen Source Data", float(np.abs(arr - frozen.to_numpy(float)).max()), 0.0, 1e-12))
        checks.append(check(f"{ATTACK_LABELS[m]}: diagonal within [{lo}, {hi}] (display)", bool(round(np.diag(arr).min(), 2) >= lo and round(np.diag(arr).max(), 2) <= hi), True))
    replay_off = matrices["exact_template_replay"].to_numpy(float)[off]
    checks.append(check("Exact replay off-diagonal within [-0.02, -0.01] (display)", bool(round(replay_off.min(), 2) >= EXPECTED_REPLAY_OFFDIAG[0] and round(replay_off.max(), 2) <= EXPECTED_REPLAY_OFFDIAG[1]), True))
    return 0 if report(checks, out, "Supplementary Fig. S3 authority checks") else 1


if __name__ == "__main__":
    raise SystemExit(main())
