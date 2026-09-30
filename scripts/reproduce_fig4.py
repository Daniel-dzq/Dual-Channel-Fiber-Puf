#!/usr/bin/env python3
"""Fig. 4 - fiber-length optimization (7, 9, 11, 13, 15 cm; five devices; eight challenges).

Lightweight mode: recomputes the length-selection statistics from the frozen
cross-round pair scores in ``Source_Data/Fig4/Fig4c_pair_NCC_scores.csv``.

* S_intra, S_inter,c and S_inter,d are Round A vs Round B scores;
* G_min(L) = min[Q05(S_intra) - Q95(S_inter,c), Q05(S_intra) - Q95(S_inter,d)];
* L* = argmax G_min(L); hierarchical bootstrap (device, then challenge within
  device) with B = 5000 replicates and seed 20260721; ties resolved toward the
  shorter length.
* H_PSD (Supplementary Note 4) is complementary spatial-frequency characterization and is not
  part of the selection criterion.

Raw mode (decode ``raw_fiber_length.zip``) is documented in docs/REPRODUCIBILITY.md
and driven by ``python -m experiment00.cli analyze --config configs/fiber_length_optimization.yaml``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiment00.authentication import length_separation_metrics
from experiment00.bootstrap import hierarchical_bootstrap_length_metrics
from experiment00.length_selection import select_length
from experiment00.public_scores import load_fig4_pair_scores
from puf_common.checks import check, open_dataset, report, reproduction_parser

BOOTSTRAP_B = 5000
BOOTSTRAP_SEED = 20260721

EXPECTED = {
    "selected_length_cm": 9,
    "Q05_S_intra_9cm": 0.5499508470790516,
    "Q95_S_inter_c_9cm": 0.1601584391516029,
    "Q95_S_inter_d_9cm": 0.01417100814600619,
    "G_min_9cm": 0.3897924079274487,
    "H_PSD_median_9cm": 0.10708552481221825,
    "representative_NCC_F01_C01_9cm": 0.8205578290959264,
    "bootstrap_9cm_selected": 4956,
    "bootstrap_complete": 4967,
}


def plot(pairs: pd.DataFrame, metrics: pd.DataFrame, selection: pd.DataFrame, psd: pd.DataFrame, out) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lengths = metrics["length_cm"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for st, color in (("S_intra", "tab:green"), ("S_inter_challenge", "tab:orange"), ("S_inter_device", "tab:gray")):
        data = [pairs.loc[(pairs.length_cm == L) & (pairs.score_type == st), "score"].to_numpy() for L in lengths]
        parts = axes[0].violinplot(data, positions=lengths, widths=1.2, showmedians=True)
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.45)
    axes[0].set(xlabel="Fiber length L (cm)", ylabel="NCC", title="Fig. 4c  cross-round NCC scores")

    axes[1].plot(lengths, metrics["robust_gap_min"], "o-", color="k", label="$G_{min}(L)$")
    axes[1].plot(lengths, metrics["robust_gap_challenge"], "s--", color="tab:orange", label="Q05(S$_{intra}$) - Q95(S$_{inter,c}$)")
    axes[1].plot(lengths, metrics["robust_gap_device"], "^--", color="tab:gray", label="Q05(S$_{intra}$) - Q95(S$_{inter,d}$)")
    axes[1].axhline(0, color="0.6", lw=0.8)
    axes[1].set(xlabel="Fiber length L (cm)", ylabel="Robust gap", title="Fig. 4d  G_min(L)")
    axes[1].legend(fontsize=7)

    axes[2].bar(selection["length_cm"], selection["selection_probability"], color="tab:blue")
    attempted = int(selection.n_attempted_replicates.iloc[0])
    complete = int(selection.n_complete_replicates.iloc[0])
    axes[2].set(xlabel="Fiber length L (cm)", ylabel="Selection frequency", ylim=(0, 1.05), title=f"Fig. 4e  bootstrap ({complete}/{attempted} complete)")
    for _, r in selection.iterrows():
        axes[2].text(r.length_cm, r.selection_probability + 0.02, f"{int(r.n_selected)}/{complete}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig4_length_optimization.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    med = psd.groupby("L_cm")["H_PSD"].quantile([0.25, 0.5, 0.75]).unstack()
    ax.errorbar(med.index, med[0.5], yerr=[med[0.5] - med[0.25], med[0.75] - med[0.5]], fmt="o-", color="tab:purple", capsize=3)
    ax.set(xlabel="Fiber length L (cm)", ylabel="$H_{PSD}$ (median, Q25-Q75)", title="Supplementary Note 4: PSD entropy")
    fig.tight_layout()
    fig.savefig(out / "spectral_entropy.png", dpi=200)
    plt.close(fig)


def main() -> int:
    p = reproduction_parser(__doc__.splitlines()[0])
    p.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_B, help="Set lower only for smoke tests; the authority uses 5000")
    args = p.parse_args()
    ds, out = open_dataset(args, "fig4")

    pairs = load_fig4_pair_scores(ds)
    metrics = length_separation_metrics(pairs)
    _, selection, pairwise = hierarchical_bootstrap_length_metrics(
        pairs, n_iterations=args.bootstrap_replicates, seed=BOOTSTRAP_SEED
    )
    decision = select_length(metrics, selection)
    psd = ds.read_csv("Source_Data/Fig4/Fig4e_H_PSD_per_video.csv")

    metrics.to_csv(out / "length_metrics.csv", index=False)
    selection.to_csv(out / "bootstrap_selection.csv", index=False)
    pairwise.to_csv(out / "pairwise_length_probabilities.csv", index=False)
    psd.groupby("L_cm")["H_PSD"].describe().to_csv(out / "psd_entropy_by_length.csv")
    (out / "selection_decision.json").write_text(pd.Series(decision.to_dict()).to_json(indent=2), encoding="utf-8")
    if not args.no_figures:
        plot(pairs, metrics, selection, psd, out)

    nine = pairs[pairs.length_cm == 9]
    g = nine.loc[nine.score_type == "S_intra", "score"].to_numpy()
    c = nine.loc[nine.score_type == "S_inter_challenge", "score"].to_numpy()
    d = nine.loc[nine.score_type == "S_inter_device", "score"].to_numpy()
    rep = nine[(nine.score_type == "S_intra") & (nine.fiber_id == "F01") & (nine.challenge == "C01")]["score"].item()
    row9 = metrics.set_index("length_cm").loc[9]
    n9 = int(selection.set_index("length_cm").loc[9, "n_selected"])
    full_b = args.bootstrap_replicates == BOOTSTRAP_B

    checks = [
        check("selected length (cm)", int(decision.selected_length_cm), EXPECTED["selected_length_cm"], 0),
        check("Q05(S_intra) at 9 cm", float(np.quantile(g, 0.05)), EXPECTED["Q05_S_intra_9cm"]),
        check("Q95(S_inter,c) at 9 cm", float(np.quantile(c, 0.95)), EXPECTED["Q95_S_inter_c_9cm"]),
        check("Q95(S_inter,d) at 9 cm", float(np.quantile(d, 0.95)), EXPECTED["Q95_S_inter_d_9cm"]),
        check("G_min at 9 cm", float(row9["robust_gap_min"]), EXPECTED["G_min_9cm"]),
        check("H_PSD median at 9 cm (complementary)", float(psd.loc[psd.L_cm == 9, "H_PSD"].median()), EXPECTED["H_PSD_median_9cm"]),
        check("representative NCC F01/C01 at 9 cm (Fig. 4b)", rep, EXPECTED["representative_NCC_F01_C01_9cm"]),
    ]
    if full_b:
        checks.extend([
            check("complete bootstrap replicates", int(selection.n_complete_replicates.iloc[0]), EXPECTED["bootstrap_complete"], 0),
            check("bootstrap replicates selecting 9 cm", n9, EXPECTED["bootstrap_9cm_selected"], 0),
        ])
    return 0 if report(checks, out, "Fig. 4 authority checks") else 1


if __name__ == "__main__":
    raise SystemExit(main())
