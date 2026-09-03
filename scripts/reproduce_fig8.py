#!/usr/bin/env python3
"""Fig. 8 - response-reconstruction attacks under complete and partial CRP disclosure.

Lightweight mode recomputes, from the frozen formal attack tables
(``analysis_ready_data/formal/track_c_*``, ``track_d_*``, ``analysis_ready_data/attacks/*``
and ``Source_Data/Fig8``):

* Fig. 8a - median attack score S_A per device-state unit for the six methods under
  complete disclosure of the 128 enrolled CRPs (mean response, exact replay, ridge,
  kernel ridge, RFF ridge, small MLP);
* Fig. 8b - exact-replay source-to-target state matrix and the same-state minus
  cross-state difference Delta S_A (median over the 560 ordered device-specific pairs);
* Fig. 8c - partial disclosure (N_L = 16, 32, 64, 96 of 128 CRPs; five partitions):
  hidden-response attack scores and chance-normalized Top-1 lift;
* Fig. 8d - complete vs partial disclosure summary.

Exact replay is a digital enrollment-database replay benchmark, not a physical clone.
Round B responses were never used for training; hidden challenges were excluded from
model fitting and PCA; cross-state evaluation reuses the frozen source-state model.

Raw mode: ``python -m experiment4_security.identity_credential.cli formal --config configs/formal_reconfiguration.yaml``
followed by ``python -m experiment4_security.cli partial-leakage --config configs/partial_disclosure.yaml``.
"""

from __future__ import annotations

import pandas as pd

from experiment4_security.ml_attack.summaries import ATTACK_LABELS, same_minus_cross_delta, state_matrix
from puf_common.checks import check, open_dataset, report, reproduction_parser

METHOD_ORDER = ["mean_response", "exact_template_replay", "ridge_clone", "kernel_ridge_clone", "random_fourier_ridge_clone", "small_mlp_clone"]
EXPECTED_DELTA = {"exact_template_replay": 0.94, "ridge_clone": 0.80, "kernel_ridge_clone": 0.79, "random_fourier_ridge_clone": 0.64}
EXPECTED = {"n_units": 80, "n_crps": 128, "leak_sizes": [16, 32, 64, 96], "partial_evaluations_per_condition": 400, "hidden_reference_median": 0.93}


def plot(same_state: pd.DataFrame, replay: pd.DataFrame, partial: pd.DataFrame, out) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    data = [same_state.loc[same_state.attack_method == m, "median_S_A"].to_numpy() for m in METHOD_ORDER]
    axes[0].boxplot(data, tick_labels=[ATTACK_LABELS[m] for m in METHOD_ORDER], showfliers=False)
    axes[0].axhline(float(same_state.median_genuine.median()), color="tab:green", ls="--", lw=0.9, label="genuine S_intra median")
    axes[0].set(ylabel="Median attack score $S_A$ per unit", title="Fig. 8a  complete disclosure (128 CRPs)")
    axes[0].tick_params(axis="x", rotation=30, labelsize=7)
    axes[0].legend(fontsize=7)
    im = axes[1].imshow(replay.to_numpy(float), cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set(xticks=range(8), yticks=range(8), xticklabels=replay.columns, yticklabels=replay.index, xlabel="Target state", ylabel="Source state", title="Fig. 8b  exact replay")
    fig.colorbar(im, ax=axes[1], fraction=0.046, label="median $S_A$")
    for m, color in (("Ridge", "tab:blue"), ("Kernel Ridge", "tab:orange"), ("RFF Ridge", "tab:purple"), ("Small MLP", "tab:brown")):
        s = partial[partial.attack_method == m].groupby("N_L")["median_S_A"].median()
        axes[2].plot(s.index, s.values, "o-", color=color, label=m)
    axes[2].axhline(float(partial.genuine_hidden_reference.median()), color="tab:green", ls="--", lw=0.9, label="genuine hidden reference")
    axes[2].set(xlabel="Disclosed CRPs $N_L$ (of 128)", ylabel="Median hidden-response $S_A$", xticks=EXPECTED["leak_sizes"], title="Fig. 8c  partial disclosure")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig8_attacks.png", dpi=200)
    plt.close(fig)


def main() -> int:
    args = reproduction_parser(__doc__.splitlines()[0]).parse_args()
    ds, out = open_dataset(args, "fig8")

    c = ds.read_csv("analysis_ready_data/formal/track_c_same_state_clone_summary.csv")
    d = ds.read_csv("analysis_ready_data/formal/track_d_clone_transfer_matrix_summary.csv")
    partial = ds.read_csv("Source_Data/Fig8/Fig8c_partial_disclosure_unit_evaluations.csv")
    frozen_replay = ds.read_csv("Source_Data/Fig8/Fig8b_exact_replay_state_matrix.csv", index_col=0)

    replay = state_matrix(d[d.attack_method == "exact_template_replay"], "median_S_A")
    deltas = {m: same_minus_cross_delta(d[d.attack_method == m]) for m in METHOD_ORDER if m in set(d.attack_method)}
    delta_summary = pd.DataFrame({"attack_method": list(deltas), "label": [ATTACK_LABELS[m] for m in deltas], "median_delta_S_A": [float(v.median()) for v in deltas.values()], "n_pairs": [len(v) for v in deltas.values()]})
    lift = partial.groupby(["attack_method", "N_L"])["Top1_lift"].median().unstack()
    n_per_condition = partial.groupby(["attack_method", "N_L"]).size()

    c[["device_id", "source_state", "attack_method", "median_S_A", "median_genuine", "rg_software_clone"]].to_csv(out / "fig8a_unit_attack_scores.csv", index=False)
    replay.to_csv(out / "fig8b_exact_replay_state_matrix.csv")
    delta_summary.to_csv(out / "fig8b_delta_S_A.csv", index=False)
    lift.to_csv(out / "fig8c_median_top1_lift.csv")
    partial.groupby(["attack_method", "N_L"])["median_S_A"].median().unstack().to_csv(out / "fig8c_median_hidden_S_A.csv")
    if not args.no_figures:
        plot(c, replay, partial, out)

    checks = [
        check("device-state units per method (Fig. 8a)", int(c.groupby("attack_method").size().min()), EXPECTED["n_units"], 0),
        check("enrolled CRPs per unit (complete disclosure)", int(c.n_positive.max()), EXPECTED["n_crps"], 0),
        check("Fig. 8b matrix equals frozen Source Data", float((replay - frozen_replay.reindex_like(replay)).abs().max().max()), 0.0, 1e-12),
        check("Delta S_A pairs per method", int(delta_summary.n_pairs.min()), 560, 0),
        *[check(f"Delta S_A {ATTACK_LABELS[m]}", round(float(deltas[m].median()), 2), e, 0) for m, e in EXPECTED_DELTA.items()],
        check("Delta S_A equals frozen delta_clone_same_to_cross column", float(max((v - d.loc[v.index, "delta_clone_same_to_cross"]).abs().max() for v in deltas.values())), 0.0, 1e-12),
        check("partial disclosure levels N_L", sorted(partial.N_L.unique().tolist()), EXPECTED["leak_sizes"]),
        check("evaluations per (method, N_L) = 80 units x 5 partitions", int(n_per_condition.min()), EXPECTED["partial_evaluations_per_condition"], 0),
        check("max over conditions of median Top-1 lift (<= 1)", float(lift.max().max()) <= 1.0, True),
        check("genuine hidden-response reference median (~0.93)", round(float(partial.genuine_hidden_reference.median()), 2), EXPECTED["hidden_reference_median"], 0),
    ]
    return 0 if report(checks, out, "Fig. 8 authority checks") else 1


if __name__ == "__main__":
    raise SystemExit(main())
