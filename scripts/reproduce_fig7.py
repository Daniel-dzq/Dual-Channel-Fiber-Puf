#!/usr/bin/env python3
"""Fig. 7 - formal mechanical-reconfiguration experiment (10 devices x 8 states = 80 device-state units).

Lightweight mode recomputes, from the frozen formal-run tables in
``analysis_ready_data/formal`` and ``analysis_ready_data/fiber_id_9d`` and from
``Source_Data/Fig7``:

* Fig. 7a - unit quality classes (Valid / Partial / Failed) from RG_C, Top-1 and EER
  of each unit (128 challenges; the F02/M1/C099 non-independent Round A/B pair is
  excluded from that unit's independent-pair summary, hence n = 127 there);
* Fig. 7b - green score hierarchy: Q05 of the 80 unit-level S_intra medians and Q95
  of the S_inter,c (80 unit medians), S_inter,d (8 state medians) and S_inter,s
  (560 source-target pair medians) distributions;
* Fig. 7c - same-state median (diagonal) vs cross-state median (off-diagonal) and the
  minimum cross-state revocation margin RG_state;
* Fig. 7d - red identity scores of the 9-D Fiber-ID: medians for same-device/same-state,
  same-device/cross-state and different-device pairs, global AUC/EER (genuine =
  same device across states, impostor = different device) and the held-out
  feature-standardization evaluation subset (F06-F10).

The Round-A common component is estimated from enrollment responses only and applied
unchanged to Round B and to cross-state queries. T_G and n_req from the eight-challenge
threshold-development dataset are not used here.

Raw mode (decode ``raw_formal_F01..F10.zip``) is driven by
``python -m experiment4_security.identity_credential.cli formal --config configs/formal_reconfiguration.yaml``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiment4_security.ml_attack.track_a_database_auth import database_authentication_conclusion
from puf_common.checks import check, open_dataset, report, reproduction_parser
from puf_common.metrics import auc_roc, equal_error_rate

HELD_OUT = ["F06", "F07", "F08", "F09", "F10"]
EXPECTED = {
    "n_units": 80, "valid": 77, "partial": 3, "failed": 0,
    "median_RGC": 0.363, "Q05_S_intra": 0.886, "Q95_S_inter_c": -0.038, "Q95_S_inter_d": -0.005, "Q95_S_inter_s": 0.002,
    "same_state_median": 0.925, "cross_state_median": -0.012, "min_revocation_margin": 0.386,
    "red_same_state": -0.31, "red_cross_state": -0.39, "red_different_device": -6.17,
    "red_global_EER": 0.096, "heldout_AUC": 0.9834, "heldout_EER": 0.0996,
}


def plot(units: pd.DataFrame, matrix: pd.DataFrame, red: pd.DataFrame, out) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    colors = units.quality.map({"Valid": "tab:green", "Partial": "tab:orange", "Failed": "tab:red"})
    axes[0].scatter(units.rg_challenge, units.top1, c=colors, s=18)
    axes[0].axvline(0.05, color="0.6", ls=":", lw=0.8)
    axes[0].axhline(0.90, color="0.6", ls=":", lw=0.8)
    axes[0].set(xlabel="Challenge robust gap RG$_C$", ylabel="Top-1 (128 challenges)", title=f"Fig. 7a  {int((units.quality == 'Valid').sum())} Valid / {int((units.quality == 'Partial').sum())} Partial / {int((units.quality == 'Failed').sum())} Failed")
    im = axes[1].imshow(matrix.to_numpy(float), cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set(xticks=range(8), yticks=range(8), xticklabels=matrix.columns, yticklabels=matrix.index, xlabel="Query state", ylabel="Enrollment state", title="Fig. 7c  median S over devices")
    fig.colorbar(im, ax=axes[1], fraction=0.046, label="NCC")
    groups = [("same device, same state", red[red.r0].S_R), ("same device, cross-state", red[red.r1].S_R), ("different device", red[~red.same_device].S_R)]
    axes[2].boxplot([g.to_numpy() for _, g in groups], tick_labels=[f"{n}\n(n={len(g)})" for n, g in groups], showfliers=False)
    axes[2].set(ylabel="Red identity score $q_R$", title="Fig. 7d  9-D Fiber-ID")
    axes[2].tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig7_mechanical_reconfiguration.png", dpi=200)
    plt.close(fig)


def main() -> int:
    args = reproduction_parser(__doc__.splitlines()[0]).parse_args()
    ds, out = open_dataset(args, "fig7")

    a = ds.read_csv("analysis_ready_data/formal/track_a_database_authentication_summary.csv")
    b = ds.read_csv("analysis_ready_data/formal/track_b_template_transfer_matrix_summary.csv")
    sd = ds.read_csv("analysis_ready_data/formal/track_s_d_device_mismatch_summary.csv")
    cross = ds.read_csv("Source_Data/Fig7/Fig7c_cross_state_pair_scores.csv")
    red = ds.read_csv("analysis_ready_data/fiber_id_9d/fig7/pairs_9d.csv")

    units = a[["device_id", "state_id", "n_positive", "rg_challenge", "top1", "eer_challenge", "median_genuine", "median_S_C"]].copy()
    units["quality"] = [database_authentication_conclusion(r.rg_challenge, r.top1, r.eer_challenge).split("_")[-1].capitalize() for r in units.itertuples()]
    counts = units.quality.value_counts().reindex(["Valid", "Partial", "Failed"], fill_value=0)

    off = b[~b.is_diagonal]
    same_state_median = float(a.median_genuine.median())
    cross_state_median = float(cross.S_inter_s_pair_median.median())
    min_margin = float(cross.RG_state.min())
    matrix = b.groupby(["source_state", "target_state"])["median_S_X"].median().unstack()

    genuine, impostor = red.loc[red.r1, "S_R"].to_numpy(float), red.loc[~red.same_device, "S_R"].to_numpy(float)
    ho = red[red.device_a.isin(HELD_OUT) & red.device_b.isin(HELD_OUT)]
    ho_gen, ho_imp = ho.loc[ho.r1, "S_R"].to_numpy(float), ho.loc[~ho.same_device, "S_R"].to_numpy(float)

    units.to_csv(out / "fig7a_unit_quality.csv", index=False)
    matrix.to_csv(out / "fig7c_state_matrix.csv")
    pd.Series({
        "same_state_median": same_state_median, "cross_state_median": cross_state_median, "min_revocation_margin": min_margin,
        "red_global": {"AUC": auc_roc(genuine, impostor), "EER": equal_error_rate(genuine, impostor), "n_genuine": len(genuine), "n_impostor": len(impostor)},
        "red_heldout_F06_F10": {"AUC": auc_roc(ho_gen, ho_imp), "EER": equal_error_rate(ho_gen, ho_imp), "n_genuine": len(ho_gen), "n_impostor": len(ho_imp)},
    }).to_json(out / "fig7_summary.json", indent=2)
    if not args.no_figures:
        plot(units, matrix, red, out)

    checks = [
        check("device-state units", len(units), EXPECTED["n_units"], 0),
        check("F02/M1 unit uses 127 independent pairs", int(units.loc[(units.device_id == "F02") & (units.state_id == "M1"), "n_positive"].item()), 127, 0),
        check("Valid units", int(counts["Valid"]), EXPECTED["valid"], 0),
        check("Partial units", int(counts["Partial"]), EXPECTED["partial"], 0),
        check("Failed units", int(counts["Failed"]), EXPECTED["failed"], 0),
        check("recomputed classes equal frozen state_conclusion", int((units.quality.str.upper() == a.state_conclusion.str.split("_").str[-1]).sum()), 80, 0),
        check("median challenge robust gap RG_C", round(float(units.rg_challenge.median()), 3), EXPECTED["median_RGC"], 0),
        check("Fig. 7b Q05(S_intra) unit medians", round(float(np.quantile(units.median_genuine, 0.05)), 3), EXPECTED["Q05_S_intra"], 0),
        check("Fig. 7b Q95(S_inter,c) unit medians", round(float(np.quantile(units.median_S_C, 0.95)), 3), EXPECTED["Q95_S_inter_c"], 0),
        check("Fig. 7b Q95(S_inter,d) state medians", round(float(np.quantile(sd.median_S_D, 0.95)), 3), EXPECTED["Q95_S_inter_d"], 0),
        check("Fig. 7b Q95(S_inter,s) pair medians", round(float(np.quantile(cross.S_inter_s_pair_median, 0.95)), 3), EXPECTED["Q95_S_inter_s"], 0),
        check("Fig. 7c same-state median", round(same_state_median, 3), EXPECTED["same_state_median"], 0),
        check("Fig. 7c cross-state median", round(cross_state_median, 3), EXPECTED["cross_state_median"], 0),
        check("Fig. 7c minimum revocation margin", round(min_margin, 3), EXPECTED["min_revocation_margin"], 0),
        check("Fig. 7c frozen table equals track B off-diagonal", float(np.abs(np.sort(cross.S_inter_s_pair_median) - np.sort(off.median_S_X)).max()), 0.0, 1e-12),
        check("Fig. 7d red median same device / same state", round(float(red.loc[red.r0, "S_R"].median()), 2), EXPECTED["red_same_state"], 0),
        check("Fig. 7d red median same device / cross-state", round(float(np.median(genuine)), 2), EXPECTED["red_cross_state"], 0),
        check("Fig. 7d red median different device", round(float(np.median(impostor)), 2), EXPECTED["red_different_device"], 0),
        check("Fig. 7d global red EER", round(equal_error_rate(genuine, impostor), 3), EXPECTED["red_global_EER"], 0),
        check("held-out F06-F10 AUC", round(auc_roc(ho_gen, ho_imp), 4), EXPECTED["heldout_AUC"], 0),
        check("held-out F06-F10 EER", round(equal_error_rate(ho_gen, ho_imp), 4), EXPECTED["heldout_EER"], 0),
    ]
    return 0 if report(checks, out, "Fig. 7 authority checks") else 1


if __name__ == "__main__":
    raise SystemExit(main())
