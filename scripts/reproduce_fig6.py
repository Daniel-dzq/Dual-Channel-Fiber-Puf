#!/usr/bin/env python3
"""Fig. 6 - fixed-state dual-channel authentication (15 devices, eight challenges, three temporal windows).

Lightweight mode recomputes, from the frozen Source Data tables:

* Fig. 6c/d - Hamming distances of the 2048-bit binary responses (within-class,
  different challenge, different device) and the binary PUF metrics;
* Fig. 6e - red identity scores q_R of the frozen 9-D Fiber-ID (15 same-device,
  210 different-device comparisons): ROC, AUC, EER (501-threshold sweep) and the
  robust gap Q05(genuine) - Q95(impostor);
* Fig. 6f - closed-set Top-1 retrieval: red identity (15 devices), green credential
  (15 devices x 8 challenges; W1+W2 enrollment, W3 query) and joint dual-channel
  acceptance.

Raw mode (decode ``raw_fixed_state.zip``) is driven by
``python -m experiment2.cli run --config configs/fixed_state_dual_channel.yaml``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from puf_common.checks import check, open_dataset, report, reproduction_parser
from puf_common.metrics import auc_roc, equal_error_rate_with_threshold, robust_gap

EXPECTED = {
    "HD_intra_mean": 0.202,
    "HD_inter_c_mean": 0.512,
    "HD_inter_d_mean": 0.489,
    "reliability": 0.798,
    "uniqueness": 0.489,
    "diffuseness": 0.512,
    "uniformity": 0.455,
    "bit_aliasing": 0.455,
    "red_auc": 0.953651,
    "red_eer": 0.064286,
    "red_robust_gap": -0.204386030272718,
    "red_top1": "15/15",
    "green_top1": "119/120",
    "joint": "14/15",
}


def plot(hd: pd.DataFrame, genuine, impostor, roc: pd.DataFrame, green: pd.DataFrame, out) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for cls, color, label in (("HD_intra", "tab:green", "within-class"), ("HD_inter_c", "tab:orange", "different challenge"), ("HD_inter_d", "tab:gray", "different device")):
        axes[0].hist(hd.loc[hd.score_class == cls, "HD"], bins=40, range=(0, 0.7), alpha=0.6, color=color, label=label, density=True)
    axes[0].axvline(0.5, color="k", ls=":", lw=0.8)
    axes[0].set(xlabel="Normalized Hamming distance", ylabel="Density", title="Fig. 6c  2048-bit binary responses")
    axes[0].legend(fontsize=8)

    axes[1].hist(impostor, bins=40, alpha=0.6, color="tab:gray", label="different device", density=True)
    axes[1].hist(genuine, bins=15, alpha=0.7, color="tab:red", label="same device", density=True)
    axes[1].set(xlabel="Red identity score $q_R$", ylabel="Density", title="Fig. 6e  Fiber-ID (9-D) identity scores")
    axes[1].legend(fontsize=8)
    ins = axes[1].inset_axes([0.12, 0.45, 0.35, 0.45])
    ins.plot(roc.FPR, roc.TPR, color="tab:red")
    ins.plot([0, 1], [0, 1], color="0.7", lw=0.7)
    ins.set(xlabel="FPR", ylabel="TPR", xticks=[0, 1], yticks=[0, 1])
    ins.tick_params(labelsize=6)

    top = green[green["rank"] == 1]
    correct = top[top.is_correct]
    axes[2].scatter(top.device_id, top.q_G, s=14, color="0.6", label="Top-1 candidate")
    axes[2].scatter(correct.device_id, correct.q_G, s=14, color="tab:green", label="correct")
    axes[2].set(xlabel="Device", ylabel="Top-1 green score $q_G$", title=f"Fig. 6f  green retrieval {len(correct)}/{len(top)}")
    axes[2].tick_params(axis="x", rotation=90, labelsize=7)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig6_fixed_state_authentication.png", dpi=200)
    plt.close(fig)


def main() -> int:
    args = reproduction_parser(__doc__.splitlines()[0]).parse_args()
    ds, out = open_dataset(args, "fig6")

    hd = ds.read_csv("Source_Data/Fig6/Fig6c_Hamming_distance_scores.csv")
    hd_mean = hd.groupby("score_class")["HD"].mean()
    binary = ds.read_csv("analysis_ready_data/fixed_state/figure_data/figure3e_standard_puf_metrics.csv").set_index("metric")["estimate"]

    red = ds.read_csv("Source_Data/Fig6/Fig6e_red_identity_scores.csv")
    genuine = red.loc[red.comparison_class == "same device", "q_R"].to_numpy(float)
    impostor = red.loc[red.comparison_class == "different device", "q_R"].to_numpy(float)
    eer, tau = equal_error_rate_with_threshold(genuine, impostor)
    thresholds = np.sort(np.unique(np.concatenate([genuine, impostor])))
    roc = pd.DataFrame({"threshold": thresholds, "FPR": [float(np.mean(impostor >= t)) for t in thresholds], "TPR": [float(np.mean(genuine >= t)) for t in thresholds]})

    red_ret = ds.read_csv("Source_Data/Fig6/Fig6f_red_retrieval.csv")
    green_ret = ds.read_csv("Source_Data/Fig6/Fig6f_green_retrieval.csv")
    joint = ds.read_csv("Source_Data/Fig6/Fig6f_joint_retrieval.csv")
    red_top1 = red_ret[(red_ret["rank"] == 1) & red_ret.is_correct]
    green_top1 = green_ret[(green_ret["rank"] == 1) & green_ret.is_correct]
    n_red_q, n_green_q = red_ret.query_device.nunique(), len(green_ret[["device_id", "query_challenge"]].drop_duplicates())
    joint_ok = int(joint.joint_correct.sum())
    # joint acceptance requires the correct red identity and all eight green challenges retrieved
    joint_recomputed = int((joint.red_correct & joint.all_8_green_correct).sum())

    summary = {
        "HD_mean": hd_mean.to_dict(),
        "red": {"n_genuine": len(genuine), "n_impostor": len(impostor), "AUC": auc_roc(genuine, impostor), "EER": eer, "EER_threshold": tau, "robust_gap": robust_gap(genuine, impostor)},
        "retrieval": {"red_top1": f"{len(red_top1)}/{n_red_q}", "green_top1": f"{len(green_top1)}/{n_green_q}", "joint": f"{joint_ok}/{len(joint)}"},
    }
    pd.Series(summary).to_json(out / "fig6_summary.json", indent=2)
    roc.to_csv(out / "fig6e_ROC.csv", index=False)
    if not args.no_figures:
        plot(hd, genuine, impostor, roc, green_ret, out)

    checks = [
        check("Fig. 6c mean HD within-class", round(float(hd_mean["HD_intra"]), 3), EXPECTED["HD_intra_mean"], 0),
        check("Fig. 6c mean HD different challenge", round(float(hd_mean["HD_inter_c"]), 3), EXPECTED["HD_inter_c_mean"], 0),
        check("Fig. 6c mean HD different device", round(float(hd_mean["HD_inter_d"]), 3), EXPECTED["HD_inter_d_mean"], 0),
        check("Fig. 6d Reliability", round(float(binary["Reliability"]), 3), EXPECTED["reliability"], 0),
        check("Fig. 6d Uniqueness", round(float(binary["Uniqueness"]), 3), EXPECTED["uniqueness"], 0),
        check("Fig. 6d Diffuseness", round(float(binary["Challenge_diffuseness"]), 3), EXPECTED["diffuseness"], 0),
        check("Fig. 6d Uniformity", round(float(binary["Uniformity"]), 3), EXPECTED["uniformity"], 0),
        check("Fig. 6d Bit-aliasing", round(float(binary["Bit_aliasing"]), 3), EXPECTED["bit_aliasing"], 0),
        check("Fig. 6e red AUC", round(summary["red"]["AUC"], 6), EXPECTED["red_auc"], 0),
        check("Fig. 6e red EER", round(summary["red"]["EER"], 6), EXPECTED["red_eer"], 0),
        check("Fig. 6e red robust gap RG_R", summary["red"]["robust_gap"], EXPECTED["red_robust_gap"]),
        check("Fig. 6f red Top-1", summary["retrieval"]["red_top1"], EXPECTED["red_top1"]),
        check("Fig. 6f green Top-1", summary["retrieval"]["green_top1"], EXPECTED["green_top1"]),
        check("Fig. 6f joint acceptance", summary["retrieval"]["joint"], EXPECTED["joint"]),
        check("Fig. 6f joint == red AND all eight green correct", joint_recomputed, joint_ok, 0),
    ]
    return 0 if report(checks, out, "Fig. 6 authority checks") else 1


if __name__ == "__main__":
    raise SystemExit(main())
