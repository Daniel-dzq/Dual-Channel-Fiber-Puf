#!/usr/bin/env python3
"""Supplementary Fig. S2 - state-resolved challenge and device discrimination after reconfiguration.

* (a) challenge robust gap RG_C and (b) closed-set Top-1 for the 80 device-state units
  (``analysis_ready_data/formal/track_a_database_authentication_summary.csv``);
* (c) different-device score S_inter,d per mechanical state: median and Q95 over the
  11,520 ordered different-device comparisons of each state;
* (d) device robust gap RG_device(s) = Q05(S_intra(s)) - Q95(S_inter,d(s)) and the
  state-resolved AUC (``track_s_d_device_mismatch_summary.csv``).
"""

from __future__ import annotations

import numpy as np

from puf_common.checks import check, open_dataset, report, reproduction_parser

EXPECTED = {"RGC_min": 0.109, "RGC_max": 0.551, "n_per_state": 11520, "RG_device_min": 0.60, "RG_device_max": 0.69, "AUC_min": 0.9999, "partial_device": "F06"}


def plot(a, sd, out) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, col, title in ((axes[0], "rg_challenge", "(a) RG$_C$"), (axes[1], "top1", "(b) Top-1")):
        m = a.pivot(index="device_id", columns="state_id", values=col)
        im = ax.imshow(m.to_numpy(float), cmap="viridis", aspect="auto")
        ax.set(xticks=range(8), yticks=range(10), xticklabels=m.columns, yticklabels=m.index, title=title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[2].errorbar(sd.state_id, sd.median_S_D, yerr=[np.zeros(len(sd)), sd.q95_S_D - sd.median_S_D], fmt="o", color="tab:gray", capsize=3)
    axes[2].axhline(0, color="0.7", lw=0.8)
    axes[2].set(title="(c) S$_{inter,d}$ median / Q95 per state", ylabel="NCC")
    axes[3].bar(sd.state_id, sd.rg_device, color="tab:blue")
    axes[3].set(title="(d) RG$_{device}$(s)", ylim=(0, 0.8))
    fig.tight_layout()
    fig.savefig(out / "supp_fig_s2.png", dpi=200)
    plt.close(fig)


def main() -> int:
    args = reproduction_parser(__doc__.splitlines()[0]).parse_args()
    ds, out = open_dataset(args, "supp_fig_s2")

    a = ds.read_csv("analysis_ready_data/formal/track_a_database_authentication_summary.csv")
    sd = ds.read_csv("analysis_ready_data/formal/track_s_d_device_mismatch_summary.csv")
    frozen = ds.read_csv("Source_Data/FigS2/FigS2c_S_inter_d_by_state_summaries.csv").set_index("state")

    a[["device_id", "state_id", "rg_challenge", "top1", "eer_challenge", "state_conclusion"]].to_csv(out / "s2ab_unit_metrics.csv", index=False)
    sd.to_csv(out / "s2cd_device_mismatch_by_state.csv", index=False)
    if not args.no_figures:
        plot(a, sd, out)

    partial_devices = sorted(a.loc[a.state_conclusion.str.endswith("PARTIAL"), "device_id"].unique())
    checks = [
        check("all 80 RG_C positive", bool((a.rg_challenge > 0).all()), True),
        check("RG_C minimum (display)", round(float(a.rg_challenge.min()), 3), EXPECTED["RGC_min"], 0),
        check("RG_C maximum (display)", round(float(a.rg_challenge.max()), 3), EXPECTED["RGC_max"], 0),
        check("Partial units all from one device", "/".join(partial_devices), EXPECTED["partial_device"]),
        check("different-device comparisons per state", int(sd.n_S_D.min()), EXPECTED["n_per_state"], 0),
        check("S2(c) medians equal frozen Source Data", float(np.abs(sd.set_index("state_id").median_S_D - frozen.median_S_inter_d).max()), 0.0, 1e-12),
        check("S2(c) Q95 equal frozen Source Data", float(np.abs(sd.set_index("state_id").q95_S_D - frozen.Q95_S_inter_d).max()), 0.0, 1e-12),
        check("RG_device minimum (display)", round(float(sd.rg_device.min()), 2), EXPECTED["RG_device_min"], 0),
        check("RG_device maximum (display)", round(float(sd.rg_device.max()), 2), EXPECTED["RG_device_max"], 0),
        check("state-resolved AUC minimum (display)", round(float(sd.auc_device.min()), 4), EXPECTED["AUC_min"], 0),
    ]
    return 0 if report(checks, out, "Supplementary Fig. S2 authority checks") else 1


if __name__ == "__main__":
    raise SystemExit(main())
