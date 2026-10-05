#!/usr/bin/env python3
"""Fig. 5 - dual-channel readout: short-term repeatability and the wavelength / injection-pathway control.

Lightweight mode:

* Fig. 5b - device-level short-term NCC of the red (axial, 650 nm) and green
  (lateral, 532 nm) channels across 15 devices
  (``analysis_ready_data/fixed_state/metrics/short_term_ncc.csv``; each device value
  is the median of three pairwise temporal-window NCCs).
* Fig. 5d - radial autocorrelation FWHM per condition (wavelength x injection geometry),
  condition median across five devices with a device-level cluster bootstrap
  (B = 10000, seed 42).
* Fig. 5e - two-dimensional PSD centroid f_c per condition, same bootstrap with seed 43.

Raw mode (decode ``raw_wavelength_pathway.zip`` / ``raw_fixed_state.zip``) is driven by
``python -m experiment02b.cli run --config configs/wavelength_pathway_control.yaml`` and
``python -m experiment2.cli analyze --config configs/fixed_state_dual_channel.yaml``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiment02b.publication_figure import BOOTSTRAP_N, CONDITIONS, cluster_bootstrap_stat
from puf_common.checks import check, open_dataset, report, reproduction_parser

SEED_FIG5D, SEED_FIG5E = 42, 43
EXPECTED = {"red_mean_repeatability": 0.94, "green_mean_repeatability": 0.77}


def condition_bootstrap(values: pd.DataFrame, col: str, seed: int, rng=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed) if rng is None else rng
    rows = []
    for wl, geom in CONDITIONS:
        v = values[(values.wavelength_nm == wl) & (values.injection_geometry == geom)].sort_values("device_id")[col].to_numpy(float)
        lo, hi, n_ok = cluster_bootstrap_stat(v, np.median, rng)
        rows.append({"wavelength_nm": wl, "injection_geometry": geom, "n_devices": len(v), "condition_median": float(np.median(v)), "CI95_lower": lo, "CI95_upper": hi, "bootstrap_N": BOOTSTRAP_N, "bootstrap_n_ok": n_ok, "random_seed": seed})
    return pd.DataFrame(rows)


def plot(rep: pd.DataFrame, fwhm: pd.DataFrame, fc: pd.DataFrame, out) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for ch, color in (("red", "tab:red"), ("green", "tab:green")):
        sub = rep[rep.channel == ch]
        axes[0].plot(sub.device_id, sub.median_ncc, "o", color=color, label=f"{ch} (mean {sub.median_ncc.mean():.2f})")
    axes[0].set(xlabel="Device", ylabel="Short-term NCC (device median)", ylim=(0, 1.02), title="Fig. 5b  short-term repeatability")
    axes[0].tick_params(axis="x", rotation=90, labelsize=7)
    axes[0].legend(fontsize=8)
    for ax, table, ylabel, title in ((axes[1], fwhm, "Radial ACF FWHM (pixels)", "Fig. 5d"), (axes[2], fc, "PSD centroid $f_c$ (cycles/px)", "Fig. 5e")):
        for wl, color in ((532, "tab:green"), (650, "tab:red")):
            t = table[table.wavelength_nm == wl]
            x = np.array([0, 1]) + (0.05 if wl == 650 else -0.05)
            ax.errorbar(x, t.condition_median, yerr=[t.condition_median - t.CI95_lower, t.CI95_upper - t.condition_median], fmt="o-", color=color, capsize=3, label=f"{wl} nm")
        ax.set(xticks=[0, 1], xticklabels=["axial", "lateral"], ylabel=ylabel, title=f"{title}  condition median, 95% CI")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig5_dual_channel_readout.png", dpi=200)
    plt.close(fig)


def main() -> int:
    args = reproduction_parser(__doc__.splitlines()[0]).parse_args()
    ds, out = open_dataset(args, "fig5")

    st = ds.read_csv("analysis_ready_data/fixed_state/metrics/short_term_ncc.csv")
    rep = st[st.record_type.isin(["red_before", "green_challenge"])].groupby(["device_id", "channel"], as_index=False)["median_ncc"].median()
    red, green = (rep.loc[rep.channel == c, "median_ncc"].mean() for c in ("red", "green"))

    per_video = ds.read_csv("Source_Data/Fig5/Fig5d_per_video_radial_ACF_FWHM.csv")
    device_fwhm = per_video.groupby(["wavelength_nm", "injection_geometry", "device_id"], as_index=False)["radial_ACF_FWHM_px"].median()
    fwhm = condition_bootstrap(device_fwhm, "radial_ACF_FWHM_px", SEED_FIG5D)

    psd = ds.read_csv("analysis_ready_data/wavelength_pathway/figures/plotting_data_psd.csv")
    psd = psd[psd.kind == "fiber_condition"].rename(columns={"excitation_geometry": "injection_geometry", "fiber_id": "device_id"})
    # The original PSD panel shares Generator(43) with the preceding ACF panel.
    rng_e = np.random.default_rng(SEED_FIG5E)
    condition_bootstrap(psd, "acf_fwhm_px_median", SEED_FIG5E, rng_e)
    fc = condition_bootstrap(psd, "psd_centroid_cyc_per_px_median", SEED_FIG5E, rng_e)

    rep.to_csv(out / "fig5b_device_short_term_ncc.csv", index=False)
    fwhm.to_csv(out / "fig5d_condition_bootstrap_CI.csv", index=False)
    fc.to_csv(out / "fig5e_condition_bootstrap_CI.csv", index=False)
    if not args.no_figures:
        plot(rep, fwhm, fc, out)

    ref_d = ds.read_csv("Source_Data/Fig5/Fig5d_condition_bootstrap_CI.csv")
    ref_e = ds.read_csv("Source_Data/Fig5/Fig5e_condition_bootstrap_CI.csv")
    cols = ["CI95_lower", "CI95_upper", "condition_median"]
    checks = [
        check("Fig. 5b red mean short-term repeatability (15 devices)", round(float(red), 2), EXPECTED["red_mean_repeatability"], 0),
        check("Fig. 5b green mean short-term repeatability (15 devices)", round(float(green), 2), EXPECTED["green_mean_repeatability"], 0),
        check("Fig. 5d bootstrap CI max |difference| vs frozen table", float(np.abs(fwhm[cols].to_numpy() - ref_d[cols].to_numpy()).max()), 0.0, 1e-9),
        check("Fig. 5e bootstrap CI max |difference| vs frozen table", float(np.abs(fc[cols].to_numpy() - ref_e[cols].to_numpy()).max()), 0.0, 1e-9),
    ]
    return 0 if report(checks, out, "Fig. 5 authority checks") else 1


if __name__ == "__main__":
    raise SystemExit(main())
