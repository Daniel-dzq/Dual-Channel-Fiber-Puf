#!/usr/bin/env python3
"""Fig. 3 - SLM macro-pixel screening (m = 1, 2, 4, 8, 16, 32, 64; eight challenges).

Lightweight mode: reads the frozen per-m screening table
``analysis_ready_data/macro_pixel/macro_pixel_summary.csv`` and the m = 2 NCC
matrices in ``Source_Data/Fig3``. The robust separation margin is
G(m) = Q5%(S_intra) - Q95%(S_inter,c) on common-component-removed responses;
m is ranked by G(m) with near-ties (within 5% of the best) resolved toward the
smaller macro-pixel, which is how m = 2 was selected over m = 4.

Raw mode (decode ``raw_macro_pixel.zip``) is driven by
``python -m e01.cli screen --config configs/macro_pixel_screening.yaml`` (see docs/REPRODUCIBILITY.md).
"""

from __future__ import annotations

import numpy as np

from e01.analysis.green_screening import rank_macros
from puf_common.checks import check, open_dataset, report, reproduction_parser

EXPECTED = {"selected_m": 2, "G_m2": 0.5595, "G_m4": 0.5669, "raw_offdiag_min": 0.89, "raw_offdiag_max": 0.94, "processed_offdiag_min": -0.39, "processed_offdiag_max": 0.25}


def plot(summary, raw, processed, out) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for ax, mat, title in ((axes[0], raw, "raw responses"), (axes[1], processed, "after common-component removal")):
        im = ax.imshow(mat.to_numpy(float), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set(xticks=range(8), yticks=range(8), xticklabels=mat.columns, yticklabels=mat.index, title=f"Fig. 3c  m = 2, {title}")
        fig.colorbar(im, ax=ax, fraction=0.046, label="NCC")
    m = summary["macro_pixel"]
    axes[2].plot(m, summary["intra_p05_detail_cm"], "o-", color="tab:green", label="Q5%(S$_{intra}$)")
    axes[2].plot(m, summary["inter_p95_detail_cm"], "s-", color="tab:orange", label="Q95%(S$_{inter,c}$)")
    axes[2].plot(m, summary["robust_gap_detail_cm"], "^-", color="k", label="G(m)")
    axes[2].set(xscale="log", xticks=m, xticklabels=m, xlabel="Macro-pixel size m", ylabel="NCC", title="Fig. 3e  G(m)")
    axes[2].minorticks_off()
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig3_macro_pixel_screening.png", dpi=200)
    plt.close(fig)


def main() -> int:
    args = reproduction_parser(__doc__.splitlines()[0]).parse_args()
    ds, out = open_dataset(args, "fig3")

    summary = ds.read_csv("analysis_ready_data/macro_pixel/macro_pixel_summary.csv").sort_values("macro_pixel")
    raw = ds.read_csv("Source_Data/Fig3/Fig3c_m2_raw_NCC_matrix.csv", index_col=0)
    processed = ds.read_csv("Source_Data/Fig3/Fig3c_m2_processed_NCC_matrix.csv", index_col=0)

    ranked = rank_macros([dict(r, quality_pass=1) for r in summary.to_dict("records")])
    selected = next(int(r["macro_pixel"]) for r in ranked if r["recommended"])
    gm = summary.set_index("macro_pixel")["robust_gap_detail_cm"]
    off = ~np.eye(8, dtype=bool)
    raw_off, proc_off = raw.to_numpy(float)[off], processed.to_numpy(float)[off]

    summary[["macro_pixel", "n_intra_detail_cm", "n_inter_detail_cm", "intra_p05_detail_cm", "inter_p95_detail_cm", "robust_gap_detail_cm"]].to_csv(out / "fig3e_G_of_m.csv", index=False)
    if not args.no_figures:
        plot(summary, raw, processed, out)

    checks = [
        check("selected macro-pixel m", selected, EXPECTED["selected_m"], 0),
        check("G(m=2) (display 0.56)", round(float(gm[2]), 4), EXPECTED["G_m2"], 0),
        check("G(m=4) (display 0.57)", round(float(gm[4]), 4), EXPECTED["G_m4"], 0),
        check("G(4) - G(2) within near-tie band (<= 0.01)", float(gm[4] - gm[2]) <= 0.01, True),
        check("m=2 raw inter-challenge NCC min (display)", round(float(raw_off.min()), 2), EXPECTED["raw_offdiag_min"], 0),
        check("m=2 raw inter-challenge NCC max (display)", round(float(raw_off.max()), 2), EXPECTED["raw_offdiag_max"], 0),
        check("m=2 processed inter-challenge NCC min (display)", round(float(proc_off.min()), 2), EXPECTED["processed_offdiag_min"], 0),
        check("m=2 processed inter-challenge NCC max (display)", round(float(proc_off.max()), 2), EXPECTED["processed_offdiag_max"], 0),
    ]
    return 0 if report(checks, out, "Fig. 3 authority checks") else 1


if __name__ == "__main__":
    raise SystemExit(main())
