# Reproducibility

## Environment

* Reference environment for the final analyses and for the smoke test of this repository:
  Python 3.11.15, macOS 15 (Apple silicon); package versions in `requirements.txt`.
  `pyproject.toml` declares the compatible ranges (Python >= 3.10). Linux is expected to work
  but was not tested.
* Installation (either route):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # pinned reference versions
pip install -e .                     # installs the packages under src/ (puf_common, e01, experiment00, ...)
```

or `conda env create -f environment.yml && conda activate fiber-puf`.

* Pure Python plus NumPy / SciPy / pandas / scikit-learn / OpenCV / matplotlib. No GPU, no
  deep-learning framework: the small MLP is `sklearn.neural_network.MLPRegressor`.

## Dataset

Download the Zenodo record [10.5281/zenodo.22267156](https://doi.org/10.5281/zenodo.22267156)
into one directory and point the code at it:

```bash
export PUF_DATA_ROOT=/path/to/zenodo_dataset
python scripts/verify_public_data.py            # 56 checks, ~3 s; add --checksums to hash the lightweight files
```

The archives may stay zipped. Only `Source_Data.zip`, `analysis_ready_data.zip`,
`challenges_calibration_masks.zip` and the top-level CSV / JSON / TXT files (about 70 MB) are
needed for lightweight mode. Nothing is written into the dataset directory; results go to
`outputs/` (or `--output-root` / `$PUF_OUTPUT_ROOT`).

## Lightweight mode (recommended)

```bash
for f in fig3 fig4 fig5 fig6 fig7 fig8 supp_fig_s2 supp_fig_s3 supp_fig_s4 threshold_development; do
  python scripts/reproduce_$f.py
done
```

Each script recomputes the reported quantities from the frozen Source Data and analysis-ready
tables, renders a working figure (`--no-figures` to skip) and exits non-zero if any authority
check fails. Runtime on the reference machine: a few seconds per script; `reproduce_fig4.py`
takes about one minute for the 5000-replicate hierarchical bootstrap (use
`--bootstrap-replicates 200` for a smoke test; the bootstrap-count checks are then skipped).
Each `scripts/reproduce_*.py` script is the authority check for that figure.

## Raw mode

Raw mode decodes the H.264 recordings and repeats preprocessing from pixels. It is slower and
needs storage for the extracted archives (19.12 GiB in total) plus intermediate templates
(several GiB per dataset; the fixed-state pipeline writes ~20 GiB of templates and caches).
Stage the archives first; the script extracts into the git-ignored `data/` folder in the layout
each config expects and writes local metadata tables whose `video_path` column points at the
extracted files:

```bash
python scripts/prepare_raw_workspace.py --data-root $PUF_DATA_ROOT --experiment fixed_state
python scripts/prepare_raw_workspace.py --data-root $PUF_DATA_ROOT --experiment formal --device F01
```

| Dataset | Archives | Command |
|---|---|---|
| Fig. 3 macro-pixel screening | `raw_macro_pixel.zip` | `python -m e01.cli --config configs/macro_pixel_screening.yaml analyze-screening` |
| Fig. 4 fiber-length optimization | `raw_fiber_length.zip` | `python -m experiment00.cli --config configs/fiber_length_optimization.yaml validate` then `analyze` |
| Fig. 5a,b / Fig. 6 fixed state | `raw_fixed_state.zip` | `python -m experiment2.cli validate --config configs/fixed_state_dual_channel.yaml`, then `run`, then `analyze` |
| Fig. 5c-e wavelength / pathway | `raw_wavelength_pathway.zip` | `python -m experiment02b.cli --config configs/wavelength_pathway_control.yaml --stage all` |
| Supplementary Note 7.1 | `raw_threshold_development.zip` | `python -m experiment3.cli run --config configs/threshold_development.yaml` (descriptive tables); `python -m experiment4_security.lifecycle.cli --config configs/lifecycle_threshold_development.yaml --data-root $PUF_DATA_ROOT` ($T_G$, $n_{\mathrm{req}}$, remount events; applies the exclusion manifest) |
| Fig. 7 / Fig. 8a-b / S2 / S3 formal experiment | `raw_formal_F01..F10.zip` | `python -m experiment4_security.identity_credential.cli --config configs/formal_reconfiguration.yaml validate`, then `formal` |
| Fig. 8c-d / S4 partial disclosure | formal run above | set `base_run` in `configs/partial_disclosure.yaml` to the formal run directory, then `python -m experiment4_security.cli partial-leakage --config configs/partial_disclosure.yaml` |

Raw-mode smoke test performed for this release: `raw_fixed_state.zip` (150 recordings) was
decoded with `experiment2.cli run` on the reference machine in about 15 minutes; the resulting
`metrics/short_term_ncc.csv` is identical to the frozen table in `analysis_ready_data`
(maximum absolute difference 0). The formal 80-unit pipeline (20,480 green recordings) takes
many hours and was not rerun for this release; its frozen outputs are the published tables.

## Seeds

| Analysis | Seed(s) |
|---|---|
| Eight-challenge bank C01-C08 | global seed 20260711 (m = 2 stream seed 20260711194) |
| 128-challenge bank `m2_128_v1` | master seed 20260719; C009-C128 via `SeedSequence([20260719, index, attempt])` |
| Fig. 4 hierarchical bootstrap | 20260721, B = 5000 |
| Fig. 5d / 5e device-level cluster bootstrap | 42 / 43, B = 10000 |
| Fixed-state binary PUF bootstrap | 20260729 (grid), B = 5000 |
| Formal experiment / attack models | RFF and MLP seed 20260721; challenge-bank bootstrap 20260721, B = 5000 |
| Partial disclosure splits | 20260801-20260805 (five partitions); bootstrap 42, B = 10000 |
| Wavelength / pathway control | 42, B = 10000 |

## Tolerance policy

Values read back from frozen tables are compared exactly. Deterministic recomputations
(quantiles, medians, counts, Hamming distances) use an absolute tolerance of 1e-9 or 1e-12.
Quantities the manuscript reports rounded are compared after the same rounding. No check
requires bitwise equality of floating-point output across platforms beyond these tolerances;
the only platform-sensitive stage is video decoding, where OpenCV / FFmpeg builds may differ in
the last decoded frame count (the pipelines trim fixed time margins and aggregate by median).

## Authority values checked

| Item | Values |
|---|---|
| Fig. 3 | $m = 2$ selected; $G(2) = 0.5595$, $G(4) = 0.5669$ |
| Fig. 4 | $L = 9$ cm; $G_{\mathrm{min}}(9\,\mathrm{cm}) = 0.3897924079274487$; bootstrap 4991/5000; representative F01/C01 NCC = 0.8205578290959264; $H_{\mathrm{PSD}}$ complementary only |
| Fig. 5 | red 0.94, green 0.77 short-term repeatability; Fig. 5d/e CIs equal frozen tables |
| Fig. 6 | red AUC 0.953651, EER 0.064286, $RG_R$ -0.204386030272718; Top-1 15/15, 119/120, joint 14/15; HD means 0.202 / 0.512 / 0.489 |
| Fig. 7 | 77 Valid / 3 Partial / 0 Failed; $RG_C$ median 0.363; revocation margin 0.386; red medians -0.31 / -0.39 / -6.17; global red EER 0.096; held-out AUC 0.9834, EER 0.0996 |
| Fig. 8 | $\Delta S_A$ 0.94 / 0.80 / 0.79 / 0.64; $N_L$ = 16/32/64/96; Top-1 lift <= 1 |
| Supplementary Note 7.1 | $T_G$ = 0.12899641700197656, $n_{\mathrm{req}}$ = 7; 9/15, 12/15, 12/15; 26/30; 19/20 |
