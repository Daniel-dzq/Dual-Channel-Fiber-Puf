# Dual-channel fiber physical unclonable functions with persistent identity and reconfigurable credentials

Analysis code for the manuscript of the same title. A short optical fiber is read out through
two channels: a red (650 nm, axial injection) channel whose speckle statistics give a
persistent device identity (the nine-feature Fiber-ID), and a green (532 nm, lateral injection)
channel whose challenge-dependent speckle responses form credentials that can be revoked and
re-enrolled by mechanically reconfiguring the fiber. This repository contains the challenge
generation, response preprocessing, zero-mean normalized cross-correlation and PUF-metric
calculations, identity-feature extraction, statistical analysis, figure reproduction and
enrollment-database disclosure attacks used in the paper.

## Workflows

Use the supplied analysis tables for figure-level numerical reconstruction, or run the raw-video pipelines to recalculate responses and scores. See [raw-data workflows](docs/RAW_REPRODUCTION.md). These are separate operations; file-integrity checks do not replace scientific comparison of outputs.

## Paper

*Dual-channel fiber physical unclonable functions with persistent identity and reconfigurable
credentials.* Ziqi Dai, Yaxin Zhang, Lu Kang, Chuanbo Li, Honglian Guo, Min Lv and Xin Tang.
School of Science, Minzu University of China. (Manuscript under review; publication DOI to be
added.)

## Repository

[GitHub repository](https://github.com/Daniel-dzq/Dual-Channel-Fiber-Puf)

## Data

Dataset: Dai et al., data for "Dual-channel fiber physical unclonable functions with persistent
identity and reconfigurable credentials", Zenodo. DOI:
[10.5281/zenodo.22267156](https://doi.org/10.5281/zenodo.22267156).

The publication recording inventory comprises 22,147 labeled H.264 MP4 files, about 19.12 GiB. The manifest-format dataset includes recordings, experimental challenge patterns, masks, analysis tables and figure source data. This repository contains no recordings; the code reads the dataset
from a directory you choose (`--data-root` or `$PUF_DATA_ROOT`). See
[docs/DATA_FORMAT.md](docs/DATA_FORMAT.md).

## Repository structure

```
configs/       one YAML per experiment (paper parameters), plus example_paths.yaml
src/
  puf_common/            zero-mean NCC, envelope normalization, masks, metrics (EER, AUC, robust gap),
                         nine-feature Fiber-ID, dataset access, authority checks
  e01/                   Fig. 3  macro-pixel screening and the eight-challenge bank
  experiment00/          Fig. 4  fiber-length optimization
  experiment2/           Fig. 5a-b, Fig. 6  fixed-state dual-channel authentication
  experiment02b/         Fig. 5c-e  wavelength x injection-pathway control
  experiment3/           Supplementary Note 7.1  threshold-development dataset (descriptive)
  experiment4_security/  Fig. 7-8, S2-S4  formal mechanical reconfiguration, 128-challenge bank,
                         threshold development (lifecycle), response-reconstruction attacks
scripts/       reproduce_fig*.py, reproduce_supp_fig_s*.py, reproduce_threshold_development.py,
               verify_public_data.py, prepare_raw_workspace.py
data/          canonical eight-challenge patterns (mp002) and threshold-development data documentation
docs/          REPRODUCIBILITY, DATA_FORMAT, FIGURE_MAP, PREPROCESSING, ATTACK_MODELS
examples/      minimal_workflow.py
```

`PUBLIC_CODE_MANIFEST.csv` lists size and SHA-256 for every public file except itself.

## Installation

For the raw workflows documented here, use Python 3.12.14 and
`requirements-reproduction.txt`. The older `requirements.txt` and
`environment.yml` preserve the historical environment; do not mix the two dependency sets.

```bash
git clone https://github.com/Daniel-dzq/Dual-Channel-Fiber-Puf.git
cd Dual-Channel-Fiber-Puf
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-reproduction.txt
pip install -e .
```

## Quick start

```bash
export PUF_DATA_ROOT=/path/to/Zenodo_release
python scripts/verify_public_data.py --data-root "$PUF_DATA_ROOT"
python scripts/reproduce_threshold_development.py --data-root "$PUF_DATA_ROOT"
```

Numerical validation results are written to `./outputs/<figure>/` (`--output-root` to change). The lightweight figure checks listed below accept `--data-root`,
`--output-root` and `--no-figures`; raw workflows have their own documented arguments.

## Reproducing figures

The figure commands below read the supplied `Source_Data/` and `processed_data/`
archives. Lightweight mode recomputes the reported quantities from the Source Data and
analysis-ready tables without decoding any video. Raw mode re-runs the preprocessing from the
recordings; it needs the `raw_*.zip` archives and considerably more time and storage
([docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)).

| Paper item | Lightweight command | Principal check |
|---|---|---|
| Fig. 3 | `python scripts/reproduce_fig3.py` | $m = 2$ selected; $G(2) = 0.5595$, $G(4) = 0.5669$ |
| Fig. 4 | `python scripts/reproduce_fig4.py` | $L = 9$ cm; $G_{\mathrm{min}} = 0.3897924079274487$; bootstrap 4956/4967 complete resamples from 5000 draws |
| Fig. 5 | `python scripts/reproduce_fig5.py` | short-term repeatability red 0.94 / green 0.77; Fig. 5d/e bootstrap CIs |
| Fig. 6 | `python scripts/reproduce_fig6.py` | red AUC 0.953651, EER 0.064286; Top-1 15/15, 119/120, joint 14/15 |
| Fig. 7 | `python scripts/reproduce_fig7.py` | 80 units: 77 Valid / 3 Partial / 0 Failed; revocation margin 0.386 |
| Fig. 8 | `python scripts/reproduce_fig8.py` | $\Delta S_A$: replay 0.94, ridge 0.80, kernel ridge 0.79, RFF 0.64; Top-1 lift $\le 1$ |
| Fig. S2 | `python scripts/reproduce_supp_fig_s2.py` | $RG_C$ 0.109-0.551; $RG_{\mathrm{device}}$ 0.60-0.69 |
| Fig. S3 | `python scripts/reproduce_supp_fig_s3.py` | five source-to-target matrices equal the frozen tables |
| Fig. S4 | `python scripts/reproduce_supp_fig_s4.py` | ridge $N_L = 96$ matrix; $d_{\mathrm{PCA}} = 15/31/63/64$ |
| Supplementary Note 7.1 | `python scripts/reproduce_threshold_development.py` | $T_G = 0.134$, $n_{\mathrm{req}} = 6$; 30/30, 20/20 |

The full mapping (configs, required Zenodo files, raw-mode pipelines) is in
[docs/FIGURE_MAP.md](docs/FIGURE_MAP.md).

The canonical source-data-only figure renderers and raw fiber-length reconstruction
commands are documented in [docs/FIGURE_SOURCE_REPRODUCTION.md](docs/FIGURE_SOURCE_REPRODUCTION.md).

## Analysis notes

* The SLM macro-pixel size $m = 2$ (Fig. 3) and the fiber length $L = 9$ cm (Fig. 4) are the
  operating parameters of all later experiments. Fig. 4 selects $L$ by the maximin robust gap

$$
G_{\min}(L)=\min[
Q_{0.05}(S_{\mathrm{intra}})-Q_{0.95}(S_{\mathrm{inter},c}),
\;
Q_{0.05}(S_{\mathrm{intra}})-Q_{0.95}(S_{\mathrm{inter},d})
].
$$

  on cross-round scores, with a hierarchical bootstrap (device, then challenge within device;
  $B = 5000$, seed 20260721). The PSD entropy $H_{\mathrm{PSD}}$ (Supplementary Note 4) is complementary
  spatial-frequency characterization and is not part of the selection criterion.
  Repeated device occurrences and repeated challenge draws retain their multiplicities.
  Inter-device and inter-challenge pairs retain their original distinct-identity definitions.
  A draw with an empty comparison class has an undefined margin. Of 5000 draws,
  4967 have defined margins for every length; selection frequencies use that denominator.
  The 9 cm, 7 cm and 11 cm selection counts are 4956, 10 and 1, respectively.
  `scripts/reproduce_length_bootstrap.py --pair-scores PATH --output DIR` reproduces
  every draw from the publication table `Fig4c_cross_round_pair_scores.csv`.
* Similarity is the signed zero-mean normalized cross-correlation on the masked, envelope-
  normalized, common-component-removed response (range $[-1, 1]$; no absolute value, no image
  registration).
* The threshold-development dataset (Supplementary Note 7.1; 15 devices, states S0-S2, eight
  challenges) and the formal 80-unit experiment (Fig. 7-8; 10 devices, states M0-M7, 128
  challenges) use different common-component protocols. The operating point $T_G = 0.134$ and
  $n_{\mathrm{req}} = 6$ belongs to Note 7.1 only; the Fig. 7 unit classes (Valid / Partial /
  Failed) are defined by $RG_C$, Top-1 and EER of each unit and do not use $T_G$ or
  $n_{\mathrm{req}}$. In the formal experiment the common component is estimated from the
  Round-A enrollment responses only and applied unchanged to Round B and to cross-state
  queries.
* The red identity descriptor (Fiber-ID) has nine active features. Exact replay in Fig. 8 is a
  digital enrollment-database replay benchmark, not a physical clone ([docs/ATTACK_MODELS.md](docs/ATTACK_MODELS.md)).
* Preprocessing differs between datasets and is summarized in [docs/PREPROCESSING.md](docs/PREPROCESSING.md).

## Publication validation

Supplementary Note 7.1 requires the complete 765-recording acquisition, including 120
genuine and 840 inter-challenge development scores. Development-session acceptance
must reproduce 11/15, 14/15 and 15/15 for requirements of eight, seven and six challenges.
The formal experiment requires 128 independent Round-A/Round-B pairs per device–state unit.

Raw-response caches are reusable only when the recording SHA-256, valid-pixel mask,
preprocessing parameters, software versions and cached response hash match. Formal
analysis starts a separate run so that summary checkpoints cannot outlive their inputs.
Table-level figure checks validate derived tables; they do not establish reproduction
from recordings. A complete release additionally requires raw-to-response provenance
and successful isolated reproduction against the manuscript and Supplementary Information.

## Citation

See [CITATION.cff](CITATION.cff). Please cite the manuscript together with the archived dataset
and the archived code version.

The experimental dataset is archived on Zenodo at
[10.5281/zenodo.22267156](https://doi.org/10.5281/zenodo.22267156).

The code version associated with the reported analyses (v1.0.0) is archived on Zenodo at
[10.5281/zenodo.22274472](https://doi.org/10.5281/zenodo.22274472).

## License

This software is released under the MIT License. See [LICENSE](LICENSE).

## Contact

Min Lv (minlv@muc.edu.cn) or Honglian Guo (hlguo@muc.edu.cn), School of Science, Minzu
University of China.
