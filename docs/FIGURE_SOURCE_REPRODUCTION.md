# Figure source-data reproduction

The publication tables for Fig. 4e, Fig. 7, Fig. 8 and Supplementary Figs. S2–S4
can be rendered without laboratory paths or intermediate response caches:

```bash
python scripts/render_credential_figures.py --source-data SOURCE_DATA --output credential_figures
python scripts/render_attack_figures.py --source-data SOURCE_DATA --output attack_figures
```

Both entry points read publication-facing CSV tables and write vector PDF, editable
SVG and 400 dpi PNG files. The output directory must be outside the input directory.
The plotted symbols follow the manuscript: $q_G$, $q_R$, $RG_C$,
$RG_{\mathrm{device}}$, $RG_{\mathrm{state}}$, $S_A$, $\Delta S_A$, $N_L$ and
$d_{\mathrm{PCA}}$.

For Fig. 4e, the full resampling distribution is reproducible from the pair table:

```bash
python scripts/reproduce_length_bootstrap.py \
  --pair-scores SOURCE_DATA/Fig4c_cross_round_pair_scores.csv \
  --output length_bootstrap
```

The raw-video entry point reads only the canonical package manifest, mask and
400 green recordings plus 25 matched dark recordings:

```bash
python scripts/reproduce_length_from_raw.py --data-root DATA --output length_raw
python scripts/reproduce_length_bootstrap.py --pair-scores length_raw/pair_scores.csv --output length_bootstrap
```

Raw reconstruction requires an empty output directory. It verifies video hashes
and decoded frame counts against the manifest and records the mask hash and runtime
versions. The point estimator uses the 5th percentile of genuine scores minus the
larger of the two 95th percentiles of inter-challenge and inter-device scores.

The bootstrap samples devices and challenges with replacement while retaining
integer pair multiplicities and original identity distinctions. Draws with an
empty score class have undefined margins. The 5,000 attempted draws include 4,967
with defined margins at every length; 9 cm wins 4,956, 7 cm wins 10 and 11 cm wins
one of these complete draws. The 33 incomplete draws remain in the output table.

These figure rendering and fiber-length entry points do not establish completion
of the separate full-dataset raw-to-result reproducibility audit.
