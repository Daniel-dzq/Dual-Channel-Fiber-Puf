# Raw-data analysis workflows

The data package uses MANIFEST.csv, raw_data/, masks/, metadata/, and
challenge_library/. Run the commands from this repository, using Python 3.12.14
and requirements-reproduction.txt for this workflow. The historical
requirements.txt describes a different environment. Floating-point results can vary with numerical-library and platform versions.

Set DATA to the absolute extracted package directory and RUN to a new output
directory. Set PYTHONPATH to this repository's src directory. No laboratory data
paths are required by the following entry points:

```bash
python scripts/prepare_manifest_workspace.py --data-root "$DATA" --output-root "$RUN/workspace"
python scripts/reproduce_fig2.py --data-root "$DATA" --output-root "$RUN/fig2"
python scripts/verify_challenge_and_model_contract.py --data-root "$DATA" --output "$RUN/challenge_model_contract.json"
python -m e01.cli analyze-screening --config "$RUN/workspace/macro_pixel.yaml"
python scripts/export_macro_pixel_source.py --config "$RUN/workspace/macro_pixel.yaml" --output "$RUN/fig3"
python scripts/reproduce_length_from_raw.py --data-root "$DATA" --output "$RUN/length" --include-spatial --workers 1
python -m experiment02b.cli --config "$RUN/workspace/wavelength_pathway.yaml" --stage all
python scripts/reproduce_radial_acf.py --config "$RUN/workspace/wavelength_pathway.yaml"
python scripts/export_wavelength_source.py --raw-output "$RUN/workspace/wavelength_pathway/outputs" --output "$RUN/fig5"
python -m experiment2.cli run --config "$RUN/workspace/fixed_state.yaml"
python -m experiment2.cli analyze --config "$RUN/workspace/fixed_state.yaml"
# ANALYSIS_RUN is the new fixed_state_analysis/runs/... directory.
python scripts/export_fixed_state_source.py --raw-output "$RUN/workspace/fixed_state/outputs" --analysis-run "$ANALYSIS_RUN" --output "$RUN/fig6"
python scripts/export_representative_fields.py --data-root "$DATA" --workspace "$RUN/workspace" --output "$RUN/representative_fields"
python scripts/export_length_representatives.py --data-root "$DATA" --pair-scores "$RUN/length/pair_scores.csv" --selection "$DATA/Source_Data/Fig4/Fig4b_display_identities.csv" --output "$RUN/length_representatives"
python scripts/reproduce_threshold_from_raw.py --data-root "$DATA" --output "$RUN/threshold" --workers 2
python scripts/reproduce_formal_red_from_raw.py --data-root "$DATA" --output "$RUN/formal_red" --workers 1
python scripts/reproduce_formal_from_raw.py --data-root "$DATA" --output-root "$RUN/formal_green" --workers 6
python scripts/reproduce_cross_device_from_raw.py --data-root "$DATA" --output "$RUN/cross_device" --workers 6
```

Formal green reproduction processes one device at a time, retaining its fresh
response vectors through complete and partial disclosure attacks. It records
input/code fingerprints and recording hashes. Per-device completion explicitly
excludes cross-device green scores and red identity. Model checkpoints consume
additional disk space. Only completed disposable predictors may be removed;
retain source recordings, score tables, and provenance.

The fixed-state analysis includes historical overlapping-window diagnostics.
Fig. 6(f) uses export_fixed_state_source.py: W1+W2 enrollment, W3 query, and
an enrollment-only common component. A diagnostic Top-1 of 1.0 must not replace
the figure's held-out result.

Fig. 5(d) uses the intensity-based radial ACF method of current SI S4.4.
The old detail/Hann/same-size-FFT metric is a separate diagnostic. Fig. 5(e)
retains the original shared random stream: Generator(43), after four preceding
ACF-condition bootstrap calls.

Export numerical source tables from the new runs before running lightweight
checks. Passing those checks alone does not establish full raw reproduction.
Raw-data computation and reconstruction from stored score tables are separate workflows.
