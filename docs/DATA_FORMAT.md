# Data format

The public dataset (Zenodo, DOI [10.5281/zenodo.22267156](https://doi.org/10.5281/zenodo.22267156))
is read by `puf_common.public_data.PublicDataset`. Pass the download directory as `--data-root`
(or `$PUF_DATA_ROOT`); the three lightweight archives may be left zipped or extracted in place.
Nothing is written into that directory.

## Identifiers

| Token | Meaning |
|---|---|
| `F01`-`F15` | Fiber devices. Fixed-state and threshold-development datasets use 15 devices; the formal mechanical-reconfiguration experiment uses `F01`-`F10`; the wavelength / pathway control uses `F1`-`F5` (not zero-padded) |
| `M0`-`M7` | Eight mechanical states of the formal experiment (Fig. 7-8) |
| `S0`-`S2` | Three states of the threshold-development protocol (Supplementary Note 7.1); not the formal protocol |
| `C01`-`C08` | Eight-challenge bank (m = 2), used for Fig. 3-6 and the threshold-development dataset |
| `C001`-`C128` | Formal 128-challenge bank `m2_128_v1`; `C001`-`C008` are byte-identical to `C01`-`C08` |
| Round `A` / `B` | Independent presentations of the same challenge: Round A is enrollment, Round B is the independent query |
| Temporal windows W1-W3 | Three non-overlapping windows within one fixed-state recording (short-term repeatability; W1+W2 enrollment, W3 query for retrieval) |
| `red_before` / `red_after` | Red (650 nm, axial injection) identity recordings taken before and after the green challenge series of a fixed-state session |
| `green` / `C0x` records | Green (532 nm, lateral injection) credential recordings, one per challenge |

## Archives and the code that reads them

| Zenodo component | Content | Read by |
|---|---|---|
| `Source_Data.zip` (`Source_Data/FigN/...`) | Frozen per-panel tables underlying the published figures (pair scores, ROC, retrieval tables, state matrices, bootstrap draws) | all `scripts/reproduce_*.py` |
| `analysis_ready_data.zip` | Pipeline-level tables: `metadata/experiment2_metadata.csv`, `metadata/experiment3_metadata.csv`, `macro_pixel/`, `fiber_length/`, `fixed_state/{metrics,figure_data}`, `wavelength_pathway/`, `threshold_development/`, `fiber_id_9d/`, `formal/track_*`, `attacks/` | reproduction scripts; `scripts/prepare_raw_workspace.py` (metadata) |
| `challenges_calibration_masks.zip` | `challenges/` (128 canvases as PNG, `challenges_exact.npz`, `challenge_manifest.csv`, pairwise Hamming / input-NCC tables, `mp002_meta/`), `calibration/dark_reference_2048x1536.npz`, `masks/` (four valid-pixel masks, `MASKS.json`) | `scripts/verify_public_data.py`; raw-mode configs |
| `data_quality_exclusions.csv` | 81 records: 64 intentional S0 reuses (retained), 16 invalid S1 copies (`EXCLUDE_FROM_INDEPENDENT_S1_ANALYSES`), 1 non-independent formal Round A/B pair (F02/M1/C099) | `experiment4_security.lifecycle.cli`, `scripts/reproduce_threshold_development.py`, `verify_public_data.py` |
| `raw_file_manifest.csv`, `file_manifest.csv`, `final_release_checksums.sha256` | Inventory and SHA-256 of every file | `verify_public_data.py --checksums` |
| `raw_macro_pixel.zip` | `mp001/..mp064/` screening videos and `dark/dark.mp4` | `e01` (extract into `data/raw/macro_pixel/videos/`) |
| `raw_fiber_length.zip` | `videos/<L>cm_F0x_G_<A|B>_C0x.mp4` and per-(length, fiber) darks under `dark/` | `experiment00` |
| `raw_fixed_state.zip` | `RedAndGreen/{red,green}/...` (150 recordings, 15 devices x (8 green + 2 red)) | `experiment2` via the local `experiment2_metadata.csv` |
| `raw_wavelength_pathway.zip` | `M<state>_<axial|lateral>_<red|green>_F<n>.mp4` (60 videos) | `experiment02b` |
| `raw_threshold_development.zip` | `F0x_S<s>_<A|B>_C0x.mp4` (765 videos) | `experiment3`, `experiment4_security.lifecycle` via the local `experiment3_metadata.csv` |
| `raw_formal_F01..F10.zip` | `F0x/M<s>/green/<A|B>/<A|B>_<idx>_C<idx>_M<s>_F0x.mp4` and `F0x/M<s>/red/...` | `experiment4_security.identity_credential` |

`scripts/prepare_raw_workspace.py` extracts the archives into the git-ignored `data/` folder
in exactly the layouts the raw-mode configs expect and rewrites the `video_path` column of the
two metadata tables to the extracted files (`internal_path` gives the path inside the archive).

## Valid-pixel masks and dark reference

| Mask (`masks/`) | Used by |
|---|---|
| `exp01_exp02_valid_mask.png` | Fig. 3 screening; fixed-state dual-channel (Fig. 5a,b; Fig. 6) |
| `exp00_global_valid_mask.npz` | Fig. 4 fiber-length optimization |
| `exp03_valid_mask.npz` | Supplementary Note 7.1 threshold development |
| `exp04_formal_valid_mask.npz` | Fig. 7-8 formal experiment and attacks |

The masks are not interchangeable. `calibration/dark_reference_2048x1536.npz` (`dark_green`,
`dark_red`, 182 frames) is subtracted only in the fixed-state / Fig. 3 workflows; see
[PREPROCESSING.md](PREPROCESSING.md).

## Internal file names inside the analysis-ready package

Some analysis-ready files keep the numbering of the pipeline that produced them:
`fixed_state/figure_data/figure3*_*.csv` are the fixed-state panels of paper Fig. 6
(`figure3e_standard_puf_metrics.csv` = Fig. 6d binary PUF metrics), and
`fiber_id_9d/fig6`, `fig6e`, `fig7`, `lifecycle` follow the paper numbering.
`threshold_development/tau_G.json` (0.1653) is the descriptive development EER threshold of the
`threshold_development` pipeline and is not the Supplementary Note 7.1 operating point T_G = 0.129, which is
stored in `fiber_id_9d/lifecycle/threshold_development.json` (`tau_G_frozen`, `n_req_frozen`).
`fiber_length/` holds the frozen pre-final fiber-length run; the Fig. 4 authority is
`Source_Data/Fig4/`.

## Tables shipped in this repository

`data/challenge_patterns/mp002/mp002_C01..C08.png` are the canonical eight-challenge canvases
(1024 x 768, active 512 x 512 at offset (256, 128)). `data/threshold_development/` holds the
corrected green pair-score and remount-event tables of Supplementary Note 7.1 (see the README
in that folder).
