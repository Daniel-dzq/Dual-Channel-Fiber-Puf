# Preprocessing by dataset

The six datasets were acquired under different protocols and are processed by different
pipelines. The steps below are what the frozen code does; they are not interchangeable and the
public configs in `configs/` correspond one-to-one to these workflows.

Shared building blocks (`src/puf_common/`):

* **Envelope normalization** `envelope.local_ratio_detail`: detail = I / (G_sigma * I + eps) - 1
  with sigma = 42 px and eps = 1 (Gaussian blur, reflect border). Applied to the green channel.
* **Common-component removal** (`detail_cm`): subtract the mean detail image of a defined
  enrollment set; which set differs by dataset (below).
* **Zero-mean normalized cross-correlation** `ncc.zero_mean_ncc`: Pearson correlation of the
  two masked images, range [-1, 1]; the sign is kept, no absolute value, no image registration or
  shift search anywhere in the pipelines.
* **Valid-pixel masks** are dataset-specific files from `challenges_calibration_masks.zip`
  (`masks/MASKS.json`), never recomputed at analysis time.

| Dataset (figure) | Pipeline | Frames used | Dark | Mask | Common component | Score |
|---|---|---|---|---|---|---|
| Macro-pixel screening (Fig. 3) | `e01` | three fixed time windows, mean frame per window | `calibration/dark_reference_2048x1536.npz` (built from `dark/dark.mp4`), clipped at 0 | `exp01_exp02_valid_mask.png` | mean over the eight challenge templates of the same macro-pixel size | green NCC on `detail_cm`; $G(m)=Q_{5\%}(S_{\mathrm{intra}})-Q_{95\%}(S_{\mathrm{inter},c})$ |
| Fiber-length optimization (Fig. 4) | `experiment00` | trim 10 s head / tail, three temporal blocks, median | per-(length, fiber) dark recordings, mean frame | `exp00_global_valid_mask.npz` | green: mean over challenges within (length, fiber, round, block); red: none | cross-round (A vs B) NCC; $S_{\mathrm{intra}}$, $S_{\mathrm{inter},c}$, $S_{\mathrm{inter},d}$; $G_{\mathrm{min}}(L)$ |
| Fixed-state dual channel (Fig. 5a,b; Fig. 6) | `experiment2` | discard 10 s head / tail, three temporal windows (median), recording median | `calibration/dark_reference_2048x1536.npz` subtracted, clipped at 0 | `exp01_exp02_valid_mask.png` (per-device masks derived in `run`) | mean over the eight challenges of the device (green only) | green NCC on `detail_cm`; 2048-bit binary codes (sign of `detail_cm` on a shared grid); red 9-D Fiber-ID on the aggregated red intensity image, $q_R$ = -||z_before - z_after||_2 after standardization |
| Wavelength x pathway control (Fig. 5c-e) | `experiment02b` | 5 s margins, up to 60 sampled frames, three blocks | none; edge-median background (edge fraction 0.08) | 512 px crop, 384 px analysis window | none | radial autocorrelation FWHM, PSD centroid; device-level cluster bootstrap |
| Threshold development (Supplementary Note 7.1) | `experiment3`, `experiment4_security.lifecycle` | discard 10 s head / tail, three blocks, median | none | `exp03_valid_mask.npz` | mean over the eight challenge representatives of the same device, state and round (Round A and Round B use separate commons) | green NCC on `detail_cm`; $T_G$ from the development-device EER; red 9-D Fiber-ID |
| Formal mechanical reconfiguration and attacks (Fig. 7-8, S2-S4) | `experiment4_security.identity_credential`, `ml_attack` | all decoded frames of the ~8 s clip, median | none | `exp04_formal_valid_mask.npz` | Round-A enrollment mean over C001-C128 of the source state, applied unchanged to Round B and to every cross-state query (no query-side re-estimation) | green NCC on `detail_cm`; $RG_C$, $RG_{\mathrm{device}}$, $RG_{\mathrm{state}}$, Top-1, EER; red 9-D Fiber-ID standardized on F01-F05, held-out F06-F10 |

Notes

* The threshold-development and formal datasets use different common-component protocols
  (per device-state session of eight challenges vs. Round-A enrollment bank of 128 challenges);
  their thresholds are not transferable. $T_G$ and $n_{\mathrm{req}}$ classify remount sessions in
  Supplementary Note 7.1 only; the Fig. 7 unit classes (Valid / Partial / Failed) come from
  $RG_C$ > 0.05, Top-1 >= 0.90 and EER <= 0.10 over the 128 challenges of each unit
  (`ml_attack.track_a_database_auth.database_authentication_conclusion`).
* The red Fiber-ID (`puf_common.fiber_id`) has nine active features (spectral centroid,
  bandwidth, entropy, three PSD band-energy ratios, normalized speckle contrast, gradient mean,
  Laplacian energy). Stored tables carry four zero-valued `acf_*` columns for storage
  compatibility only.
* PSD entropy $H_{\mathrm{PSD}}$ (Fig. 4e) characterizes spatial-frequency content and is not part of the
  fiber-length selection criterion.
* Recordings listed in `data_quality_exclusions.csv` remain in the raw archives; the analysis
  code excludes them from the independent-pair analyses where the manifest says so
  (16 S1 records in Supplementary Note 7.1; the F02/M1/C099 Round A/B pair, hence 127
  independent pairs for that unit).
