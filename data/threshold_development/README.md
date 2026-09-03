# Threshold-development green pair scores (Supplementary Note 7.1)

Lightweight tables that make the corrected S7.1 operating point (T_G = 0.12899641700197656,
n_req = 7) recomputable without decoding `raw_threshold_development.zip`.

| File | Content |
|------|---------|
| `green_pair_scores_corrected.csv` | Masked zero-mean NCC scores between green responses of the eight-challenge remount dataset (15 devices, states S0-S2, Rounds A/B). `source = frozen_lifecycle` rows come from the frozen lifecycle analysis; `source = corrected_S1_recomputed` rows are the S1 same-state scores of F01-F05 recomputed after removing the 16 invalid S1 copies listed in the public `data_quality_exclusions.csv` (`ACCIDENTAL_S1_COPY_OF_9CM`). |
| `lifecycle_events_corrected.csv` | The 30 remount events (15 devices x S0->S1, S1->S2) evaluated with the corrected T_G / n_req: red identity pass, previous-credential failure, number of passing green challenges, joint acceptance. |

`scripts/reproduce_threshold_development.py` derives T_G, n_req, the k = 8/7/6 session
acceptance (9/15, 12/15, 12/15) and the joint event counts (26/30; 19/20 on F06-F15) from
these tables. The same numbers are obtained in raw mode by
`python -m experiment4_security.lifecycle.cli --config configs/lifecycle_threshold_development.yaml --data-root <zenodo>`.

Note: `analysis_ready_data/fiber_id_9d/lifecycle/events.csv` in the Zenodo record is the
pre-correction event table (k = 6). The frozen operating point of record is
`analysis_ready_data/fiber_id_9d/lifecycle/threshold_development.json`
(`tau_G_frozen`, `n_req_frozen`), which matches the tables here.
