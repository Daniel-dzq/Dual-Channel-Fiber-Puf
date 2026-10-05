# Publication notation

Visible figure labels follow the manuscript revision dated 2 October 2026.

| Figure | Displayed term or symbol | Meaning |
|---|---|---|
| 5d | Radial ACF FWHM (pixels) | Radial autocorrelation full width at half maximum |
| 5d | Individual fibers | Device-level medians of three technical repeats |
| 5d | Median (95% bootstrap CI) | Median across five fibers with device bootstrap interval |
| 7b | matched response; S_intra | Same-device, same-state, same-challenge response comparisons |
| 8a,d | matched q_G; Q_5%–Q_95% | 5th–95th percentiles of matched unit-level median credential scores |
| 8c | matched reference median | Matched hidden-response reference median |
| 8a–d | S_A | Attack score |
| 8b | Delta S_A | Same-state minus cross-state attack-score difference |
| 8c | N_L | Number of disclosed CRPs out of 128 |

Existing CSV column names and internal identifiers containing `genuine` are compatibility
aliases for the corresponding matched comparisons. They do not identify a different score,
normalization or analysis. In particular, `genuine_score_median` denotes matched median
`q_G`, and `hidden_genuine_score_median` denotes its hidden-challenge reference. Internal
filenames remain stable so the numerical reconstruction commands and archived tables stay
compatible. Figure labels use the publication terminology.

