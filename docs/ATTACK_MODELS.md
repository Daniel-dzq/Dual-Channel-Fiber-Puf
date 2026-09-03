# Attack models (Fig. 8, Supplementary Figs. S3-S4)

All attacks operate on the formal 80-unit dataset (10 devices x 8 mechanical states, challenge
bank C001-C128, Round A enrollment / Round B query). Code: `src/experiment4_security/ml_attack/`;
configuration: `configs/formal_green_preprocessing.yaml` (complete disclosure) and
`configs/partial_disclosure.yaml` (partial disclosure). Hyperparameters are fixed in advance
(Supplementary Table S12) and were never selected on Round B or on hidden challenges.

## Threat model

The attacker obtains the server-side enrollment database of one device-state unit: the
Round-A `detail_cm` templates T_s,c (and the enrollment common component) for the disclosed
challenges, together with the challenge patterns. The attacker never has the physical device
and never sees Round B. An attack succeeds to the extent that the generated response
R_hat(c) correlates with the genuine independent Round-B response: the attack score is
S_A = NCC(R_hat(c), Q_c) with Q_c the Round-B `detail_cm` query, compared with the genuine
S_intra distribution and evaluated through the same closed-set Top-1 / robust-gap machinery as
legitimate authentication (`batch_eval.py`).

## Complete disclosure (all 128 CRPs; Fig. 8a-b, Fig. S3)

| Method (`attack_method`) | Construction | Fixed hyperparameters |
|---|---|---|
| Mean-response baseline (`mean_response`) | R_hat(c) = mean of the disclosed templates; challenge-ignorant | - |
| Exact replay (`exact_template_replay`) | R_hat(c) = T_s,c: the stored enrollment template is replayed verbatim | - |
| Ridge (`ridge_clone`) | linear map from the challenge pattern to the PCA coefficients of the template | alpha = 10 |
| Kernel ridge (`kernel_ridge_clone`) | RBF kernel ridge from pattern to PCA coefficients | alpha = 1, gamma = 0.01 |
| RFF ridge (`random_fourier_ridge_clone`) | random Fourier features of the pattern, then ridge | 256 components, gamma = 0.01, alpha = 1, seed 20260721 |
| Small MLP (`small_mlp_clone`) | `sklearn.neural_network.MLPRegressor` from pattern to PCA coefficients | hidden layers (64, 32), Adam, 150 iterations, learning rate 1e-3, seed 20260721 |

Exact replay is a digital enrollment-database replay benchmark: it measures what an attacker
who copies the stored templates can achieve against the independent Round-B response of the same
state. It is not a physical clone of the fiber.

* Challenge input: the 256 x 256 binary macro-pixel grid of the pattern, average-pooled to a
  32 x 32 bitmap (1024-dimensional; `challenge_features.py`). No measured response enters the
  input features.
* Representation: PCA with d = 64 fitted on the Round-A enrollment templates of the source state
  only (`representations.fit_representation`); predicted coefficients are mapped back to the
  masked `detail_cm` space before scoring.
* Same-state evaluation (Track C, `track_c_clone.py`): models fitted on state M_s are scored
  against the Round-B responses of the same state; Fig. 8a shows the median S_A per unit.
* Cross-state transfer (Track D, `track_d_clone_transfer.py`): the frozen source-state templates /
  models are scored, without any refitting, against the Round-B responses of every other state
  of the same device, always subtracting the source-state enrollment common. Fig. 8b and Fig. S3
  show the 8 x 8 source-to-target matrices (median over devices); Delta S_A is the same-state
  minus cross-state difference over the 560 ordered device-specific state pairs.

## Partial disclosure (N_L of 128 CRPs; Fig. 8c-d, Fig. S4)

`experiment4_security.cli partial-leakage`, `ml_attack/track_c_pl_partial_leakage.py`,
`track_d_pl_transfer.py`, `partial_leakage_splits.py`.

* Disclosure levels N_L = 16, 32, 64, 96; the 128 challenges are grouped into 16 banks of 8 and
  each split discloses 1, 2, 4 or 6 challenges per bank (bank-balanced, nested). Five
  partitions per level with seeds 20260801-20260805; 80 units x 5 partitions = 400 evaluations
  per (method, N_L).
* Training boundary: the enrollment common component, the PCA basis
  (d_PCA = max(1, min(64, N_L - 1))) and every regression model are fitted on the disclosed
  Round-A templates only. Hidden challenges never enter the fit, the PCA or any model selection.
* Evaluation: predicted responses for the N_H = 128 - N_L hidden challenges are scored against
  the genuine Round-B responses of those challenges (hidden-response S_A) and through the
  chance-normalized closed-set Top-1 lift (1 = chance level; values above 1 would indicate
  successful prediction of undisclosed responses). Baselines: mean of the disclosed templates and
  nearest disclosed challenge (normalized Hamming distance between patterns).
* Cross-state transfer (Track D-PL): the frozen N_L-model of the source state is scored against
  hidden-challenge Round-B responses of other states without retraining (Fig. S4a, ridge, N_L = 96).
* Model audit (Fig. S4b-d): retained PCA dimension, residual NCC after removing the common
  predicted-response component, and the fraction of predictions dominated by that common
  component (`common_dominance_flag = HIGH_NCC_DOMINATED_BY_COMMON_RESPONSE`).

## Scope

The results characterize these six response-reconstruction attacks under the stated disclosure
model, the fixed challenge bank and the fixed hyperparameters. They do not establish resistance
against attackers with physical access, adaptive challenge selection, or other model families.
