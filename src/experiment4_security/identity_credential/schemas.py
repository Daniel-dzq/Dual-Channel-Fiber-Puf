"""Frozen schemas, expected inventory, and metric / conclusion vocabularies."""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Devices / states / challenges
# ---------------------------------------------------------------------------

DEVICES: tuple[str, ...] = (
    "F01",
    "F02",
    "F03",
    "F04",
    "F05",
    "F06",
    "F07",
    "F08",
    "F09",
    "F10",
)
DEVELOPMENT_DEVICES: tuple[str, ...] = ("F01", "F02", "F03", "F04", "F05")
HELDOUT_DEVICES: tuple[str, ...] = ("F06", "F07", "F08", "F09", "F10")
STATES: tuple[str, ...] = ("M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7")
ROUNDS: tuple[str, ...] = ("A", "B")
CHALLENGE_ID_START = 1
CHALLENGE_ID_END = 128
N_CHALLENGES = CHALLENGE_ID_END - CHALLENGE_ID_START + 1

N_DEVICES = len(DEVICES)
N_STATES = len(STATES)
N_GREEN_VIDEOS_EXPECTED = N_DEVICES * N_STATES * 2 * N_CHALLENGES  # 20480
N_RED_VIDEOS_EXPECTED = N_DEVICES * N_STATES * 2  # 160  (before + after)
N_TOTAL_VIDEOS_EXPECTED = N_GREEN_VIDEOS_EXPECTED + N_RED_VIDEOS_EXPECTED  # 20640
N_AB_PAIRS_EXPECTED = N_DEVICES * N_STATES * N_CHALLENGES  # 10240

EXPECTED_INVENTORY: dict[str, int] = {
    "n_devices": N_DEVICES,
    "n_states_per_device": N_STATES,
    "n_green_round_a_per_state": N_CHALLENGES,
    "n_green_round_b_per_state": N_CHALLENGES,
    "n_red_before_per_state": 1,
    "n_red_after_per_state": 1,
    "n_green_videos": N_GREEN_VIDEOS_EXPECTED,
    "n_red_videos": N_RED_VIDEOS_EXPECTED,
    "n_total_videos": N_TOTAL_VIDEOS_EXPECTED,
    "n_ab_pairs": N_AB_PAIRS_EXPECTED,
    "n_state_conflicts": 0,
    "n_duplicate_samples": 0,
}

# ---------------------------------------------------------------------------
# Representation / scores (unified with ml_attack metric_schema)
# ---------------------------------------------------------------------------

REPRESENTATION_GREEN = "fullres_detail_cm_enrollment_frozen"
RED_FEATURE_DIM = 13
RED_FEATURE_SOURCE = "formal.lifecycle.red_identity.extract_red_vector"
RED_SCORE_DEF = "S_R = -standardized_euclidean(z_a, z_b)"

SCORE_GENUINE = "GENUINE"
SCORE_CHALLENGE_MISMATCH = "CHALLENGE_MISMATCH"
SCORE_DEVICE_MISMATCH = "DEVICE_MISMATCH"
SCORE_CROSS_STATE_CREDENTIAL = "CROSS_STATE_CREDENTIAL"
SCORE_SOFTWARE_CLONE = "SOFTWARE_CLONE"
SCORE_RED_IDENTITY = "RED_IDENTITY"

POSITIVE_CLASS = "GENUINE"
SCORE_DIRECTION = "HIGHER_IS_MORE_GENUINE"

# ---------------------------------------------------------------------------
# Track IDs
# ---------------------------------------------------------------------------

TRACK_A = "A_database_authentication"
TRACK_B = "B_old_credential_revocation"
TRACK_C = "C_same_state_software_clone"
TRACK_D = "D_cross_state_clone_transfer"
TRACK_R0 = "R0_red_within_state_before_after"
TRACK_R1 = "R1_red_cross_state_identity"
TRACK_J = "J_joint_identity_credential_lifecycle"
TRACK_E = "E_red_conditioned_attack"

# ---------------------------------------------------------------------------
# Data / run status tokens (safe to emit in prepare mode)
# ---------------------------------------------------------------------------

STATUS_DATA_WAITING = "DATA_WAITING"
STATUS_DATA_PARTIAL = "DATA_PARTIAL"
STATUS_DATA_READY = "DATA_READY"
STATUS_DATA_BLOCKED = "DATA_BLOCKED"

MODE_PREPARE = "prepare_only"
MODE_VALIDATE = "validate_dataset"
MODE_FORMAL = "formal_full_run"

# ---------------------------------------------------------------------------
# Forbidden formal outputs until DATA_READY + explicit formal run
# ---------------------------------------------------------------------------

FORBIDDEN_PREPARE_METRIC_TOKENS: tuple[str, ...] = (
    "auc_red",
    "eer_red",
    "auc_red_identity",
    "eer_red_identity",
    "red_top1",
    "red_top1_identity",
    "top1_red",
    "q_R_auc",
    "q_R_eer",
    "red_conditioning_gain",
    "joint_identity_credential_conclusion",
    "RED_CONDITIONED_ATTACK_CONCLUSION",
)

FORBIDDEN_RED_ROLES: tuple[str, ...] = (
    "green speckle corrector",
    "green key predictor",
    "mechanical state classifier",
    "green database auto-selector",
    "old green key restorer",
)

# Formal conclusion tokens — may exist in code, must NOT be written in prepare/
ALLOWED_GREEN_CONCLUSIONS: tuple[str, ...] = (
    "DATABASE_AUTHENTICATION_VALID",
    "DATABASE_AUTHENTICATION_PARTIAL",
    "DATABASE_AUTHENTICATION_FAILED",
    "OLD_TEMPLATE_DATABASE_REVOKED",
    "STATE_BOUND_CREDENTIALS_CONFIRMED",
    "PARTIAL_CROSS_STATE_CREDENTIAL_TRANSFER",
    "CREDENTIAL_RECONFIGURATION_NOT_SECURE",
    "SOFTWARE_CLONE_VALID_SAME_STATE_ONLY",
    "SOFTWARE_CLONE_RECONFIGURATION_RISK",
    "PARTIAL_CROSS_STATE_CLONE_TRANSFER",
    "MODEL_NOT_VALID_IN_SOURCE_STATE",
)

# Joint / red / attack conclusions — code may define them but prepare+incomplete must not emit values
GATED_CONCLUSION_FAMILIES: tuple[str, ...] = (
    "RED_IDENTITY_",
    "JOINT_",
    "RED_CONDITIONED_",
    "IDENTITY_CREDENTIAL_",
)

ABSOLUTE_ASR = None
ASR_STATUS = "SCORE_SPACE_OR_PROTOCOL_MISMATCH"

# ---------------------------------------------------------------------------
# Naming patterns (frozen)
# ---------------------------------------------------------------------------

# Green: {round}_{index}_C{cid:03d}_{state}_{device}.mp4  e.g. A_1_C001_M0_F01.mp4
GREEN_FILENAME_RE = r"^(?P<round>[AB])_(?P<index>\d+)_C(?P<cid>\d{3})_(?P<state>M[0-7])_(?P<device>F\d{2})\.mp4$"
# Red: R_{before|after}_{state}_{device}.mp4
RED_FILENAME_RE = r"^R_(?P<phase>before|after)_(?P<state>M[0-7])_(?P<device>F\d{2})\.mp4$"

SAMPLE_ID_GREEN = "{device}_{state}_{round}_C{cid:03d}"
SAMPLE_ID_RED = "{device}_{state}_R_{phase}"


def challenge_ids() -> list[str]:
    return [f"C{i:03d}" for i in range(CHALLENGE_ID_START, CHALLENGE_ID_END + 1)]


def expected_inventory_row() -> dict[str, Any]:
    return dict(EXPECTED_INVENTORY)
