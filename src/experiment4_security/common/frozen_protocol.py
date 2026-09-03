"""Frozen scientific protocol constants for Experiment 4.

Red = persistent device identity.
Green = challenge- and mounting-state-bound credential.
Red must never align, correct, predict, or reconstruct green responses.
"""

from __future__ import annotations

from puf_common.fiber_id import FIBER_ID_FEATURE_NAMES as RED_LOWDIM_NAMES

# OpenCV BGR plane indices
BGR_GREEN_INDEX = 1
BGR_RED_INDEX = 2

ENVELOPE_SIGMA_PX = 42.0
ENVELOPE_EPS = 1.0

# Red identity: nine-feature Fiber-ID descriptor (no PCA).

DEVELOPMENT_DEVICES = ("F01", "F02", "F03", "F04", "F05")
FROZEN_TEST_DEVICES = ("F06", "F07", "F08", "F09", "F10", "F11", "F12", "F13", "F14", "F15")

LIFECYCLE_STATES = ("S0", "S1", "S2")
ENROLLMENT_STATE = "S0"
QUERY_STATES = ("S1", "S2")

SESSION_RULE_CANDIDATES = (8, 7, 6)  # challenges that must pass out of 8

GENERATION_VERSION = "m2_128_v1"
MASTER_SEED = 20260719
