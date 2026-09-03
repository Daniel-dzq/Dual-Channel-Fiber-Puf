"""Track D-PL — frozen cross-state transfer of partial-leakage models.

Per device, loops over all 8 source states and calls
``track_c_pl_partial_leakage.run_device_source_state`` once per source
state. That function already fits PL0-PL5 on the leaked subset, predicts
each hidden challenge exactly once, and scores the SAME frozen prediction
against every target state (no refit, no PCA update, no mean update, no
target-state data in the fit path).

This module is the per-device orchestrator: it owns Round-B vector loading
(shared across all 8 source states of one device) and stitches together the
per-source-state tables into full per-device transfer tables.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from experiment4_security.ml_attack.config import ModelsConfig
from experiment4_security.ml_attack.partial_leakage_splits import LeakSplit
from experiment4_security.ml_attack.track_c_pl_partial_leakage import (
    load_round_vectors,
    run_device_source_state,
)

logger = logging.getLogger(__name__)


def run_device_partial_leakage(
    device_id: str,
    *,
    shared_cache_root: Path,
    states: list[str],
    all_challenge_ids: list[str],
    challenge_to_bank: dict[str, str],
    challenge_features: dict[str, Any],
    hamming_matrix,
    splits_by_rep: dict[int, dict[int, LeakSplit]],
    models_cfg: ModelsConfig,
    pca_dimension_max: int,
    source_states: list[str] | None = None,
    model_keys: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Full per-device Track C-PL + D-PL computation.

    ``source_states`` restricts which source states are (re)computed —
    used for --device/--leak-size/--model debug filtering and resume; when
    None, all ``states`` are processed as source states.
    """
    t0 = time.time()
    source_states = source_states or list(states)

    logger.info("[Track D-PL] device=%s loading Round-B vectors for %d states", device_id, len(states))
    b_vectors_all_states = {
        t: load_round_vectors(shared_cache_root, device_id, t, "B", all_challenge_ids) for t in states
    }

    c_pl_summaries: list[pd.DataFrame] = []
    c_pl_hidden: list[pd.DataFrame] = []
    d_pl_summaries: list[pd.DataFrame] = []
    d_pl_hidden: list[pd.DataFrame] = []

    for source_state in source_states:
        logger.info("[Track C-PL/D-PL] device=%s source_state=%s", device_id, source_state)
        result = run_device_source_state(
            device_id,
            source_state,
            shared_cache_root=shared_cache_root,
            all_challenge_ids=all_challenge_ids,
            challenge_to_bank=challenge_to_bank,
            challenge_features=challenge_features,
            hamming_matrix=hamming_matrix,
            splits_by_rep=splits_by_rep,
            models_cfg=models_cfg,
            pca_dimension_max=pca_dimension_max,
            b_vectors_all_states=b_vectors_all_states,
            states_for_transfer=states,
            model_keys=model_keys,
        )
        c_pl_summaries.append(result["c_pl_summary"])
        c_pl_hidden.append(result["c_pl_hidden_scores"])
        d_pl_summaries.append(result["transfer_summary"])
        d_pl_hidden.append(result["transfer_hidden_scores"])

    out = {
        "device_id": device_id,
        "c_pl_summary": pd.concat(c_pl_summaries, ignore_index=True) if c_pl_summaries else pd.DataFrame(),
        "c_pl_hidden_scores": pd.concat(c_pl_hidden, ignore_index=True) if c_pl_hidden else pd.DataFrame(),
        "d_pl_transfer_summary": pd.concat(d_pl_summaries, ignore_index=True) if d_pl_summaries else pd.DataFrame(),
        "d_pl_transfer_hidden_scores": pd.concat(d_pl_hidden, ignore_index=True) if d_pl_hidden else pd.DataFrame(),
        "elapsed_s": time.time() - t0,
    }
    logger.info("[Track D-PL] device=%s done elapsed=%.1fs", device_id, out["elapsed_s"])
    return out
