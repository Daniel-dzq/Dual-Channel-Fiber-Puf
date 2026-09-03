"""Construct red and green official score groups for Experiment 3."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

from experiment3.config import Experiment3Config
from experiment3.green_features import GreenChallengeFeatures, green_similarity
from experiment3.red_features import (
    apply_standardizer,
    fit_standardizer,
    red_similarity_score,
)


def _score_row(**kwargs: Any) -> dict[str, Any]:
    return dict(kwargs)


def construct_red_scores(
    packs: list[dict[str, Any]],
    *,
    development_devices: list[str],
    cfg: Experiment3Config,
    enrollment_state: str | None = None,
    allow_non_s0_standardizer: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build red S_R groups and fit standardizer on development S0 enrollment only."""
    if not packs:
        return pd.DataFrame(), {"mu": None, "sd": None, "n_fit": 0}

    if not development_devices:
        raise RuntimeError(
            "Development device set is empty. "
            "Red standardizer cannot be fitted. Formal evaluation cannot continue."
        )

    enroll = enrollment_state or cfg.experiment.enrollment_state
    dev_packs = [p for p in packs if p["device_id"] in development_devices]
    if not dev_packs:
        raise RuntimeError(
            "No development-device red packs available for standardizer fit. "
            "Refusing all-device fallback."
        )

    s0_vecs = [
        p["representative_vector"]
        for p in packs
        if p["device_id"] in development_devices and p["state_id"] == enroll
    ]
    if not s0_vecs:
        raise RuntimeError(
            f"No development S0 ({enroll}) red vectors for standardizer fit. "
            "Refusing to fit on query states or frozen devices."
        )
    _ = allow_non_s0_standardizer  # reserved; production always requires S0-only

    mu, sd = fit_standardizer(s0_vecs)

    standardized: dict[tuple[str, str], np.ndarray] = {}
    block_std: dict[tuple[str, str], list[np.ndarray]] = {}
    for p in packs:
        key = (p["device_id"], p["state_id"])
        standardized[key] = apply_standardizer(p["representative_vector"], mu, sd)
        block_std[key] = [
            apply_standardizer(v, mu, sd) for v in p["block_vectors"]
        ]

    rows: list[dict[str, Any]] = []

    # 1) same device, same state, different temporal blocks
    for (device, state), blocks in block_std.items():
        for i, j in itertools.combinations(range(len(blocks)), 2):
            s = red_similarity_score(blocks[i], blocks[j])
            rows.append(
                _score_row(
                    channel="red",
                    group="same_device_same_state_diff_block",
                    score=s,
                    device_id_a=device,
                    device_id_b=device,
                    state_id_a=state,
                    state_id_b=state,
                    block_id_a=i + 1,
                    block_id_b=j + 1,
                    challenge_id_a="",
                    challenge_id_b="",
                    round_id_a="",
                    round_id_b="",
                )
            )

    # 2) same device, different state
    by_device: dict[str, list[str]] = {}
    for device, state in standardized:
        by_device.setdefault(device, []).append(state)
    for device, states in by_device.items():
        for sa, sb in itertools.combinations(sorted(states), 2):
            s = red_similarity_score(standardized[(device, sa)], standardized[(device, sb)])
            rows.append(
                _score_row(
                    channel="red",
                    group="same_device_diff_state",
                    score=s,
                    device_id_a=device,
                    device_id_b=device,
                    state_id_a=sa,
                    state_id_b=sb,
                    block_id_a=0,
                    block_id_b=0,
                    challenge_id_a="",
                    challenge_id_b="",
                    round_id_a="",
                    round_id_b="",
                )
            )

    # 3) different device (all state pairs)
    devices = sorted({d for d, _ in standardized})
    for da, db in itertools.combinations(devices, 2):
        states_a = by_device.get(da, [])
        states_b = by_device.get(db, [])
        for sa in states_a:
            for sb in states_b:
                s = red_similarity_score(
                    standardized[(da, sa)], standardized[(db, sb)]
                )
                rows.append(
                    _score_row(
                        channel="red",
                        group="diff_device",
                        score=s,
                        device_id_a=da,
                        device_id_b=db,
                        state_id_a=sa,
                        state_id_b=sb,
                        block_id_a=0,
                        block_id_b=0,
                        challenge_id_a="",
                        challenge_id_b="",
                        round_id_a="",
                        round_id_b="",
                    )
                )

    meta = {
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "n_fit": len(s0_vecs),
        "fit_devices": sorted(
            {
                p["device_id"]
                for p in packs
                if p["device_id"] in development_devices and p["state_id"] == enroll
            }
        ),
        "fit_states": [enroll],
        "fit_rounds": ["enrollment"],
        "contains_frozen_data": False,
        "standardized": standardized,
    }
    if not meta["fit_devices"]:
        raise RuntimeError("Standardizer fit_devices empty after S0 filter")
    return pd.DataFrame(rows), meta


def evaluate_s0_gallery_queries(
    standardized: dict[tuple[str, str], np.ndarray],
    *,
    enrollment_state: str,
    query_states: list[str],
) -> pd.DataFrame:
    """Identify each query against all available S0 device templates."""
    if enrollment_state != "S0":
        raise RuntimeError(f"Gallery enrollment_state must be S0, got {enrollment_state}")
    if not set(query_states).issubset({"S1", "S2"}):
        raise RuntimeError(f"Query states must be subset of {{S1,S2}}, got {query_states}")
    gallery = {
        d: v
        for (d, s), v in standardized.items()
        if s == enrollment_state
    }
    if not gallery:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    gallery_devices = sorted(gallery.keys())
    for (qd, qs), qvec in standardized.items():
        if qs not in query_states:
            continue
        if qs == enrollment_state:
            raise RuntimeError("Query state must not equal enrollment/gallery state")
        scores = {
            gd: red_similarity_score(qvec, gallery[gd]) for gd in gallery_devices
        }
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        rank_map = {d: i + 1 for i, (d, _) in enumerate(ranked)}
        correct_score = scores.get(qd, float("nan"))
        wrong_scores = [sc for d, sc in scores.items() if d != qd]
        best_wrong = max(wrong_scores) if wrong_scores else float("nan")
        margin = (
            correct_score - best_wrong
            if np.isfinite(correct_score) and np.isfinite(best_wrong)
            else float("nan")
        )
        top1 = ranked[0][0] if ranked else ""
        rows.append(
            {
                "query_device": qd,
                "query_state": qs,
                "true_device": qd,
                "predicted_top1": top1,
                "correct": bool(top1 == qd),
                "rank_correct": rank_map.get(qd, len(gallery_devices) + 1),
                "score_correct": correct_score,
                "score_best_wrong": best_wrong,
                "identity_margin": margin,
                "n_gallery": len(gallery_devices),
                "mrr_contribution": 1.0 / rank_map.get(qd, len(gallery_devices) + 1),
                **{f"score_vs_{gd}": scores[gd] for gd in gallery_devices},
            }
        )
    return pd.DataFrame(rows)


def construct_green_scores(
    features: dict[tuple[str, str, str, str], GreenChallengeFeatures],
    mask: np.ndarray,
    *,
    state_order: list[str] | None = None,
) -> pd.DataFrame:
    """Fixed comparison direction: Round A versus Round B where applicable."""
    rows: list[dict[str, Any]] = []
    keys = list(features.keys())
    devices = sorted({k[0] for k in keys})
    states = state_order or sorted({k[1] for k in keys})
    challenges = sorted({k[3] for k in keys})

    # 1) same device, same state, same challenge: A vs B (genuine / re-enrollment)
    for d in devices:
        for s in states:
            for c in challenges:
                ka, kb = (d, s, "A", c), (d, s, "B", c)
                if ka in features and kb in features:
                    sc = green_similarity(features[ka], features[kb], mask)
                    rows.append(
                        _score_row(
                            channel="green",
                            group="same_device_same_state_same_challenge",
                            score=sc,
                            device_id_a=d,
                            device_id_b=d,
                            state_id_a=s,
                            state_id_b=s,
                            round_id_a="A",
                            round_id_b="B",
                            challenge_id_a=c,
                            challenge_id_b=c,
                        )
                    )

    # 2) same device, same state, different challenge: A Ci vs B Cj, i!=j
    for d in devices:
        for s in states:
            for ci, cj in itertools.product(challenges, challenges):
                if ci == cj:
                    continue
                ka, kb = (d, s, "A", ci), (d, s, "B", cj)
                if ka in features and kb in features:
                    sc = green_similarity(features[ka], features[kb], mask)
                    rows.append(
                        _score_row(
                            channel="green",
                            group="same_device_same_state_diff_challenge",
                            score=sc,
                            device_id_a=d,
                            device_id_b=d,
                            state_id_a=s,
                            state_id_b=s,
                            round_id_a="A",
                            round_id_b="B",
                            challenge_id_a=ci,
                            challenge_id_b=cj,
                        )
                    )

    # 3) same device, different state, same challenge: earlier A vs later B
    for d in devices:
        for i, sa in enumerate(states):
            for sb in states[i + 1 :]:
                for c in challenges:
                    ka, kb = (d, sa, "A", c), (d, sb, "B", c)
                    if ka in features and kb in features:
                        sc = green_similarity(features[ka], features[kb], mask)
                        rows.append(
                            _score_row(
                                channel="green",
                                group="same_device_diff_state_same_challenge",
                                score=sc,
                                device_id_a=d,
                                device_id_b=d,
                                state_id_a=sa,
                                state_id_b=sb,
                                round_id_a="A",
                                round_id_b="B",
                                challenge_id_a=c,
                                challenge_id_b=c,
                            )
                        )

    # 4) different device, same state, same challenge: A vs B
    for da, db in itertools.combinations(devices, 2):
        for s in states:
            for c in challenges:
                ka, kb = (da, s, "A", c), (db, s, "B", c)
                if ka in features and kb in features:
                    sc = green_similarity(features[ka], features[kb], mask)
                    rows.append(
                        _score_row(
                            channel="green",
                            group="diff_device_same_state_same_challenge",
                            score=sc,
                            device_id_a=da,
                            device_id_b=db,
                            state_id_a=s,
                            state_id_b=s,
                            round_id_a="A",
                            round_id_b="B",
                            challenge_id_a=c,
                            challenge_id_b=c,
                        )
                    )

    return pd.DataFrame(rows)


def build_score_matrix_s0_gallery(
    query_df: pd.DataFrame,
    gallery_devices: list[str],
) -> pd.DataFrame:
    """Wide matrix: rows = queries, columns = gallery device scores."""
    if query_df.empty:
        return query_df
    cols = ["query_device", "query_state", "rank_correct", "identity_margin", "correct"]
    score_cols = [c for c in query_df.columns if c.startswith("score_vs_")]
    return query_df[cols + score_cols].copy()
