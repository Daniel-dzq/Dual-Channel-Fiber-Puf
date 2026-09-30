"""Hierarchical resampling of eligible device/challenge score pairs.

Each sampled device occurrence receives its own with-replacement challenge draw.
Integer pair multiplicities retain the original same/different-identity definitions.
Undefined replicates are retained as NaN, never treated as successful selections.
"""
from __future__ import annotations
from itertools import combinations
import numpy as np
import pandas as pd

SCORE_TYPES = ('S_intra', 'S_inter_challenge', 'S_inter_device')


def _maximin_from_pairs(sub: pd.DataFrame) -> float:
    scores = [sub.loc[sub.score_type == kind, 'score'].to_numpy(float) for kind in SCORE_TYPES]
    if any(len(x) == 0 or not np.isfinite(x).all() for x in scores):
        return float('nan')
    g, c, d = scores
    return float(np.quantile(g, .05) - max(np.quantile(c, .95), np.quantile(d, .95)))


def pair_multiplicities(table, device_draw, challenge_draws, devices, challenges):
    """Count eligible pairs in the explicitly expanded hierarchical sample.

    Same-device pairs come from a single sampled device occurrence; different-device
    pairs join distinct original devices and the same original challenge. Copies of
    one original identity are never relabelled as different devices or challenges.
    """
    device_index = {d: i for i, d in enumerate(devices)}
    challenge_index = {c: k for k, c in enumerate(challenges)}
    counts = np.zeros((len(device_draw), len(challenges)), dtype=np.int64)
    for slot, draw in enumerate(challenge_draws):
        counts[slot] = np.bincount(draw, minlength=len(challenges))
    node = np.zeros((len(devices), len(challenges)), dtype=np.int64)
    within = np.zeros((len(devices), len(challenges), len(challenges)), dtype=np.int64)
    for original_device, c in zip(device_draw, counts):
        node[original_device] += c
        within[original_device] += np.outer(c, c)
    weights = []
    for row in table.itertuples(index=False):
        i, j = device_index[row.fiber_id], device_index[row.fiber_id_b]
        k, l = challenge_index[row.challenge], challenge_index[row.challenge_b]
        if row.score_type == 'S_intra':
            if i != j or k != l: raise ValueError('Invalid genuine identity')
            w = node[i, k]
        elif row.score_type == 'S_inter_challenge':
            if i != j or k == l: raise ValueError('Invalid inter-challenge identity')
            w = within[i, k, l]
        elif row.score_type == 'S_inter_device':
            if i == j or k != l: raise ValueError('Invalid inter-device identity')
            w = node[i, k] * node[j, k]
        else:
            raise ValueError(f'Unknown score type {row.score_type}')
        weights.append(w)
    return np.asarray(weights, dtype=np.int64)


def _validate_pairs(table):
    if not np.isfinite(table.score.to_numpy(float)).all():
        raise ValueError('Nonfinite pair score')
    devices = sorted(set(table.fiber_id) | set(table.fiber_id_b))
    challenges = sorted(set(table.challenge) | set(table.challenge_b))
    expected = {(SCORE_TYPES[0], d, c, d, c) for d in devices for c in challenges}
    expected |= {(SCORE_TYPES[1], d, a, d, b) for d in devices for a,b in combinations(challenges, 2)}
    expected |= {(SCORE_TYPES[2], a, c, b, c) for a,b in combinations(devices, 2) for c in challenges}
    observed = list(zip(table.score_type, table.fiber_id, table.challenge, table.fiber_id_b, table.challenge_b))
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError('Incomplete, repeated or noncanonical eligible pair identities')
    return devices, challenges


def bootstrap_replicates(pair_scores, *, n_iterations=5000, seed=20260721):
    if n_iterations < 1: raise ValueError('n_iterations must be positive')
    rng = np.random.default_rng(seed)
    lengths = sorted(int(x) for x in pair_scores.length_cm.unique())
    inputs = {}
    for length in lengths:
        table = pair_scores[pair_scores.length_cm == length].reset_index(drop=True)
        devices,challenges = _validate_pairs(table)
        inputs[length] = (table, devices, challenges)
    rows = []
    for iteration in range(n_iterations):
        result = {'bootstrap_replicate': iteration + 1}
        for length,(table,devices,challenges) in inputs.items():
            device_draw = rng.choice(len(devices), size=len(devices), replace=True)
            challenge_draws = [rng.choice(len(challenges), size=len(challenges), replace=True) for _ in device_draw]
            weights = pair_multiplicities(table, device_draw, challenge_draws, devices, challenges)
            scores = [np.repeat(table.loc[table.score_type == kind, 'score'].to_numpy(float), weights[table.score_type == kind]) for kind in SCORE_TYPES]
            result[f'L{length}'] = (float(np.quantile(scores[0], .05) - max(np.quantile(scores[1], .95), np.quantile(scores[2], .95))) if all(len(x) for x in scores) else np.nan)
            result[f'L{length}_genuine_pairs'] = len(scores[0])
            result[f'L{length}_inter_challenge_pairs'] = len(scores[1])
            result[f'L{length}_inter_device_pairs'] = len(scores[2])
        complete = all(np.isfinite(result[f'L{x}']) for x in lengths)
        result['complete_replicate'] = complete
        result['selected_length_cm'] = min(lengths, key=lambda x: (-result[f'L{x}'], x)) if complete else np.nan
        rows.append(result)
    return pd.DataFrame(rows)


def hierarchical_bootstrap_length_metrics(pair_scores, *, n_iterations=5000, seed=20260721, confidence_level=.95):
    if not 0 < confidence_level < 1: raise ValueError('Invalid confidence level')
    boot = bootstrap_replicates(pair_scores,n_iterations=n_iterations,seed=seed)
    lengths=sorted(int(x) for x in pair_scores.length_cm.unique());alpha=1-confidence_level
    complete=boot[boot.complete_replicate];summary=[];selection=[];pairwise=[]
    for length in lengths:
        values=boot[f'L{length}'].dropna().to_numpy(float)
        summary.append({'length_cm':length,'metric':'robust_gap_min','mean':float(np.mean(values)) if len(values) else np.nan,'ci_low':float(np.quantile(values,alpha/2)) if len(values) else np.nan,'ci_high':float(np.quantile(values,1-alpha/2)) if len(values) else np.nan,'n_valid_replicates':len(values),'n_attempted_replicates':n_iterations})
        n=int((complete.selected_length_cm==length).sum())
        selection.append({'length_cm':length,'n_selected':n,'selection_probability':n/len(complete) if len(complete) else np.nan,'n_complete_replicates':len(complete),'n_attempted_replicates':n_iterations})
    for a,b in combinations(lengths,2):
        valid=boot[[f'L{a}',f'L{b}']].dropna()
        for left,right in [(a,b),(b,a)]:
            pairwise.append({'length_a_cm':left,'length_b_cm':right,'p_a_gt_b':float((valid[f'L{left}']>valid[f'L{right}']).mean()),'n_valid_replicates':len(valid)})
    return pd.DataFrame(summary),pd.DataFrame(selection),pd.DataFrame(pairwise)
