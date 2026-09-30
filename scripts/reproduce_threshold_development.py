#!/usr/bin/env python3
"""Validate the complete threshold-development analysis against Supplementary Note 7.1.

The dataset supplies processed_data/threshold_development/{green,red}_pair_scores.csv.
This table-level validation does not replace raw-recording reproduction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from experiment4_security.lifecycle.thresholds import select_session_rule, select_tau
from puf_common.checks import check, open_dataset, report, reproduction_parser

DEVELOPMENT = [f'F{i:02d}' for i in range(1, 6)]
EVALUATION = [f'F{i:02d}' for i in range(6, 16)]
GENUINE = 'same_device_same_state_same_challenge'
INTER_CHALLENGE = 'same_device_same_state_diff_challenge'


def development_operating_point(green):
    dev = green[green.device_id_a.isin(DEVELOPMENT) & green.device_id_b.isin(DEVELOPMENT)]
    genuine = dev.loc[dev.group == GENUINE, 'score'].to_numpy(float)
    inter_challenge = dev.loc[dev.group == INTER_CHALLENGE, 'score'].to_numpy(float)
    if len(genuine) != 120 or len(inter_challenge) != 840:
        raise ValueError('Supplementary Note 7.1 requires 120 genuine and 840 inter-challenge development scores')
    expected_genuine = {(device, state, f'C{k:02d}')
        for device in DEVELOPMENT for state in ['S0', 'S1', 'S2'] for k in range(1, 9)}
    g = dev[dev.group == GENUINE]
    observed_genuine = list(zip(g.device_id_a, g.state_id_a, g.challenge_id_a))
    if len(set(observed_genuine)) != 120 or set(observed_genuine) != expected_genuine:
        raise ValueError('Missing or repeated genuine development identities')
    if not ((g.device_id_a == g.device_id_b) & (g.state_id_a == g.state_id_b)
            & (g.challenge_id_a == g.challenge_id_b)
            & (g.round_id_a == 'A') & (g.round_id_b == 'B')).all():
        raise ValueError('Genuine development scores must compare Round A with Round B')
    expected_inter = {(device, state, rnd, f'C{i:02d}', f'C{j:02d}')
        for device in DEVELOPMENT for state in ['S0', 'S1', 'S2']
        for rnd in ['A', 'B'] for i in range(1, 9) for j in range(i+1, 9)}
    c = dev[dev.group == INTER_CHALLENGE]
    observed_inter = list(zip(c.device_id_a, c.state_id_a, c.round_id_a,
                              c.challenge_id_a, c.challenge_id_b))
    if len(set(observed_inter)) != 840 or set(observed_inter) != expected_inter:
        raise ValueError('Missing or repeated inter-challenge development identities')
    if not ((c.device_id_a == c.device_id_b) & (c.state_id_a == c.state_id_b)
            & (c.round_id_a == c.round_id_b)).all():
        raise ValueError('Inter-challenge development scores require the same device, state and round')
    if not np.isfinite(np.concatenate([genuine, inter_challenge])).all():
        raise ValueError('Non-finite threshold-development scores')
    T_G = select_tau(genuine, inter_challenge)
    rule = select_session_rule(dev, T_G)
    return {'T_G': T_G, 'n_req': int(rule['selected_k_of_8']),
            'candidates': rule['candidates'], 'n_genuine': len(genuine),
            'n_inter_challenge': len(inter_challenge)}


def publication_score_tables(green, red):
    """Map publication column names to the internal score-pair implementation."""
    common = {'device_a': 'device_id_a', 'device_b': 'device_id_b',
              'mechanical_state_a': 'state_id_a', 'mechanical_state_b': 'state_id_b'}
    green = green.rename(columns={**common, 'green_credential_score': 'score',
        'similarity_class': 'group', 'acquisition_round_a': 'round_id_a',
        'acquisition_round_b': 'round_id_b', 'challenge_a': 'challenge_id_a',
        'challenge_b': 'challenge_id_b'}).copy()
    red = red.rename(columns={**common, 'red_identity_score': 'score',
        'comparison_class': 'group'}).copy()
    green['group'] = green['group'].replace({
        'same_device_same_state_same_challenge_similarity': GENUINE,
        'inter_challenge_similarity': INTER_CHALLENGE})
    red['group'] = red['group'].replace({'same_device_cross_state': 'same_device_diff_state',
                                       'different_device': 'diff_device'})
    for table in [green, red]:
        for col in ['state_id_a', 'state_id_b']:
            table[col] = table[col].str.replace('M', 'S', regex=False)
    for col in ['round_id_a', 'round_id_b']:
        green[col] = green[col].str.replace('Round ', '', regex=False)
    return green, red


def main():
    args = reproduction_parser(__doc__.splitlines()[0]).parse_args()
    ds, out = open_dataset(args, 'threshold_development')
    green = ds.read_csv('processed_data/threshold_development/green_pair_scores.csv')
    red = ds.read_csv('processed_data/threshold_development/red_pair_scores.csv')
    green, red = publication_score_tables(green, red)
    op = development_operating_point(green)
    dev = red[red.device_id_a.isin(DEVELOPMENT) & red.device_id_b.isin(DEVELOPMENT)]
    genuine = dev.loc[dev.group == 'same_device_diff_state', 'score'].to_numpy(float)
    different = dev.loc[dev.group == 'diff_device', 'score'].to_numpy(float)
    if len(genuine) != 15 or len(different) != 90:
        raise ValueError('Red threshold development requires 15 same-device and 90 different-device scores')
    T_R = select_tau(genuine, different)
    event_rows = []
    for device in DEVELOPMENT + EVALUATION:
        for state in ['S1', 'S2']:
            r = red[(red.device_id_a == device) & (red.device_id_b == device)
                    & (((red.state_id_a == 'S0') & (red.state_id_b == state))
                       | ((red.state_id_b == 'S0') & (red.state_id_a == state)))]
            g = green[(green.device_id_a == device) & (green.state_id_a == state)
                      & (green.group == GENUINE)]
            if len(r) != 1 or len(g) != 8:
                raise ValueError(f'Incomplete authentication event: {device} {state}')
            red_identity_score = float(r.score.iloc[0])
            accepted_challenges = int((g.score >= op['T_G']).sum())
            event_rows.append({'device_id': device, 'mechanical_state': state.replace('S', 'M'),
                'red_identity_score': red_identity_score, 'accepted_challenges': accepted_challenges,
                'joint_authentication_decision': red_identity_score >= T_R and accepted_challenges >= op['n_req']})
    events = pd.DataFrame(event_rows)
    evaluation = events[events.device_id.isin(EVALUATION)]
    sessions = {r['k_of_8']: round(r['session_pass_rate'] * r['n_events']) for r in op['candidates']}
    checks = [check('genuine development scores', op['n_genuine'], 120, 0),
        check('inter-challenge development scores', op['n_inter_challenge'], 840, 0),
        check('red identity threshold T_R', round(T_R, 3), -1.748, 0),
        check('green credential threshold T_G', round(op['T_G'], 3), 0.134, 0),
        check('required accepted challenges n_req', op['n_req'], 6, 0),
        *[check(f'{k}-of-8 accepted development sessions', sessions[k], n, 0)
          for k, n in [(8, 11), (7, 14), (6, 15)]],
        check('current-state joint authentication events', int(events.joint_authentication_decision.sum()), 30, 0),
        check('evaluation joint authentication events', int(evaluation.joint_authentication_decision.sum()), 20, 0)]
    events.to_csv(out / 'authentication_events.csv', index=False)
    return 0 if report(checks, out, 'Supplementary Note 7.1 publication validation') else 1

if __name__ == '__main__':
    raise SystemExit(main())
