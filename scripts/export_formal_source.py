#!/usr/bin/env python3
"""Export Fig. 7–8 / Fig. S2–S4 source tables from complete raw-run outputs.

No historical score tables are used. Outputs go to a separate directory so an
incomplete or changed run cannot overwrite the released publication data.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from experiment4_security.ml_attack.partial_leakage_reporting import collapse_repetitions, model_level_summary
from experiment4_security.ml_attack.batch_eval import auc_eer_genuine_vs_negative, robust_gap_q05_minus_q95

STATES = [f'M{i}' for i in range(8)]
DEVICES = [f'F{i:02}' for i in range(1, 11)]
FULL = ['mean_response', 'exact_template_replay', 'ridge_clone', 'kernel_ridge_clone', 'random_fourier_ridge_clone', 'small_mlp_clone']
NAMES = dict(zip(FULL, ['Mean-response baseline', 'Exact replay', 'Ridge', 'Kernel Ridge', 'RFF Ridge', 'Small MLP']))
NAMES.update(mean_leaked_response='Mean-response baseline', nearest_leaked_challenge='Nearest-challenge baseline')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['formal-root', 'cross-device-root', 'red-root', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    roots = [args.formal_root.resolve(), args.cross_device_root.resolve(), args.red_root.resolve()]
    out = args.output.resolve()
    if any(out == root or root in out.parents or out in root.parents for root in roots):
        raise ValueError('Use an output directory separate from all raw-run output roots')
    inputs = []
    def read(path):
        inputs.append({'path': str(path), 'sha256': digest(path)})
        return pd.read_csv(path)
    def merged(filename, count, keys):
        table = pd.concat([read(roots[0] / dev / filename) for dev in DEVICES], ignore_index=True)
        if len(table) != count or table.duplicated(keys).any():
            raise ValueError(f'Incomplete or duplicate evaluations: {filename}')
        return table
    for dev in DEVICES:
        status = json.loads((roots[0] / dev / 'COMPLETE.json').read_text())
        if not {'A', 'B', 'C', 'D', 'C-PL', 'D-PL'}.issubset(status['tracks']):
            raise ValueError(f'Full attack stages not completed: {dev}')
    a = merged('track_a_database_authentication_summary.csv', 80, ['device_id', 'state_id'])
    b = merged('track_b_template_transfer_matrix_summary.csv', 640, ['device_id', 'source_state', 'target_state'])
    c = merged('track_c_same_state_clone_summary.csv', 480, ['device_id', 'source_state', 'attack_method'])
    d = merged('track_d_clone_transfer_matrix_summary.csv', 3200, ['device_id', 'source_state', 'target_state', 'attack_method'])
    p = merged('track_c_pl_partial_leakage_summary.csv', 9600, ['device_id', 'source_state', 'repetition', 'leak_size', 'attack_method'])
    q = merged('track_d_pl_partial_leakage_transfer_summary.csv', 76800, ['device_id', 'source_state', 'target_state', 'repetition', 'leak_size', 'attack_method'])
    # Reuse the exact float64 genuine scores already computed by authentication.
    # This also avoids a second, potentially float32, calculation of that reference.
    auth_scores = pd.concat([read(roots[0] / dev / 'track_a_scores.csv.gz') for dev in DEVICES], ignore_index=True)
    genuine = auth_scores[auth_scores.is_genuine]
    if len(genuine) != 10240 or genuine.duplicated(['device_id','source_state','challenge_id']).any():
        raise ValueError('Authentication reference incomplete')
    device_rows = []
    for state in STATES:
        pairs = read(roots[1] / state / 'scores.csv.gz')
        if len(pairs) != 11520 or pairs.duplicated(['source_device','target_device','state_id','challenge_id']).any() or not pairs.state_id.eq(state).all():
            raise ValueError(f'Cross-device analysis incomplete: {state}')
        scores = pairs.S_D.to_numpy(float)
        reference = genuine.loc[genuine.source_state.eq(state),'score'].to_numpy(float)
        ae = auc_eer_genuine_vs_negative(reference, scores)
        device_rows.append({'state_id':state,'n_S_D':len(scores),'median_S_D':np.median(scores),'q95_S_D':np.percentile(scores,95),'rg_device':robust_gap_q05_minus_q95(reference,scores),'device_mismatch_status':'AVAILABLE_MULTI_DEVICE','auc_device':ae['auc'],'eer_device':ae['eer']})
    sd = pd.DataFrame(device_rows)
    red = read(roots[2] / 'pairs_9d.csv')
    red_summary_path = roots[2] / 'summary.json'
    inputs.append({'path': str(red_summary_path), 'sha256': digest(red_summary_path)})
    red_summary = json.loads(red_summary_path.read_text())['pooled']
    model = model_level_summary(collapse_repetitions(q), bootstrap_iterations=10000, bootstrap_seed=42)
    src = out / 'source_data'; src.mkdir(parents=True, exist_ok=True)
    processed = out / 'processed_data'; processed.mkdir(exist_ok=True)
    def export(table, columns, name):
        table = table[list(columns)].rename(columns=columns).copy()
        if 'method' in table:
            table['method'] = table.method.map(NAMES)
            if table.method.isna().any():
                raise ValueError('Unknown attack label')
        table.to_csv(src / name, index=False)
    def matrix(table, column):
        return table.groupby(['source_state', 'target_state'])[column].median().unstack().reindex(index=STATES, columns=STATES)
    for table, filename in [(a,'track_a_database_authentication_summary.csv'), (b,'track_b_template_transfer_matrix_summary.csv'), (c,'track_c_same_state_clone_summary.csv'), (d,'track_d_clone_transfer_matrix_summary.csv'), (p,'track_c_pl_partial_leakage_summary.csv'), (q,'track_d_pl_partial_leakage_transfer_summary.csv'), (model,'track_d_pl_model_level_summary.csv'), (sd,'track_s_d_device_mismatch_summary.csv')]:
        table.to_csv(processed / filename, index=False)
    quality = a[['device_id','state_id','rg_challenge','top1','eer_challenge','state_conclusion']].rename(columns={'device_id':'device','state_id':'mechanical_state','rg_challenge':'robust_challenge_margin','top1':'challenge_retrieval_accuracy','eer_challenge':'equal_error_rate','state_conclusion':'quality_class'})
    quality.quality_class = quality.quality_class.str.replace('DATABASE_AUTHENTICATION_', '').str.title()
    quality.to_csv(src / 'Fig7a_and_S2ab_device_state_quality.csv', index=False)
    sd.rename(columns={'state_id':'mechanical_state','n_S_D':'inter_device_comparison_count','median_S_D':'inter_device_similarity_median','q95_S_D':'inter_device_similarity_upper_tail_quantile','rg_device':'device_robust_gap','auc_device':'device_auc','eer_device':'device_eer'}).to_csv(src / 'FigS2cd_inter_device_discrimination.csv', index=False)
    groups = [a.median_S_G, a.median_S_C, sd.median_S_D, b.loc[~b.is_diagonal, 'median_S_X']]
    labels = ['Same device, state and challenge','Different challenge','Different device','Different mechanical state']
    pd.concat([pd.DataFrame({'comparison':label,'green_credential_score':values}) for label,values in zip(labels,groups)]).to_csv(src / 'Fig7b_similarity_summary_points.csv', index=False)
    matrix(b, 'median_S_X').to_csv(src / 'Fig7c_cross_state_matrix.csv')
    export(b, {'device_id':'device','source_state':'source_mechanical_state','target_state':'target_mechanical_state','is_diagonal':'is_diagonal','median_S_X':'cross_state_similarity_median','rg_cross_state_credential':'cross_state_revocation_margin'}, 'Fig7c_device_state_comparisons.csv')
    pd.concat([pd.DataFrame({'comparison':label,'red_identity_score':values}) for label,values in zip(['Same device, same state','Same device, cross state','Different device'], [red.loc[red.r0,'S_R'],red.loc[red.r1,'S_R'],red.loc[~red.same_device,'S_R']])]).to_csv(src / 'Fig7d_red_identity_scores.csv', index=False)
    pd.DataFrame(red_summary['per_device']).rename(columns={'AUC':'red_identity_auc','EER':'red_identity_eer'}).to_csv(src / 'Fig7d_red_identity_discrimination.csv', index=False)
    pd.DataFrame([{'global_red_identity_eer':red_summary['cross_state_eer']}]).to_csv(src / 'Fig7d_global_identity_summary.csv', index=False)
    export(c, {'device_id':'device','source_state':'source_mechanical_state','attack_method':'method','median_S_A':'attack_score_median','source_state_valid':'reconstruction_or_replay_success','median_genuine':'genuine_score_median'}, 'Fig8a_complete_disclosure_unit_scores.csv')
    export(d, {'device_id':'device','source_state':'source_mechanical_state','target_state':'target_mechanical_state','attack_method':'method','is_diagonal':'same_state','median_S_A':'attack_score_median','delta_clone_same_to_cross':'state_specificity_gap'}, 'Fig8b_and_S3_complete_disclosure_transfer.csv')
    band = c.loc[c.attack_method.eq('exact_template_replay'),'median_genuine'].quantile([.05,.95]).to_numpy()
    pd.DataFrame([{'genuine_lower_tail_quantile':band[0],'genuine_upper_tail_quantile':band[1]}]).to_csv(src / 'Fig8a_genuine_reference.csv', index=False)
    summary = []
    for method in FULL[1:]:
        sub = d[d.attack_method.eq(method)]
        matrix(sub, 'median_S_A').to_csv(src / f'FigS3_{NAMES[method].lower().replace(" ","_")}_matrix.csv')
        summary.append({'method':method,'same':sub.loc[sub.is_diagonal,'median_S_A'].median(),'cross':sub.loc[~sub.is_diagonal,'median_S_A'].median(),'gap':sub.loc[~sub.is_diagonal,'delta_clone_same_to_cross'].median()})
    summary = pd.DataFrame(summary)
    export(summary, {'method':'method','same':'same_state_attack_score_median','cross':'cross_state_attack_score_median','gap':'state_specificity_gap'}, 'Fig8b_same_and_cross_state_summary.csv')
    pc = {'device_id':'device','source_state':'source_mechanical_state','repetition':'partition','split_seed':'partition_seed','leak_size':'disclosed_crp_count','attack_method':'method','n_hidden':'hidden_challenge_count','median_S_A_partial':'hidden_attack_score_median','median_S_G_hidden':'hidden_genuine_score_median','Top1_hidden':'hidden_challenge_retrieval_accuracy','normalized_retrieval_lift':'top1_lift','median_residual_ncc':'hidden_challenge_mean_residual_ncc','effective_pca_dim':'retained_pca_dimension','common_dominated_evaluation':'common_dominated_evaluation'}
    p['common_dominated_evaluation'] = p.common_dominance_flag.eq('HIGH_NCC_DOMINATED_BY_COMMON_RESPONSE')
    export(p, pc, 'Fig8c_and_S4_partial_disclosure_unit_results.csv')
    export(q, {**{k:v for k,v in pc.items() if k not in ['median_residual_ncc','effective_pca_dim','common_dominated_evaluation']},'target_state':'target_mechanical_state','is_diagonal':'same_state'}, 'Fig8d_and_S4a_partial_disclosure_transfer.csv')
    curves = p.groupby(['attack_method','leak_size'])[['median_S_A_partial','normalized_retrieval_lift']].median().reset_index()
    export(curves, {'attack_method':'method','leak_size':'disclosed_crp_count','median_S_A_partial':'hidden_attack_score_median','normalized_retrieval_lift':'top1_lift_median'}, 'Fig8c_partial_disclosure_summary.csv')
    p.groupby('leak_size').median_S_G_hidden.median().rename_axis('disclosed_crp_count').rename('hidden_genuine_score_median').to_csv(src / 'Fig8c_hidden_genuine_reference.csv')
    matrix(q[q.attack_method.eq('ridge_clone') & q.leak_size.eq(96)], 'median_S_A_partial').to_csv(src / 'FigS4a_ridge_96_disclosed_matrix.csv')
    audit = []
    for (method,n),group in p[p.attack_method.isin(FULL[2:])].groupby(['attack_method','leak_size']):
        dominated = group.common_dominated_evaluation
        audit.append({'method':method,'disclosed_crp_count':n,'retained_pca_dimension':group.effective_pca_dim.median(),'median_unit_mean_residual_ncc':group.median_residual_ncc.median(),'common_dominated_evaluation_count':int(dominated.sum()),'evaluation_count':len(group),'common_dominated_evaluation_fraction':dominated.mean()})
    audit = pd.DataFrame(audit)
    export(audit, {x:x for x in audit.columns}, 'FigS4bcd_model_audit.csv')
    comparisons = [{'disclosure':'Complete','method':row.method,'same_state':row.same,'cross_state':row.cross} for row in summary.itertuples()]
    for method in FULL[2:5]:
        row = model[model.attack_method.eq(method) & model.leak_size.eq(96)].iloc[0]
        comparisons.append({'disclosure':'Partial: 96/128','method':method,'same_state':row.diagonal_median_S_A,'cross_state':row.off_diagonal_median_S_A})
    export(pd.DataFrame(comparisons), {'disclosure':'disclosure','method':'method','same_state':'same_state_attack_score_median','cross_state':'cross_state_attack_score_median'}, 'Fig8d_disclosure_summary.csv')
    (out / 'export_provenance.json').write_text(json.dumps({'inputs':inputs,'outputs':{str(f.relative_to(out)):digest(f) for f in sorted(out.rglob('*.csv'))}}, indent=2))


if __name__ == '__main__':
    main()
