#!/usr/bin/env python3
"""Reproduce Fig. 4e from the publication's explicit cross-round pair table."""
from pathlib import Path
import argparse
import pandas as pd
from experiment00.bootstrap import bootstrap_replicates

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pair-scores',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--replicates',type=int,default=5000)
    p.add_argument('--seed',type=int,default=20260721)
    args=p.parse_args()
    pairs=pd.read_csv(args.pair_scores).rename(columns={
        'fiber_length_cm':'length_cm','device_a':'fiber_id','challenge_a':'challenge',
        'device_b':'fiber_id_b','challenge_b':'challenge_b',
        'similarity_class':'score_type','green_credential_score':'score'})
    pairs['score_type']=pairs.score_type.replace({
        'same_device_same_challenge_similarity':'S_intra',
        'inter_challenge_similarity':'S_inter_challenge',
        'inter_device_similarity':'S_inter_device'})
    boot=bootstrap_replicates(pairs,n_iterations=args.replicates,seed=args.seed)
    complete=boot[boot.complete_replicate]
    rows=[]
    for length in sorted(pairs.length_cm.unique()):
        count=int((complete.selected_length_cm==length).sum())
        rows.append({'fiber_length_cm':length,'selection_count':count,
            'selection_frequency':count/len(complete) if len(complete) else float('nan'),
            'attempted_replicates':len(boot),'complete_replicates':len(complete),
            'undefined_replicates':len(boot)-len(complete)})
    args.output.mkdir(parents=True,exist_ok=True)
    boot=boot.rename(columns={f'L{x}':f'minimum_separation_margin_{x}_cm' for x in pairs.length_cm.unique()})
    boot.to_csv(args.output/'bootstrap_replicates.csv',index=False)
    pd.DataFrame(rows).to_csv(args.output/'bootstrap_selection_frequency.csv',index=False)
    print(f'{len(complete)} of {len(boot)} resampling draws have defined margins at every length.')
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
