#!/usr/bin/env python3
"""Decode authentic macro-pixel videos and export individual Fig. 3 scores."""
from pathlib import Path
import argparse,json,gc
import numpy as np
import pandas as pd
import cv2
from e01.config import load_config
from e01.analysis.green_screening import load_dark_green,prepare_dark_for_videos,read_challenge_windows,discover_screening_videos
from e01.analysis.envelope import local_ratio_detail
from e01.analysis.ncc import zero_mean_ncc

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);cfg=load_config(a.config);base=cfg.root/'analysis/screening';params=json.loads((base/'locked_analysis_params.json').read_text());mask=cv2.imread(str(base/'valid_mask.png'),0)>0;dark,_=load_dark_green(cfg)
 discovered=discover_screening_videos(cfg.root/'videos',cfg.experiment.macro_pixel_sizes,list(cfg.experiment.challenge_ids),allow_partial=False);dark=prepare_dark_for_videos(dark,next(iter(next(iter(discovered.values())).values())),cfg);rows=[]
 for m,paths in sorted(discovered.items()):
  raw={};detail={}
  for cid,path in sorted(paths.items()):
   cw=read_challenge_windows(path,m,cid,cfg,dark);raw[cid]=np.mean(np.stack(cw.g_windows),axis=0);detail[cid]=[local_ratio_detail(w,sigma=params['envelope_sigma_px'],eps_env=params['envelope_eps']) for w in cw.g_windows]
  ids=sorted(detail);means={c:np.mean(np.stack(detail[c]),axis=0) for c in ids};common=np.mean(np.stack([means[c] for c in ids]),axis=0);reps={c:means[c]-common for c in ids}
  for c in ids:
   for i in range(3):
    for j in range(i+1,3):rows.append(dict(m=m,score_class='S_intra',challenge_a=c,challenge_b=c,window_a=i+1,window_b=j+1,NCC=zero_mean_ncc(detail[c][i]-common,detail[c][j]-common,mask=mask)))
  for i,c in enumerate(ids):
   for other in ids[i+1:]:rows.append(dict(m=m,score_class='S_inter_c',challenge_a=c,challenge_b=other,window_a='',window_b='',NCC=zero_mean_ncc(reps[c],reps[other],mask=mask)))
  if m==2:
   for label,images in [('raw',raw),('processed',reps)]:
    mat=np.array([[zero_mean_ncc(images[c],images[k],mask=mask) for k in ids] for c in ids]);pd.DataFrame(mat,index=ids,columns=ids).to_csv(a.output/f'Fig3c_m2_{label}_NCC_matrix.csv')
   np.savez_compressed(a.output/'Fig3b_m2_representative_fields.npz',**{f'C{int(c):02}_raw':raw[c] for c in ids},**{f'C{int(c):02}_detail_cm':reps[c] for c in ids},valid_mask=mask)
  del raw,detail,means,common,reps,cw;gc.collect();print('Completed m=',m,flush=True)
 df=pd.DataFrame(rows);df.to_csv(a.output/'Fig3d_individual_NCC_scores.csv',index=False);summary=[]
 for m,g in df.groupby('m'):
  intra=g.loc[g.score_class=='S_intra','NCC'];inter=g.loc[g.score_class=='S_inter_c','NCC'];summary.append(dict(m=m,n_intra=len(intra),n_inter=len(inter),Q05_S_intra=intra.quantile(.05),Q95_S_inter_c=inter.quantile(.95),G=intra.quantile(.05)-inter.quantile(.95)))
 summary=pd.DataFrame(summary);summary.to_csv(a.output/'Fig3e_G_from_raw_scores.csv',index=False);ref=pd.read_csv(base/'macro_pixel_summary.csv').set_index('macro_pixel');delta=max(abs(r.G-ref.loc[r.m,'robust_gap_detail_cm']) for r in summary.itertuples());assert delta<1e-10
 (a.output/'raw_export_summary.json').write_text(json.dumps({'individual_scores':len(df),'max_G_difference_vs_raw_screening':delta,'protocol':params},indent=2))
if __name__=='__main__':main()
