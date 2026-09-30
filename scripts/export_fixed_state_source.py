#!/usr/bin/env python3
"""Export Fig. 6 from fresh raw outputs, with W12 enrollment / W3 query."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import cv2
from puf_common.features import mean_template,build_common_from_templates,subtract_common
from puf_common.ncc import zero_mean_ncc
from puf_common.metrics import auc_roc,equal_error_rate_with_threshold,robust_gap

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['raw-output','analysis-run','output']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);fd=a.analysis_run/'figure_data'
 for source,target in [('figure3c_challenge_matrix.csv','Fig6a_challenge_matrix.csv'),('figure3c_device_matrix.csv','Fig6b_device_matrix.csv')]:
  table=pd.read_csv(fd/source,index_col=0)
  assert table.shape==((8,8) if 'challenge' in source else (15,15))
  table.to_csv(a.output/target)
 (a.output/'Fig6ab_aggregation.json').write_text(json.dumps({'Fig6a':'8 by 8 challenge matrix averaged over all 15 devices','Fig6b':'15 by 15 device matrix averaged over all 8 shared challenges','source':'Fresh raw-video analysis; reciprocal cross-window score definitions are in the analysis code.'},indent=2))
 red=pd.read_csv(fd/'figure3b_red_identity_distribution.csv');red['comparison_class']=red.score_type.map({'SAME_DEVICE':'same device','DIFFERENT_DEVICE':'different device'})
 assert not red.comparison_class.isna().any()
 red.to_csv(a.output/'Fig6e_red_identity_scores.csv',index=False);rr=[]
 for query,g in red.groupby('device_query',sort=True):
  for rank,(_,r) in enumerate(g.sort_values(['q_R','device_enrollment'],ascending=[False,True]).iterrows(),1):rr.append(dict(query_device=query,candidate_device=r.device_enrollment,q_R=r.q_R,rank=rank,is_correct=query==r.device_enrollment))
 rr=pd.DataFrame(rr);rr.to_csv(a.output/'Fig6f_red_retrieval.csv',index=False);green=[];joint=[]
 for device in sorted(red.device_query.unique()):
  mask=cv2.imread(str(a.raw_output/'templates/masks'/f'{device}_valid_mask.png'),0)>0
  with np.load(a.raw_output/'fixed_state_analysis/_shared_cache/green_detail_cm'/f'{device}_detail_cm.npz') as pack:
   cids=sorted(k[:-5] for k in pack.files if k.endswith('_d_b1'));assert len(cids)==8
   reps={c:mean_template([pack[f'{c}_d_b1'],pack[f'{c}_d_b2']]) for c in cids}
   common=build_common_from_templates([reps[c] for c in cids]);templates={c:subtract_common(reps[c],common) for c in cids};allok=True
   for c in cids:
    q=subtract_common(pack[f'{c}_d_b3'],common);scores=sorted([(g,float(zero_mean_ncc(q,templates[g],mask=mask))) for g in cids],key=lambda x:(-x[1],x[0]));allok=allok and scores[0][0]==c
    for rank,(g,score) in enumerate(scores,1):green.append(dict(device_id=device,query_challenge=c,candidate_challenge=g,q_G=score,rank=rank,is_correct=g==c,protocol='W12_enroll_W3_query_enrollment_only_common'))
  rok=bool(rr.loc[(rr.query_device==device)&(rr['rank']==1),'is_correct'].iloc[0]);joint.append(dict(device_id=device,red_correct=rok,all_8_green_correct=allok,joint_correct=rok and allok))
 green=pd.DataFrame(green);joint=pd.DataFrame(joint);green.to_csv(a.output/'Fig6f_green_retrieval.csv',index=False);joint.to_csv(a.output/'Fig6f_joint_retrieval.csv',index=False)
 hd=pd.read_csv(fd/'figure3d_binary_hd_scores.csv');hd['score_class']=hd.pair_type.map({'HD_G':'HD_intra','HD_C':'HD_inter_c','HD_D':'HD_inter_d'});hd=hd.rename(columns={'hd':'HD'});hd.to_csv(a.output/'Fig6c_Hamming_distance_scores.csv',index=False)
 pd.read_csv(fd/'figure3e_standard_puf_metrics.csv').to_csv(a.output/'Fig6d_binary_metrics.csv',index=False)
 genuine=red.loc[red.comparison_class=='same device','q_R'].to_numpy();impostor=red.loc[red.comparison_class=='different device','q_R'].to_numpy();eer,t=equal_error_rate_with_threshold(genuine,impostor);thresholds=np.r_[np.inf,np.unique(np.r_[genuine,impostor])[::-1],-np.inf]
 pd.DataFrame([dict(threshold=t,FPR=float(np.mean(impostor>=t)),TPR=float(np.mean(genuine>=t))) for t in thresholds]).to_csv(a.output/'Fig6e_ROC.csv',index=False)
 summary={'red_top1':int(rr.loc[rr['rank']==1,'is_correct'].sum()),'red_queries':15,'green_top1':int(green.loc[green['rank']==1,'is_correct'].sum()),'green_queries':120,'joint_devices':int(joint.joint_correct.sum()),'devices':15,'red_AUC':auc_roc(genuine,impostor),'red_EER':eer,'red_RG':robust_gap(genuine,impostor),'HD_means':hd.groupby('score_class').HD.mean().to_dict(),'protocol':'W1+W2 enrollment, W3 query; enrollment-only common; red_before-only standardizer'}
 (a.output/'raw_export_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
