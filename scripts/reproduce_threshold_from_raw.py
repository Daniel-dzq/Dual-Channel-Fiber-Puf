#!/usr/bin/env python3
"""Reproduce SI S7.1 threshold-development scores from the public raw recordings.

Green: trim 10 s at each end, three temporal medians, sigma42 local ratio,
mean detail, then per-device/state/round eight-challenge common subtraction.
Red: trimmed intensity median, frozen nine-feature descriptor; standardization
is fitted only on F01-F05, before scoring F06-F15.
"""
from pathlib import Path
import argparse,csv,json,hashlib,itertools
from concurrent.futures import ThreadPoolExecutor
import numpy as np,pandas as pd,cv2
from puf_common.fiber_id import extract_fiber_id_vector,FIBER_ID_FEATURE_NAMES

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--workers',type=int,default=2);a=p.parse_args();root=a.data_root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
 rows=[r for r in csv.DictReader((root/'MANIFEST.csv').open()) if r['dataset']=='threshold_development'];assert len(rows)==765
 mask=np.load(root/'masks/threshold_development_and_formal_valid_pixel_mask.npy').astype(bool)
 def read(r):
  f=(root/r['release_path']).resolve();assert f.is_relative_to(root)
  if hashlib.sha256(f.read_bytes()).hexdigest()!=r['sha256']:raise ValueError(f'Hash mismatch: {f}')
  cap=cv2.VideoCapture(str(f));fps=cap.get(cv2.CAP_PROP_FPS);n=int(r['decoded_frame_count']);trim=round(10*fps);frames=[];count=0
  while True:
   ok,frame=cap.read()
   if not ok:break
   if trim<=count<n-trim:frames.append(frame[:,:,1 if r['optical_channel']=='green' else 2].copy())
   count+=1
  cap.release()
  if count!=n or len(frames)<3:raise ValueError(f'Invalid decode/trim: {f}')
  arr=np.stack(frames)
  if r['optical_channel']=='red':return extract_fiber_id_vector(np.median(arr,axis=0).astype(np.float32),mask)
  result=np.zeros(mask.sum(),dtype=np.float64)
  for block in np.array_split(arr,3):
   im=np.median(block,axis=0).astype(np.float32).astype(np.float64);detail=im/(cv2.GaussianBlur(im,(0,0),sigmaX=42,sigmaY=42,borderType=cv2.BORDER_REFLECT101)+1)-1;result+=detail[mask]/3
  return result
 groups={}
 for r in rows:
  if r['optical_channel']=='green':groups.setdefault((r['device'],r['mechanical_state']),[]).append(r)
 def unit(item):
  (device,state),rr=item;vec={};ans=[]
  for rnd in ['A','B']:
   selected=sorted([r for r in rr if r['acquisition_round']=='Round '+rnd],key=lambda r:r['challenge']);assert len(selected)==8
   arr=np.stack([read(r) for r in selected]);arr=(arr-arr.mean(axis=0)).astype(np.float32).astype(np.float64);arr-=arr.mean(axis=1,keepdims=True);arr/=np.linalg.norm(arr,axis=1,keepdims=True);vec[rnd]=arr
  def row(i,j,ra,rb,kind,score):return dict(device_a=device,mechanical_state_a=state,acquisition_round_a='Round '+ra,challenge_a=f'C{i+1:02}',device_b=device,mechanical_state_b=state,acquisition_round_b='Round '+rb,challenge_b=f'C{j+1:02}',similarity_class=kind,green_credential_score=float(score))
  for rnd in ['A','B']:
   sim=vec[rnd]@vec[rnd].T
   for i,j in itertools.combinations(range(8),2):ans.append(row(i,j,rnd,rnd,'inter_challenge_similarity',sim[i,j]))
  sim=vec['A']@vec['B'].T
  for i in range(8):ans.append(row(i,i,'A','B','same_device_same_state_same_challenge_similarity',sim[i,i]))
  return ans
 green=[]
 with ThreadPoolExecutor(max_workers=a.workers) as pool:
  for i,result in enumerate(pool.map(unit,sorted(groups.items())),1):green+=result;print('Green device/state',i,'/45',flush=True)
 pd.DataFrame(green).to_csv(out/'green_pair_scores.csv',index=False)
 reds=sorted([r for r in rows if r['optical_channel']=='red'],key=lambda r:(r['device'],r['mechanical_state']))
 with ThreadPoolExecutor(max_workers=a.workers) as pool:features=list(pool.map(read,reds))
 keys=[(r['device'],r['mechanical_state']) for r in reds];fit=np.stack([v for key,v in zip(keys,features) if int(key[0][1:])<=5]);mu=fit.mean(axis=0);sd=fit.std(axis=0);sd[sd<1e-12]=1;z=[(v-mu)/sd for v in features];red=[]
 for i,j in itertools.combinations(range(len(keys)),2):
  da,sa=keys[i];db,sb=keys[j];red.append(dict(device_a=da,mechanical_state_a=sa,device_b=db,mechanical_state_b=sb,comparison_class='same_device_cross_state' if da==db else 'different_device',red_identity_score=-float(np.linalg.norm(z[i]-z[j]))))
 pd.DataFrame(red).to_csv(out/'red_pair_scores.csv',index=False);np.savez(out/'red_features.npz',features=np.stack(features),device_state=np.array(keys),feature_names=np.array(FIBER_ID_FEATURE_NAMES),mean=mu,standard_deviation=sd)
 (out/'raw_reproduction_provenance.json').write_text(json.dumps({'raw_files':[{'path':r['release_path'],'sha256':r['sha256']} for r in rows],'mask_sha256':hashlib.sha256((root/'masks/threshold_development_and_formal_valid_pixel_mask.npy').read_bytes()).hexdigest(),'green_pairs':len(green),'red_pairs':len(red),'numpy':np.__version__,'opencv':cv2.__version__},indent=2));print('Completed',len(green),'green and',len(red),'red scores',flush=True)
if __name__=='__main__':main()
