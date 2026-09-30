"""Reconstruct all length-screening NCC pairs from the canonical raw package."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import csv,json,hashlib,time,gc
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-root',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--workers',type=int,default=1)
parser.add_argument('--include-spatial',action='store_true',help='Also export raw-intensity PSD entropy and C01 Round A representative fields')
args=parser.parse_args()
DATA=args.data_root.resolve();OUT=args.output.resolve()
if OUT.exists() and any(OUT.iterdir()):raise SystemExit('Output directory must be empty to prevent stale-result reuse.')
if args.workers<1:raise SystemExit('Workers must be positive')
OUT.mkdir(parents=True,exist_ok=True)
rows=[r for r in csv.DictReader((DATA/'MANIFEST.csv').open()) if r['dataset']=='fiber_length']
mask=np.load(DATA/'masks/fiber_length_valid_pixel_mask.npy').astype(bool);cv2.setNumThreads(1)
lookup={(int(r['fiber_length_cm']),r['device'],r['acquisition_round'],r['challenge'],r['acquisition_role']):r for r in rows}
def read(r,dark=None):
 p=DATA/r['release_path'];cap=cv2.VideoCapture(str(p));fps=cap.get(cv2.CAP_PROP_FPS);n=int(r['decoded_frame_count']);start=round(10*fps) if dark is not None else 0;end=n-round(10*fps) if dark is not None else n
 if end<=start:raise ValueError(p)
 frames=[];total=0;acc=None
 while True:
  ok,f=cap.read()
  if not ok:break
  if dark is None:
   x=f[:,:,1].astype(np.float64);acc=x if acc is None else acc+x
  elif start<=total<end:
   frames.append(np.maximum(f[:,:,1].astype(np.float64)-dark,0).astype(np.float32))
  total+=1
 cap.release()
 if total!=n:raise ValueError((p,total,n))
 if hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Source content changed')
 if dark is None:return acc/n
 im=np.median(np.stack(frames),axis=0).astype(np.float32).astype(np.float64)
 del frames
 if args.include_spatial:
  from puf_common.psd_features import extract_psd_features
  mets=extract_psd_features(im.astype(np.float32),mask,n_radial_bins=24)
  entropy=float(mets['psd_spectral_entropy'])
  if not np.isfinite(entropy):raise ValueError('Undefined PSD entropy')
  folder=OUT/'spatial_per_video';folder.mkdir(exist_ok=True)
  tag=f"L{int(r['fiber_length_cm']):02}_{r['device']}_{r['acquisition_round'].replace(' ','_')}_{r['challenge']}"
  (folder/(tag+'.json')).write_text(json.dumps({'L_cm':int(r['fiber_length_cm']),'device':r['device'],'acquisition_round':r['acquisition_round'],'challenge':r['challenge'],'H_PSD':entropy,'raw_release_path':r['release_path']}))
  if r['challenge']=='C01' and r['acquisition_round']=='Round A':np.save(folder/(tag+'_raw_intensity.npy'),im)
 return (im/(cv2.GaussianBlur(im,(0,0),sigmaX=42,sigmaY=42,borderType=cv2.BORDER_REFLECT101)+1)-1)[mask]
def device(item):
 L,f=item;dark=read(lookup[L,f,'','','dark_reference']);result={}
 for rnd in ['A','B']:
  a=np.stack([read(lookup[L,f,'Round '+rnd,f'C{k:02}','green_credential_response'],dark) for k in range(1,9)])
  a-=a.mean(axis=0);a-=a.mean(axis=1,keepdims=True);a/=np.linalg.norm(a,axis=1,keepdims=True);result[rnd]=a
 print('processed',L,f,flush=True);return f,result
all_rows=[];metrics=[];t=time.time()
for L in [7,9,11,13,15]:
 target=OUT/f'L_{L:02}_cross_round_ncc.npy'
 with ThreadPoolExecutor(max_workers=args.workers) as pool:d=dict(pool.map(device,[(L,f'F{i:02}') for i in range(1,6)]))
 a=np.concatenate([d[f'F{i:02}']['A'] for i in range(1,6)]);b=np.concatenate([d[f'F{i:02}']['B'] for i in range(1,6)]);matrix=a@b.T;np.save(target,matrix);del d,a,b;gc.collect()
 current=[]
 def pair(i,k,j,l,kind,value):
  current.append(dict(fiber_length_cm=L,similarity_class=kind,device_a=f'F{i+1:02}',challenge_a=f'C{k+1:02}',device_b=f'F{j+1:02}',challenge_b=f'C{l+1:02}',green_credential_score=float(value)))
 for i in range(5):
  for k in range(8):pair(i,k,i,k,'same_device_same_challenge_similarity',matrix[i*8+k,i*8+k])
  for k in range(8):
   for l in range(k+1,8):pair(i,k,i,l,'inter_challenge_similarity',(matrix[i*8+k,i*8+l]+matrix[i*8+l,i*8+k])/2)
 for i in range(5):
  for j in range(i+1,5):
   for k in range(8):pair(i,k,j,k,'inter_device_similarity',(matrix[i*8+k,j*8+k]+matrix[j*8+k,i*8+k])/2)
 g,c,d=[np.array([x['green_credential_score'] for x in current if x['similarity_class']==kind]) for kind in ['same_device_same_challenge_similarity','inter_challenge_similarity','inter_device_similarity']]
 qg,qc,qd=np.quantile(g,.05),np.quantile(c,.95),np.quantile(d,.95)
 metrics.append(dict(fiber_length_cm=L,genuine_count=len(g),inter_challenge_count=len(c),inter_device_count=len(d),genuine_lower_tail_quantile=float(qg),inter_challenge_upper_tail_quantile=float(qc),inter_device_upper_tail_quantile=float(qd),minimum_separation_margin=float(qg-max(qc,qd))))
 all_rows+=current
 for name,data in [('pair_scores.csv',all_rows),('length_metrics.csv',metrics)]:
  with (OUT/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
 print('completed length',L,'elapsed',round(time.time()-t,1),metrics[-1],flush=True)
(OUT/'raw_reproduction_provenance.json').write_text(json.dumps({'source_package':'Zenodo_release','source_files':[{'path':r['release_path'],'sha256':r['sha256']} for r in rows if r['optical_channel']=='green' or r['acquisition_role']=='dark_reference'],'mask_sha256':hashlib.sha256((DATA/'masks/fiber_length_valid_pixel_mask.npy').read_bytes()).hexdigest(),'numpy':np.__version__,'opencv':cv2.__version__,'protocol':'Native green plane; matched dark mean; discard first/last round(10*fps) decoded frames; central pixelwise median; local Gaussian sigma42 epsilon1 REFLECT101; eight-challenge common per device/round; reciprocal cross-round mismatch NCC means. Exact identifiers retained.'},indent=2))

if args.include_spatial:
 import pandas as pd
 records=[json.loads(p.read_text()) for p in sorted((OUT/'spatial_per_video').glob('*.json'))]
 assert len(records)==400
 pd.DataFrame(records).to_csv(OUT/'H_PSD_per_video.csv',index=False)
