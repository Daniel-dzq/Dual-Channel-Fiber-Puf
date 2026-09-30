#!/usr/bin/env python3
"""Decode the recorded Fig. 4(b) display identities; verify their NCC and selection.

The publication's display identities are explicit inputs because the two central
observations of an even-sized group are mathematically tied around its median.
Re-selecting by binary floating-point distance can change the displayed recording.
"""
import argparse,csv,hashlib,json
from pathlib import Path
import cv2,numpy as np,pandas as pd

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['data-root','pair-scores','selection','output']:p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();root=a.data_root.resolve();a.output.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
 rows=[r for r in csv.DictReader((root/'MANIFEST.csv').open()) if r['dataset']=='fiber_length'];lookup={(int(r['fiber_length_cm']),r['device'],r['acquisition_round'],r['challenge'],r['acquisition_role']):r for r in rows};mask=np.load(root/'masks/fiber_length_valid_pixel_mask.npy').astype(bool);pairs=pd.read_csv(a.pair_scores);selection=pd.read_csv(a.selection);audit=[]
 def read(r,dark=None):
  path=root/r['release_path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256'];cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);n=int(r['decoded_frame_count']);start=round(10*fps) if dark is not None else 0;end=n-round(10*fps) if dark is not None else n;frames=[];acc=None;count=0
  while True:
   ok,f=cap.read()
   if not ok:break
   if dark is None:
    x=f[:,:,1].astype(np.float64);acc=x if acc is None else acc+x
   elif start<=count<end:frames.append(np.maximum(f[:,:,1].astype(np.float64)-dark,0).astype(np.float32))
   count+=1
  cap.release();assert count==n
  return acc/n if dark is None else np.median(np.stack(frames),axis=0).astype(np.float32)
 for sel in selection.itertuples(index=False):
  L=int(sel.L_cm);device=sel.device_id;challenge=sel.challenge_id;g=pairs[(pairs.fiber_length_cm==L)&(pairs.similarity_class=='same_device_same_challenge_similarity')];target=float(g[(g.device_a==device)&(g.challenge_a==challenge)].green_credential_score.iloc[0]);median=float(g.green_credential_score.median());min_distance=float((g.green_credential_score-median).abs().min());excess=abs(target-median)-min_distance;assert excess<1e-10
  dark=read(lookup[L,device,'','','dark_reference']);vectors={};raw={};sources=[]
  for rnd in ['A','B']:
   details=[]
   for k in range(1,9):
    cid=f'C{k:02}';r=lookup[L,device,'Round '+rnd,cid,'green_credential_response'];im=read(r,dark);x=im.astype(np.float64);detail=(x/(cv2.GaussianBlur(x,(0,0),sigmaX=42,sigmaY=42,borderType=cv2.BORDER_REFLECT101)+1)-1)[mask];details.append(detail);sources.append({'path':r['release_path'],'sha256':r['sha256']})
    if cid==challenge:raw[rnd]=im
   details=np.stack(details);details-=details.mean(axis=0);vectors[rnd]=details[int(challenge[1:])-1]
  x=vectors['A']-vectors['A'].mean();y=vectors['B']-vectors['B'].mean();ncc=float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)));assert abs(ncc-target)<1e-9
  np.savez_compressed(a.output/f'Fig4b_L{L:02}_{device}_{challenge}.npz',Round_A_raw_intensity=raw['A'],Round_B_raw_intensity=raw['B'],absolute_intensity_difference=np.abs(raw['A']-raw['B']),Round_A_detail_cm_masked=vectors['A'],Round_B_detail_cm_masked=vectors['B'])
  audit.append({'L_cm':L,'device_id':device,'challenge_id':challenge,'NCC':ncc,'group_median':median,'selection_distance_excess':excess,'raw_files':sources,'dark_reference':lookup[L,device,'','','dark_reference']['release_path']});print('Exported',L,device,challenge,ncc,flush=True)
 (a.output/'Fig4b_provenance.json').write_text(json.dumps({'selection':'Published display identities retained; each is nearest to the group median within floating-point precision.','masked_vector_order':'C-order boolean indexing by masks/fiber_length_valid_pixel_mask.npy','display_rule':'Numerical fields only; display transformations remain those of the original figure code.','records':audit},indent=2))
if __name__=='__main__':main()
