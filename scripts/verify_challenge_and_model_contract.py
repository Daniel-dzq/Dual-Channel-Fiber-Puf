#!/usr/bin/env python3
"""Check archived challenge pixels and the numerical constants in SI Table S12.

This is a contract check, not a substitute for fitting and evaluating all attacks.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from experiment4_security.ml_attack.challenge_features import build_challenge_features
from experiment4_security.ml_attack.partial_leakage_models import effective_pca_dimension

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 repo=Path(__file__).resolve().parents[1];root=a.data_root.resolve();lib=root/'challenge_library/formal_challenges';m=pd.read_csv(lib/'challenge_manifest.csv');ids=list(m.challenge_id);checks=[];grids={}
 def check(name,ok,details=None):checks.append({'check':name,'pass':bool(ok),'details':details})
 check('128 unique challenge identities',len(ids)==128 and len(set(ids))==128)
 for r in m.itertuples(index=False):
  f=lib/(r.challenge_id+'.png');canvas=np.asarray(Image.open(f).convert('L'));active=canvas[r.active_offset_y_px:r.active_offset_y_px+r.active_height_px,r.active_offset_x_px:r.active_offset_x_px+r.active_width_px];grid=(active[::2,::2]==255)
  check(r.challenge_id+' SHA256',hashlib.sha256(f.read_bytes()).hexdigest()==r.sha256)
  check(r.challenge_id+' binary constant 2x2 cells',np.isin(canvas,[0,255]).all() and np.array_equal(active,np.repeat(np.repeat(grid.astype(np.uint8)*255,2,0),2,1)))
  check(r.challenge_id+' 32768 open logical cells',grid.shape==(256,256) and grid.sum()==32768)
  grids[r.challenge_id]=grid
 features=build_challenge_features(ids,manifest=m,resolved_paths={c:lib/(c+'.png') for c in ids})
 for cid in ids:
  independent=np.array([[grids[cid][y:y+8,x:x+8].mean() for x in range(0,256,8)] for y in range(0,256,8)]).ravel(order='C')
  check(cid+' published 1024-component input',np.array_equal(independent,features[cid].bitmap_32))
 hd=pd.read_csv(lib/'pairwise_hamming_distance.csv');errors=[]
 for r in hd.itertuples(index=False):errors.append(abs(float(np.mean(grids[r.challenge_id_a]!=grids[r.challenge_id_b]))-r.normalized_hamming))
 keys={tuple(sorted((r.challenge_id_a,r.challenge_id_b))) for r in hd.itertuples(index=False)}
 check('All 8128 unordered challenge Hamming distances',len(hd)==len(keys)==8128 and max(errors)==0,{'max_absolute_difference':max(errors)})
 check('Response mask dimension',int(np.load(root/'masks/threshold_development_and_formal_valid_pixel_mask.npy').astype(bool).sum())==1676190)
 expected={'ridge_alpha':10.,'kernel_ridge_alpha':1.,'kernel_ridge_gamma':.01,'rff_n_components':256,'rff_gamma':.01,'rff_alpha':1.,'mlp_hidden_sizes':[64,32],'mlp_max_iter':150,'mlp_learning_rate':.001,'mlp_random_seed':20260721}
 for name in ['formal_green_preprocessing.yaml','partial_disclosure.yaml']:
  cfg=yaml.safe_load((repo/'configs'/name).read_text())
  for key,value in expected.items():check(name+': '+key,cfg['models'][key]==value,{'observed':cfg['models'][key],'SI_S12':value})
 pcfg=yaml.safe_load((repo/'configs/partial_disclosure.yaml').read_text())
 check('Partial disclosure seeds',pcfg['seeds']==list(range(20260801,20260806)))
 check('Partial disclosure dimensions',[effective_pca_dimension(n,pcfg['pca_dimension_max']) for n in [16,32,64,96]]==[15,31,63,64])
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'scope':'Actual pixels, archived Hamming table, response-mask size and SI S12 numerical configuration; does not certify full attack results or every prose statement.','checks':checks,'passed':sum(x['pass'] for x in checks),'total':len(checks)},indent=2))
 print(sum(x['pass'] for x in checks),'/',len(checks));assert all(x['pass'] for x in checks)
if __name__=='__main__':main()
