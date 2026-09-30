#!/usr/bin/env python3
"""Bounded-disk raw reproduction of formal green A/B and complete/partial attacks.

Reads only the manifest-format public data and public analysis modules. Processes
one device at a time, retaining its vectors through all attack evaluations, then
removes only those newly generated temporary vectors. Intermediate source tables
are retained. Cross-device and red-identity stages are separate, not implied here.
"""
import argparse,csv,gc,hashlib,json,logging,shutil,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2,numpy as np,pandas as pd
from experiment4_security.ml_attack.config import MLAttackConfig
from experiment4_security.ml_attack.video_preprocessing import process_and_cache_clip
from experiment4_security.identity_credential.vector_provenance import response_provenance,cache_matches,write_provenance
from experiment4_security.identity_credential.green_result_adapter import build_device_commons
from experiment4_security.ml_attack.track_a_database_auth import run_track_a
from experiment4_security.ml_attack.enrollment import enrollment_templates,query_vectors
from experiment4_security.ml_attack.batch_eval import evaluate_cross_state_credential
from experiment4_security.identity_credential.green_clone_tracks import run_green_tracks_cd_multidevice
from experiment4_security.ml_attack.challenge_features import build_challenge_features
from experiment4_security.ml_attack.partial_leakage_splits import load_bank_map,build_split_manifest,all_splits_by_repetition
from experiment4_security.ml_attack.partial_leakage_models import load_hamming_distance_matrix
from experiment4_security.ml_attack.partial_leakage_pipeline import PartialLeakageConfig
from experiment4_security.ml_attack.track_c_pl_partial_leakage import load_round_vectors,run_device_source_state

REPO=Path(__file__).resolve().parents[1]
IDS=[f'C{i:03}' for i in range(1,129)];STATES=[f'M{i}' for i in range(8)]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
def stage_valid(directory,stage,identity):
 marker=directory/(stage+'_COMPLETE.json')
 if not marker.exists():return False
 record=json.loads(marker.read_text())
 return record['input_contract']==identity and all((directory/name).is_file() and sha(directory/name)==digest for name,digest in record['outputs'].items())
def mark_stage(directory,stage,identity,names):
 record={'input_contract':identity,'outputs':{name:sha(directory/name) for name in names}}
 (directory/(stage+'_COMPLETE.json')).write_text(json.dumps(record,indent=2))
def save(df,p):df.to_csv(p,index=False,compression='gzip' if p.suffix=='.gz' else None)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--devices',nargs='+',default=[f'F{i:02}' for i in range(1,11)]);p.add_argument('--workers',type=int,default=2);p.add_argument('--skip-attacks',action='store_true');a=p.parse_args()
 root=a.data_root.resolve();out=a.output_root.resolve();out.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
 logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
 rows=list(csv.DictReader((root/'MANIFEST.csv').open()));rows=[r for r in rows if r['dataset']=='formal_mechanical_reconfiguration' and r['optical_channel']=='green']
 assert len(rows)==20480
 mask=np.load(root/'masks/threshold_development_and_formal_valid_pixel_mask.npy').astype(bool)
 cfg=MLAttackConfig.from_yaml(REPO/'configs/formal_green_preprocessing.yaml');pcfg=PartialLeakageConfig.from_yaml(REPO/'configs/partial_disclosure.yaml')
 lib=root/'challenge_library/formal_challenges';cm=pd.read_csv(lib/'challenge_manifest.csv');feat=build_challenge_features(IDS,manifest=cm,resolved_paths={c:lib/(c+'.png') for c in IDS});features={c:v.bitmap_32 for c,v in feat.items()};del feat
 bankmap=load_bank_map(lib/'challenge_manifest.csv');banks={c:b for b,ids in bankmap.items() for c in ids};sm=build_split_manifest(pcfg.seeds,bankmap,leak_sizes=pcfg.leak_sizes,per_bank_counts=pcfg.per_bank_leak_counts);save(sm,out/'split_manifest.csv');splits=all_splits_by_repetition(sm)
 ham=load_hamming_distance_matrix(str(lib/'pairwise_hamming_distance.csv'),IDS)
 identity={'manifest_sha256':sha(root/'MANIFEST.csv'),'mask_sha256':sha(root/'masks/threshold_development_and_formal_valid_pixel_mask.npy'),'green_config_sha256':sha(REPO/'configs/formal_green_preprocessing.yaml'),'partial_config_sha256':sha(REPO/'configs/partial_disclosure.yaml'),'analysis_sources_sha256':hashlib.sha256(''.join(str(f.relative_to(REPO))+sha(f) for f in sorted((REPO/'src').rglob('*.py'))).encode()).hexdigest()}
 ip=out/'input_contract.json'
 if ip.exists() and json.loads(ip.read_text())!=identity:raise RuntimeError('Input/code contract changed: use a fresh output directory')
 ip.write_text(json.dumps(identity,indent=2))
 for device in a.devices:
  d=out/device;d.mkdir(exist_ok=True)
  required_tracks={'A','B'} if a.skip_attacks else {'A','B','C','D','C-PL','D-PL'}
  if (d/'COMPLETE.json').exists() and required_tracks.issubset(json.loads((d/'COMPLETE.json').read_text())['tracks']):continue
  cache=d/'temporary_cache';vd=cache/'vectors';vd.mkdir(parents=True,exist_ok=True)
  subset=sorted([r for r in rows if r['device']==device],key=lambda r:(r['mechanical_state'],r['acquisition_round'],r['challenge']));assert len(subset)==2048
  if shutil.disk_usage(out).free+sum(f.stat().st_size for f in vd.glob('*.npy'))<18*1024**3:raise RuntimeError('Need at least 18 GiB free before starting a device')
  def decode(r):
   source=(root/r['release_path']).resolve();assert source.is_relative_to(root)
   if sha(source)!=r['sha256']:raise RuntimeError(f'Recording hash mismatch: {source}')
   sid=f"{device}_{r['mechanical_state']}_{r['acquisition_round'][-1]}_{r['challenge']}";v=vd/(sid+'.npy');expected=response_provenance(source,mask,cfg.preprocessing)
   if cache_matches(v,expected):return {'sample_id':sid,'raw_sha256':r['sha256'],'vector_sha256':sha(v),'reused_verified_checkpoint':True}
   qc=process_and_cache_clip(source,sample_id=sid,valid_mask=mask,cfg=cfg,vector_cache_dir=vd)
   if qc['decode_status']!='OK' or qc['qc_status']=='ERROR':raise RuntimeError(f'QC failed {sid}: {qc}')
   write_provenance(v,expected)
   return {'sample_id':sid,'raw_sha256':r['sha256'],'vector_sha256':sha(v),'qc':qc,'reused_verified_checkpoint':False}
  with (d/'response_provenance.jsonl').open('w') as f,ThreadPoolExecutor(max_workers=a.workers) as pool:
   for i,r in enumerate(pool.map(decode,subset),1):
    f.write(json.dumps(r,default=str)+'\n');f.flush()
    if i%32==0:logging.info('%s decoded and verified %s/2048',device,i)
  def lookup(s,r,c):return np.load(vd/f'{device}_{s}_{r}_{c}.npy',mmap_mode='r')
  ab_names=['track_a_database_authentication_summary.csv','track_a_scores.csv.gz','track_b_template_transfer_matrix_summary.csv','track_b_scores.csv.gz']
  if not stage_valid(d,'AB',identity):
   commons=build_device_commons(device,STATES,IDS,detail_lookup=lookup)
   ts,ss,meta=run_track_a(STATES,IDS,commons,detail_lookup=lookup,device_id=device);ts['device_id']=device;save(ts,d/'track_a_database_authentication_summary.csv');save(ss,d/'track_a_scores.csv.gz')
   bs=[];bsc=[]
   for source in STATES:
    templates=enrollment_templates(commons[source],IDS,detail_lookup=lookup)
    for target in STATES:
     queries=query_vectors(commons[source],target,IDS,detail_lookup=lookup);ev=evaluate_cross_state_credential(templates,queries,IDS,device_id=device,source_state=source,target_state=target,genuine_same_state=meta['genuine_by_state'][source]);ev['summary_row']['device_id']=device;bs.append(ev['summary_row']);bsc+=ev['score_rows'];del queries
    del templates;gc.collect();logging.info('%s Track B source %s complete',device,source)
   save(pd.DataFrame(bs),d/'track_b_template_transfer_matrix_summary.csv');save(pd.DataFrame(bsc),d/'track_b_scores.csv.gz');del bsc,commons;gc.collect()
   mark_stage(d,'AB',identity,ab_names)
  genuine_table=pd.read_csv(d/'track_a_scores.csv.gz')
  genuine_by_state={state:genuine_table[(genuine_table.source_state==state)&genuine_table.is_genuine]['score'].to_numpy() for state in STATES}
  del genuine_table;gc.collect()
  if not a.skip_attacks:
   cd_names=['track_c_same_state_clone_summary.csv','track_d_clone_transfer_matrix_summary.csv','track_c_scores.csv.gz','track_d_scores.csv.gz']
   if not stage_valid(d,'CD',identity):
    cd=run_green_tracks_cd_multidevice([device],STATES,IDS,features,detail_lookups={device:lookup},genuine_by_device_state={device:genuine_by_state},models_cfg=cfg.models,pca_dimension=64,n_parallel_targets=1,checkpoint_dir=d/'cd_checkpoints')
    for key,name in [('track_c_summary','track_c_same_state_clone_summary.csv'),('track_d_summary','track_d_clone_transfer_matrix_summary.csv'),('track_c_scores','track_c_scores.csv.gz'),('track_d_scores','track_d_scores.csv.gz')]:save(cd[key],d/name)
    del cd;gc.collect()
    mark_stage(d,'CD',identity,cd_names)
   for model in (d/'cd_checkpoints').rglob('track_c_predictors_*.joblib'):
    record={'path':str(model.relative_to(out)),'bytes':model.stat().st_size,'sha256':sha(model),'reason':'C/D exports verified; temporary fitted predictor can be regenerated'}
    with (out/'removed_temporary_models.jsonl').open('a') as log:log.write(json.dumps(record)+'\n')
    model.unlink()
   # Persist one source/split at a time so a long model run can resume safely.
   parts=d/'partial';parts.mkdir(exist_ok=True)
   b={s:load_round_vectors(cache,device,s,'B',IDS) for s in STATES}
   for source in STATES:
    for rep,leaks in splits.items():
     for size,split in leaks.items():
      sf=parts/f'{source}_rep{rep}_N{size}.csv';hf=parts/f'{source}_rep{rep}_N{size}_scores.csv.gz'
      if sf.exists() and hf.exists():continue
      r=run_device_source_state(device,source,shared_cache_root=cache,all_challenge_ids=IDS,challenge_to_bank=banks,challenge_features=features,hamming_matrix=ham,splits_by_rep={rep:{size:split}},models_cfg=pcfg.models_cfg,pca_dimension_max=pcfg.pca_dimension_max,b_vectors_all_states=b,states_for_transfer=STATES)
      save(r['transfer_hidden_scores'],hf);save(r['transfer_summary'],sf);del r;gc.collect();logging.info('%s partial %s rep=%s N_L=%s complete',device,source,rep,size)
   del b;gc.collect()
   merged=pd.concat([pd.read_csv(f) for f in sorted(parts.glob('*.csv'))],ignore_index=True);save(merged,d/'track_d_pl_partial_leakage_transfer_summary.csv');save(merged[merged.is_diagonal].drop(columns=['target_state','is_diagonal']),d/'track_c_pl_partial_leakage_summary.csv')
  (d/'COMPLETE.json').write_text(json.dumps({'device':device,'raw_green_videos':2048,'tracks':['A','B']+([] if a.skip_attacks else ['C','D','C-PL','D-PL']),'cross_device_and_red_included':False},indent=2))
  del genuine_by_state;gc.collect();shutil.rmtree(cache)
  logging.info('%s COMPLETE; temporary raw-response cache removed',device)
 logging.info('Requested device stages complete. Cross-device and red stages are separate.')
if __name__=='__main__':main()
