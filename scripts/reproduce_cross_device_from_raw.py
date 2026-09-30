#!/usr/bin/env python3
"""Reproduce formal cross-device S_D, one mechanical state at a time.

Temporary vectors are decoded from the public package, verified by content and
protocol, then removed only after that state's score tables are saved. No
laboratory caches are read. Approximately 17 GiB of temporary disk is required.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,csv,json,hashlib,logging,shutil,gc
import numpy as np,pandas as pd,cv2
from experiment4_security.ml_attack.config import MLAttackConfig
from experiment4_security.ml_attack.video_preprocessing import process_and_cache_clip
from experiment4_security.identity_credential.vector_provenance import response_provenance,cache_matches,write_provenance
from experiment4_security.ml_attack.enrollment import fit_enrollment_common
from experiment4_security.ml_attack.batch_eval import row_zero_mean_ncc
from experiment4_security.identity_credential.green_multidevice_metrics import evaluate_device_mismatch_for_state
REPO=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=2);a=p.parse_args();root=a.data_root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1);logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
 rows=[r for r in csv.DictReader((root/'MANIFEST.csv').open()) if r['dataset']=='formal_mechanical_reconfiguration' and r['optical_channel']=='green'];assert len(rows)==20480
 mask=np.load(root/'masks/threshold_development_and_formal_valid_pixel_mask.npy').astype(bool);cfg=MLAttackConfig.from_yaml(REPO/'configs/formal_green_preprocessing.yaml');ids=[f'C{i:03}' for i in range(1,129)];devices=[f'F{i:02}' for i in range(1,11)]
 contract={'manifest':sha(root/'MANIFEST.csv'),'mask':sha(root/'masks/threshold_development_and_formal_valid_pixel_mask.npy'),'config':sha(REPO/'configs/formal_green_preprocessing.yaml'),'source':hashlib.sha256(''.join(str(f.relative_to(REPO))+sha(f) for f in sorted((REPO/'src').rglob('*.py'))).encode()).hexdigest()};cp=out/'input_contract.json'
 if cp.exists() and json.loads(cp.read_text())!=contract:raise RuntimeError('Input/code changed; use a fresh output directory')
 cp.write_text(json.dumps(contract,indent=2))
 for state in [f'M{i}' for i in range(8)]:
  d=out/state;d.mkdir(exist_ok=True)
  if (d/'COMPLETE.json').exists():continue
  cache=d/'temporary_vectors';cache.mkdir(exist_ok=True)
  already=sum(f.stat().st_size for f in cache.glob('*.npy'))
  if shutil.disk_usage(out).free+already<22*1024**3:raise RuntimeError('Need 22 GiB available including resumable temporary vectors')
  selected=sorted([r for r in rows if r['mechanical_state']==state],key=lambda r:r['release_path']);assert len(selected)==2560
  def decode(r):
   source=(root/r['release_path']).resolve();assert source.is_relative_to(root)
   if sha(source)!=r['sha256']:raise RuntimeError(f'Raw hash mismatch: {source}')
   sid=f"{r['device']}_{state}_{r['acquisition_round'][-1]}_{r['challenge']}";target=cache/(sid+'.npy');expected=response_provenance(source,mask,cfg.preprocessing)
   reused=cache_matches(target,expected)
   if not reused:
    qc=process_and_cache_clip(source,sample_id=sid,valid_mask=mask,cfg=cfg,vector_cache_dir=cache)
    if qc['decode_status']!='OK' or qc['qc_status']=='ERROR':raise RuntimeError(qc)
    write_provenance(target,expected)
   return dict(sample_id=sid,raw_sha256=r['sha256'],vector_sha256=sha(target),reused_verified_checkpoint=reused)
  with (d/'response_provenance.jsonl').open('w') as log,ThreadPoolExecutor(max_workers=a.workers) as pool:
   for i,rec in enumerate(pool.map(decode,selected),1):
    log.write(json.dumps(rec)+'\n');log.flush()
    if i%64==0:logging.info('%s raw %s/2560',state,i)
  lookups={dev:(lambda s,r,c,dev=dev:np.load(cache/f'{dev}_{s}_{r}_{c}.npy',mmap_mode='r')) for dev in devices};commons={};genuine={}
  for dev in devices:
   co=fit_enrollment_common(state,ids,detail_lookup=lookups[dev],device_id=dev);commons[dev,state]=co
   # Score each matching challenge separately, avoiding an additional full matrix.
   genuine[dev]=np.array([row_zero_mean_ncc(co.to_detail_cm(lookups[dev](state,'A',c))[None,:],co.to_detail_cm(lookups[dev](state,'B',c))[None,:])[0] for c in ids]);gc.collect()
  result=evaluate_device_mismatch_for_state(state_id=state,challenge_ids=ids,devices=devices,commons=commons,detail_lookups=lookups,genuine_by_device=genuine)
  assert len(result['score_rows'])==11520
  pd.DataFrame(result['score_rows']).to_csv(d/'scores.csv.gz',index=False,compression='gzip');pd.DataFrame([result['summary']]).to_csv(d/'summary.csv',index=False)
  pd.DataFrame([{'device':dev,'state':state,'challenge':c,'S_G':float(v)} for dev in devices for c,v in zip(ids,genuine[dev])]).to_csv(d/'genuine_scores.csv',index=False)
  (d/'COMPLETE.json').write_text(json.dumps({'state':state,'raw_videos':2560,'S_D_scores':11520},indent=2));del result,commons,genuine,lookups;gc.collect();shutil.rmtree(cache);logging.info('%s completed and temporary vectors removed',state)
 pd.concat([pd.read_csv(out/f'M{i}/summary.csv') for i in range(8)],ignore_index=True).to_csv(out/'track_s_d_device_mismatch_summary.csv',index=False)
 pd.concat([pd.read_csv(out/f'M{i}/scores.csv.gz') for i in range(8)],ignore_index=True).to_csv(out/'track_s_d_scores.csv.gz',index=False,compression='gzip')
if __name__=='__main__':main()
