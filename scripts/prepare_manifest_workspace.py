#!/usr/bin/env python3
"""Stage authentic manifest-format recordings for the public raw pipelines.

Creates links only to files inside --data-root. Never changes recordings or uses
laboratory directories. Generated YAML files override paths, not scientific settings.
"""
from pathlib import Path
import argparse,csv,hashlib,json
import yaml

REPO=Path(__file__).resolve().parents[1]
def prepare(root,out):
 root=root.resolve();out=out.resolve();out.mkdir(parents=True,exist_ok=True)
 rows=list(csv.DictReader((root/'MANIFEST.csv').open()))
 links=[]
 for r in rows:
  dataset=r['dataset'];rel=None
  if dataset=='wavelength_pathway_control':
   rel=Path('wavelength_pathway/videos')/f"M{int(r['technical_repeat'])-1}_{r['injection_pathway']}_{r['optical_channel']}_F{int(r['device'][1:])}.mp4"
  elif dataset=='macro_pixel_screening' and r['optical_channel']=='green':
   rel=Path('macro_pixel/videos')/f"mp{int(r['macro_pixel_size']):03}"/f"{int(r['challenge'][1:])}.mp4"
  if rel is None:continue
  src=(root/r['release_path']).resolve()
  if not src.is_relative_to(root):raise ValueError(f'Input escapes data root: {src}')
  if hashlib.sha256(src.read_bytes()).hexdigest()!=r['sha256']:raise ValueError(f'Hash mismatch: {src}')
  dst=out/rel;dst.parent.mkdir(parents=True,exist_ok=True)
  if dst.is_symlink():
   if dst.resolve()!=src:raise ValueError(f'Conflicting link: {dst}')
  elif dst.exists():raise ValueError(f'Unexpected existing file: {dst}')
  else:dst.symlink_to(src)
  links.append({'workspace_path':str(rel),'release_path':r['release_path'],'sha256':r['sha256']})
 for name in ['wavelength_pathway','macro_pixel','fixed_state']:
  config_name={'macro_pixel':'macro_pixel_screening','fixed_state':'fixed_state_dual_channel','wavelength_pathway':'wavelength_pathway_control'}[name]
  cfg=yaml.safe_load((REPO/'configs'/f'{config_name}.yaml').read_text())
  if name=='wavelength_pathway':cfg['paths'].update(root=str(out/name),videos_root=str(out/name/'videos'),output_dir=str(out/name/'outputs'))
  elif name=='macro_pixel':
   cfg['experiment']['root_dir']=str(out/name)
   cfg['analysis']['dark_artifact_path']=str(root/'metadata/calibration/dark_reference.npz')
  else:
   cfg['paths'].update(root=str(root),metadata_csv=str(out/'fixed_state_metadata.csv'),output_dir=str(out/name/'outputs'),videos_root=str(root))
   cfg['dark_frame']['experiment1_dark_artifact']=str(root/'metadata/calibration/dark_reference.npz')
   cfg['dark_frame']['experiment1_valid_mask']=str(root/'masks/macro_pixel_and_fixed_state_valid_pixel_mask.png')
   cfg['challenges']['source_dir']=str(REPO/'data/challenge_patterns/mp002')
   meta=list(csv.DictReader((root/'metadata/fixed_state_acquisition.csv').open()))
   for row in meta:
    p=(root/row['video_path']).resolve()
    if not p.is_relative_to(root) or not p.is_file():raise ValueError(p)
    row['video_path']=str(p)
   with (out/'fixed_state_metadata.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(meta[0]));writer.writeheader();writer.writerows(meta)
  (out/f'{name}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
 (out/'staged_inputs.json').write_text(json.dumps(links,indent=2))
 print(f'Staged {len(links)} verified video links; wrote three raw-pipeline configs under {out}')
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args();prepare(a.data_root,a.output_root)
