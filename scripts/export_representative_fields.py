#!/usr/bin/env python3
"""Export displayed camera fields with original display transforms and provenance."""
from pathlib import Path
import argparse,csv,json,hashlib,shutil
import cv2,numpy as np
from PIL import Image
from e01.analysis.plot_fig3 import _read_mid_frame_bgr,_bright_core_center,_crop_bright_spot_rgb
from experiment02b.config import load_config
from experiment02b.publication_figure import load_inputs,select_display_fiber,select_representative,build_panel_a_images,_speckle_rgb
REPO=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['data-root','workspace','output']:p.add_argument('--'+n,required=True,type=Path)
 a=p.parse_args();root=a.data_root.resolve();ws=a.workspace.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);rows=list(csv.DictReader((root/'MANIFEST.csv').open()));f3=out/'Fig3';f3.mkdir(exist_ok=True);frames={};source={}
 for m in [1,2,4,8,16,32,64]:
  r=next(r for r in rows if r['dataset']=='macro_pixel_screening' and r['optical_channel']=='green' and r['macro_pixel_size']==str(m) and r['challenge']=='C01');p=root/r['release_path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'];frames[m]=_read_mid_frame_bgr(p);source[m]=r
 side=max(max(_bright_core_center(b)[2] for b in frames.values()),256);side+=2*round(side*.18);meta=[]
 for m,frame in frames.items():
  Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).save(f3/f'Fig3b_m{m:03}_C01_camera_frame.png');Image.fromarray(_crop_bright_spot_rgb(frame,side=side,out_side=512)).save(f3/f'Fig3b_m{m:03}_C01_display.png')
  pattern=(REPO/'data/challenge_patterns/mp002/mp002_C01.png') if m==2 else root/f'challenge_library/macro_pixel_screening/m_{m:03}/C01.png';shutil.copy2(pattern,f3/f'Fig3b_m{m:03}_C01_SLM_input.png');meta.append({'m':m,'raw_release_path':source[m]['release_path'],'raw_sha256':source[m]['sha256'],'frame_rule':'15 seconds for recordings longer than 20 seconds; otherwise mid-frame','shared_crop_side_pixels':side,'display_transform':'Original bright-core crop, INTER_AREA resize to 512 pixels, channel-preserving RGB boost to 99.5th percentile with gain capped at 4. Camera-frame export has no display boost.'})
 (f3/'representative_provenance.json').write_text(json.dumps(meta,indent=2))
 f5=out/'Fig5';f5.mkdir(exist_ok=True);cfg=load_config(ws/'wavelength_pathway.yaml');data=load_inputs(ws/'wavelength_pathway/outputs');fiber,scores=select_display_fiber(data);rep=select_representative(data,display_fiber=fiber);rep['contrast_score_by_fiber']=scores;images,limits=build_panel_a_images(rep,cfg)
 for key,bgr in images.items():
  np.save(f5/f'Fig5c_{key}_median_BGR.npy',bgr)
  rgb=_speckle_rgb(bgr,int(key.split('_')[0]),limits[key])
  assert np.isfinite(rgb).all() and rgb.min()>=0 and rgb.max()<=1
  Image.fromarray(np.rint(rgb*255).astype(np.uint8)).save(f5/f'Fig5c_{key}_display.png')
 for value in rep['repeat_by_condition'].values():
  actual=Path(value.pop('source_path')).resolve();value['raw_release_path']=str(actual.relative_to(root));value['raw_sha256']=hashlib.sha256(actual.read_bytes()).hexdigest()
 rep['display_limits']=limits;rep['png_encoding']='Original display RGB in [0,1], rounded to uint8; median_BGR.npy retains the numerical field.';(f5/'Fig5c_representative_selection.json').write_text(json.dumps(rep,indent=2))
 fixed=ws/'fixed_state/outputs';items={'Fig5a_F02_red_before.npy':fixed/'templates/representative_templates/F02_red_before.npy','Fig5a_F02_red_after.npy':fixed/'fixed_state_analysis/_shared_cache/red_after_representatives/F02_red_after.npy','Fig5a_F10_C02_window1.npy':fixed/'templates/block_templates/F10_C02_block1.npy','Fig5a_F10_C02_window2.npy':fixed/'templates/block_templates/F10_C02_block2.npy','Fig5a_F10_green_C01.npy':fixed/'templates/representative_templates/F10_green_C01.npy'}
 for name,source_path in items.items():shutil.copy2(source_path,f5/name)
 (f5/'Fig5a_field_definition.json').write_text(json.dumps({'fields':list(items),'source':'Newly decoded fixed-state outputs; no historical intensity caches','red_absolute_difference':'abs(Fig5a_F02_red_before - Fig5a_F02_red_after)','green_same_challenge':'F10 C02 temporal windows 1 and 2','green_different_challenge':'F10 C01'},indent=2));print('Representative Fig3b, Fig5a,c fields exported',flush=True)
if __name__=='__main__':main()
