#!/usr/bin/env python3
"""Recompute polishing-depth and AFM roughness statistics from measured source files.
AFM exports retain their two-line instrument header. Heights are in nm.
Local S_q uses each 256x256 region's own mean; full S_q uses the full scan mean.
"""
import argparse,csv,json,hashlib,io,zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np

def depth_rows(path):
 # Read existing values without editing or requiring a spreadsheet engine.
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile(path) as z:
  strings=[]
  if 'xl/sharedStrings.xml' in z.namelist():
   strings=[''.join(n.itertext()) for n in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',ns)]
  root=ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
  for row in root.findall('.//s:sheetData/s:row',ns)[1:]:
   cells={}
   for c in row.findall('s:c',ns):
    v=c.find('s:v',ns);value=v.text if v is not None else ''.join(c.find('s:is',ns).itertext()) if c.find('s:is',ns) is not None else ''
    if c.get('t')=='s':value=strings[int(value)]
    cells[''.join(x for x in c.get('r') if x.isalpha())]=value
   for i,col in enumerate('BCDEF',1):yield {'device':cells['A'],'position':i,'depth_um':float(cells[col])}

def write_csv(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args();a.output_root.mkdir(parents=True,exist_ok=True)
 raw=a.data_root/'raw_data/fabrication_characterization';depth=list(depth_rows(raw/'side_polishing_depth_data_v4.xlsx'));assert len(depth)==75
 write_csv(a.output_root/'Fig2f_local_depth_um.csv',depth)
 dev=[]
 for d in sorted(set(r['device'] for r in depth)):
  x=np.array([r['depth_um'] for r in depth if r['device']==d]);dev.append({'device':d,'n_positions':len(x),'mean_depth_um':float(x.mean()),'sample_SD_um':float(x.std(ddof=1))})
 write_csv(a.output_root/'Fig2f_device_depth_summary.csv',dev)
 local=[];summary=[]
 for i in range(1,6):
  h=np.loadtxt(raw/'AFM'/f'Fiber{i:02}.csv',skiprows=2);assert h.shape==(1024,1024)
  vals=[]
  for br in range(4):
   for bc in range(4):
    v=float(h[br*256:(br+1)*256,bc*256:(bc+1)*256].std(ddof=0));vals.append(v);local.append({'device':f'F{i:02}','block_row':br+1,'block_column':bc+1,'local_S_q_nm':v})
  summary.append({'device':f'F{i:02}','mean_local_S_q_nm':float(np.mean(vals)),'sample_SD_local_S_q_nm':float(np.std(vals,ddof=1)),'full_scan_S_q_nm':float(h.std(ddof=0))})
 write_csv(a.output_root/'Fig2d_local_S_q_nm.csv',local);write_csv(a.output_root/'Fig2d_device_S_q_summary.csv',summary)
 x=np.array([r['depth_um'] for r in depth]);res={'n_local_depth_measurements':len(x),'overall_depth_mean_um':float(x.mean()),'overall_depth_sample_SD_um':float(x.std(ddof=1)),'device_mean_min_um':min(r['mean_depth_um'] for r in dev),'device_mean_max_um':max(r['mean_depth_um'] for r in dev),'AFM':summary,'input_sha256':{str(f.relative_to(a.data_root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in raw.rglob('*') if f.is_file()}}
 (a.output_root/'raw_reproduction_summary.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
if __name__=='__main__':main()
