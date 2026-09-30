#!/usr/bin/env python3
"""Export Fig. 5 source tables from independently decoded raw-run outputs."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from experiment02b.publication_figure import cluster_bootstrap_stat,CONDITIONS,BOOTSTRAP_N

def ci(df,col,rng,seed):
 rows=[]
 for wl,geom in CONDITIONS:
  v=df[(df.wavelength_nm==wl)&(df.excitation_geometry==geom)].sort_values('fiber_id')[col].to_numpy(float)
  lo,hi,n=cluster_bootstrap_stat(v,np.median,rng)
  rows.append(dict(wavelength_nm=wl,injection_geometry=geom,n_devices=len(v),condition_median=float(np.median(v)),CI95_lower=lo,CI95_upper=hi,bootstrap_N=BOOTSTRAP_N,bootstrap_n_ok=n,random_seed=seed))
 return pd.DataFrame(rows)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['raw-output','output']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 radial=a.raw_output/'figures/s42_radial_acf_fwhm';pv=pd.read_csv(radial/'per_video_radial_acf_fwhm.csv');pv=pv.rename(columns={'fiber_id':'device_id','excitation_geometry':'injection_geometry','radial_acf_fwhm_px':'radial_ACF_FWHM_px'});pv.to_csv(a.output/'Fig5d_per_video_radial_ACF_FWHM.csv',index=False)
 d=pd.read_csv(radial/'fig5d_device_radial_acf_fwhm.csv');ci(d,'radial_acf_fwhm_px',np.random.default_rng(42),42).to_csv(a.output/'Fig5d_condition_bootstrap_CI.csv',index=False)
 pf=pd.read_csv(a.raw_output/'metrics/per_fiber_condition_metrics.csv');rng=np.random.default_rng(43)
 ci(pf,'acf_fwhm_px_median',rng,43) # preserve original shared RNG consumption before PSD
 e=ci(pf,'psd_centroid_cyc_per_px_median',rng,43);e['rng_note']='Generator(43) after four original ACF-condition bootstrap calls';e.to_csv(a.output/'Fig5e_condition_bootstrap_CI.csv',index=False)
 pf['kind']='fiber_condition';pf.to_csv(a.output/'plotting_data_psd.csv',index=False)
 print(e.to_string(index=False))
if __name__=='__main__':main()
