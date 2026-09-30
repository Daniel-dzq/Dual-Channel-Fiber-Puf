#!/usr/bin/env python3
"""Reproduce formal Fig. 7/SI nine-feature red identity directly from public videos."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor
import argparse,csv,json,hashlib
import numpy as np,pandas as pd,cv2
from experiment4_security.identity_credential.red_video_processing import process_red_video
from experiment4_security.identity_credential.red_feature_adapter import extract_red_vector,fit_standardizer,apply_standardizer,red_score
from puf_common.metrics import auc_roc,equal_error_rate,robust_gap

def pair_table(z: dict[tuple[str, str, str], np.ndarray]) -> pd.DataFrame:
    keys = list(z.keys())
    rows: list[dict[str, Any]] = []
    for i, ka in enumerate(keys):
        for kb in keys[i:]:
            da, sa, pa = ka
            db, sb, pb = kb
            score = red_score(z[ka], z[kb])
            same_device = da == db
            same_state = sa == sb
            before_after = (
                same_device
                and same_state
                and ka != kb
                and {pa, pb} == {"before", "after"}
            )
            r1 = same_device and (not same_state)
            rows.append(
                {
                    "device_a": da,
                    "device_b": db,
                    "state_a": sa,
                    "phase_a": pa,
                    "state_b": sb,
                    "phase_b": pb,
                    "same_device": same_device,
                    "same_state": same_state,
                    "r0": before_after,
                    "r1": r1,
                    "S_R": score,
                    "distance": -score,
                }
            )
    return pd.DataFrame(rows)

def identity_metrics(pairs: pd.DataFrame, devices: tuple[str, ...]) -> dict[str, Any]:
    gen = pairs.loc[pairs["r1"], "S_R"].to_numpy(dtype=np.float64)
    imp = pairs.loc[~pairs["same_device"], "S_R"].to_numpy(dtype=np.float64)
    r0 = pairs.loc[pairs["r0"], "S_R"].to_numpy(dtype=np.float64)
    auc = float(auc_roc(gen, imp)) if gen.size and imp.size else float("nan")
    eer = float(equal_error_rate(gen, imp)) if gen.size and imp.size else float("nan")
    gap = float(robust_gap(gen, imp)) if gen.size and imp.size else float("nan")
    per = []
    for d in devices:
        g = pairs.loc[pairs["r1"] & (pairs["device_a"] == d), "S_R"].to_numpy(dtype=np.float64)
        n = pairs.loc[
            (~pairs["same_device"]) & ((pairs["device_a"] == d) | (pairs["device_b"] == d)),
            "S_R",
        ].to_numpy(dtype=np.float64)
        per.append(
            {
                "device": d,
                "AUC": float(auc_roc(g, n)) if g.size and n.size else float("nan"),
                "EER": float(equal_error_rate(g, n)) if g.size and n.size else float("nan"),
                "median_genuine_cross_state": float(np.median(g)) if g.size else float("nan"),
                "median_impostor": float(np.median(n)) if n.size else float("nan"),
                "robust_gap": float(robust_gap(g, n)) if g.size and n.size else float("nan"),
                "n_genuine": int(g.size),
                "n_impostor": int(n.size),
            }
        )
    worst = max(per, key=lambda r: (r["EER"], -r["AUC"])) if per else {}
    worst_auc = min(per, key=lambda r: r["AUC"]) if per else {}
    return {
        "n_genuine_cross_state": int(gen.size),
        "n_impostor": int(imp.size),
        "n_same_state": int(r0.size),
        "cross_state_auc": auc,
        "cross_state_eer": eer,
        "cross_state_robust_gap": gap,
        "median_same_device_cross_state_score": float(np.median(gen)) if gen.size else float("nan"),
        "median_same_device_cross_state_distance": float(np.median(-gen)) if gen.size else float("nan"),
        "median_different_device_score": float(np.median(imp)) if imp.size else float("nan"),
        "median_different_device_distance": float(np.median(-imp)) if imp.size else float("nan"),
        "median_same_state_score": float(np.median(r0)) if r0.size else float("nan"),
        "same_state_auc": float(auc_roc(r0, imp)) if r0.size and imp.size else float("nan"),
        "same_state_eer": float(equal_error_rate(r0, imp)) if r0.size and imp.size else float("nan"),
        "q05_genuine_cross_state": float(np.percentile(gen, 5)) if gen.size else float("nan"),
        "q95_impostor": float(np.percentile(imp, 95)) if imp.size else float("nan"),
        "per_device": per,
        "worst_device_by_eer": worst,
        "worst_device_by_auc": worst_auc,
        "worst_device_eer": worst.get("EER", float("nan")),
        "worst_device_auc": worst_auc.get("AUC", float("nan")),
    }

def closed_set_retrieval(
    z: dict[tuple[str, str, str], np.ndarray],
    devices: tuple[str, ...],
) -> dict[str, Any]:
    rows = []
    for (dq, sq, pq), qv in z.items():
        if dq not in devices:
            continue
        scores = []
        for cand in devices:
            parts = [v for (d, s, _p), v in z.items() if d == cand and s != sq]
            if not parts:
                continue
            tmpl = np.mean(np.stack(parts, axis=0), axis=0)
            scores.append((cand, red_score(qv, tmpl)))
        scores.sort(key=lambda x: x[1], reverse=True)
        rank = next(i for i, (c, _) in enumerate(scores, start=1) if c == dq)
        rows.append(
            {
                "query_device": dq,
                "query_state": sq,
                "query_phase": pq,
                "predicted": scores[0][0],
                "rank": rank,
                "top1": int(rank == 1),
                "top2": int(rank <= 2),
            }
        )
    df = pd.DataFrame(rows)
    return {
        "n_queries": int(len(df)),
        "top1": float(df["top1"].mean()) if len(df) else float("nan"),
        "top2": float(df["top2"].mean()) if len(df) else float("nan"),
        "rank_table": df,
    }

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--workers',type=int,default=1);a=p.parse_args();root=a.data_root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
 rows=[r for r in csv.DictReader((root/'MANIFEST.csv').open()) if r['dataset']=='formal_mechanical_reconfiguration' and r['optical_channel']=='red'];rows.sort(key=lambda r:r['release_path']);assert len(rows)==160
 mask=np.load(root/'masks/threshold_development_and_formal_valid_pixel_mask.npy').astype(bool)
 def one(r):
  source=(root/r['release_path']).resolve();assert source.is_relative_to(root)
  if hashlib.sha256(source.read_bytes()).hexdigest()!=r['sha256']:raise ValueError(source)
  key=(r['device'],r['mechanical_state'],r['acquisition_role'].rsplit('_',1)[-1]);rec=process_red_video(source,device_id=key[0],state_id=key[1],phase=key[2]);return key,extract_red_vector(rec.representative,mask)
 raw={}
 with ThreadPoolExecutor(max_workers=a.workers) as pool:
  for i,(key,vec) in enumerate(pool.map(one,rows),1):
   raw[key]=vec
   if i%8==0:print('Decoded red',i,'/160',flush=True)
 dev=tuple(f'F{i:02}' for i in range(1,6));held=tuple(f'F{i:02}' for i in range(6,11));devices=dev+held;mu,sd=fit_standardizer([v for k,v in raw.items() if k[0] in dev]);z={k:apply_standardizer(v,mu,sd) for k,v in raw.items()};pairs=pair_table(z);pairs.to_csv(out/'pairs_9d.csv',index=False)
 np.savez(out/'raw_red_features_and_standardizer.npz',features=np.stack(list(raw.values())),device_state_phase=np.array(list(raw)),mean=mu,standard_deviation=sd)
 summaries={}
 for name,subset in [('pooled',devices),('development',dev),('heldout',held)]:
  sub=pairs[pairs.device_a.isin(subset)&pairs.device_b.isin(subset)];summaries[name]=identity_metrics(sub,subset);ret=closed_set_retrieval({k:v for k,v in z.items() if k[0] in subset},subset);ret.pop('rank_table').to_csv(out/(name+'_retrieval.csv'),index=False);summaries[name]['retrieval']=ret
 (out/'summary.json').write_text(json.dumps(summaries,indent=2));(out/'provenance.json').write_text(json.dumps({'recordings':[{'release_path':r['release_path'],'sha256':r['sha256']} for r in rows],'standardizer_fit_devices':dev,'standardizer_fit_samples':80,'mask_sha256':hashlib.sha256((root/'masks/threshold_development_and_formal_valid_pixel_mask.npy').read_bytes()).hexdigest()},indent=2));print('Completed formal red identity',flush=True)
if __name__=='__main__':main()
