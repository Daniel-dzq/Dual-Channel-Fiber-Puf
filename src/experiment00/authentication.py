"""Authentication tasks: device / challenge Top-1."""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from experiment00.templates import GreenTemplates
from puf_common.metrics import auc_roc, d_prime, equal_error_rate_with_threshold
from puf_common.ncc import zero_mean_ncc


def _ncc(a, b, mask):
    return float(zero_mean_ncc(a, b, mask=mask))


def length_separation_metrics(pair_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for L, sub in tqdm(list(pair_scores.groupby("length_cm")), desc="Separation metrics", unit="len"):
        g = sub.loc[sub.score_type == "S_intra", "score"].to_numpy(float)
        ich = sub.loc[sub.score_type == "S_inter_challenge", "score"].to_numpy(float)
        idev = sub.loc[sub.score_type == "S_inter_device", "score"].to_numpy(float)

        def pack(name, impostor):
            if g.size == 0 or impostor.size == 0:
                return {
                    "median_intra": float(np.nanmedian(g)) if g.size else np.nan,
                    f"median_{name}": float(np.nanmedian(impostor)) if impostor.size else np.nan,
                    f"margin_{name}": np.nan,
                    f"robust_gap_{name}": np.nan,
                    f"auc_{name}": np.nan,
                    f"eer_{name}": np.nan,
                    f"far_{name}": np.nan,
                    f"frr_{name}": np.nan,
                    f"dprime_{name}": np.nan,
                }
            eer, thr = equal_error_rate_with_threshold(g, impostor)
            far = float(np.mean(impostor >= thr))
            frr = float(np.mean(g < thr))
            return {
                "median_intra": float(np.median(g)),
                f"median_{name}": float(np.median(impostor)),
                f"margin_{name}": float(np.median(g) - np.median(impostor)),
                f"robust_gap_{name}": float(np.quantile(g, 0.05) - np.quantile(impostor, 0.95)),
                f"auc_{name}": float(auc_roc(g, impostor)),
                f"eer_{name}": float(eer),
                f"far_{name}": far,
                f"frr_{name}": frr,
                f"dprime_{name}": float(d_prime(g, impostor)),
                f"tau_eer_{name}": float(thr),
            }

        row = {"length_cm": int(L)}
        ch = pack("challenge", ich)
        dv = pack("device", idev)
        row.update(ch)
        row.update({k: v for k, v in dv.items() if not k.startswith("median_intra")})
        row["robust_gap_min"] = float(
            np.nanmin([row.get("robust_gap_challenge", np.nan), row.get("robust_gap_device", np.nan)])
        )
        rows.append(row)
    return pd.DataFrame(rows)


def device_top1(
    templates: dict[tuple[int, str, str, str], GreenTemplates],
    mask,
    lengths,
    fibers,
    challenges,
) -> pd.DataFrame:
    rows = []
    for L in tqdm(lengths, desc="Device Top-1", unit="len"):
        for f_true in fibers:
            for ch in challenges:
                q = templates.get((L, f_true, "B", ch))
                if q is None:
                    continue
                scores = []
                for f_gal in fibers:
                    gal = templates.get((L, f_gal, "A", ch))
                    if gal is None:
                        continue
                    scores.append((f_gal, _ncc(q.video_detail_cm, gal.video_detail_cm, mask)))
                if not scores:
                    continue
                pred = max(scores, key=lambda x: x[1])[0]
                rows.append(
                    {
                        "length_cm": L,
                        "task": "device_top1",
                        "true_fiber": f_true,
                        "challenge": ch,
                        "challenge_id": ch,
                        "pred_fiber": pred,
                        "correct": int(pred == f_true),
                        "chance": 1.0 / len(fibers),
                    }
                )
    return pd.DataFrame(rows)


def challenge_top1(
    templates: dict[tuple[int, str, str, str], GreenTemplates],
    mask,
    lengths,
    fibers,
    challenges,
) -> pd.DataFrame:
    rows = []
    for L in tqdm(lengths, desc="Challenge Top-1", unit="len"):
        for f in fibers:
            for ch_true in challenges:
                q = templates.get((L, f, "B", ch_true))
                if q is None:
                    continue
                scores = []
                for ch_gal in challenges:
                    gal = templates.get((L, f, "A", ch_gal))
                    if gal is None:
                        continue
                    scores.append((ch_gal, _ncc(q.video_detail_cm, gal.video_detail_cm, mask)))
                if not scores:
                    continue
                pred = max(scores, key=lambda x: x[1])[0]
                rows.append(
                    {
                        "length_cm": L,
                        "task": "challenge_top1",
                        "fiber_id": f,
                        "true_challenge": ch_true,
                        "true_challenge_id": ch_true,
                        "pred_challenge": pred,
                        "pred_challenge_id": pred,
                        "correct": int(pred == ch_true),
                        "chance": 1.0 / len(challenges),
                    }
                )
    return pd.DataFrame(rows)


def summarize_top1(pred: pd.DataFrame) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame()
    rows = []
    for (L, task), sub in pred.groupby(["length_cm", "task"]):
        rows.append(
            {
                "length_cm": int(L),
                "task": task,
                "n_queries": int(len(sub)),
                "top1": float(sub["correct"].mean()),
                "chance": float(sub["chance"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)
