"""Fixed-state metrics (Fig. 6): S_G, S_C, S_D, q_R, RG_*, AUC_*, EER_*, retrieval."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiment2.figure_layout import REPRESENTATION
from puf_common.metrics import (
    auc_roc,
    equal_error_rate,
    equal_error_rate_with_threshold,
    robust_gap,
)

SCORE_TYPE_GENUINE = "GENUINE"
SCORE_TYPE_CHALLENGE = "CHALLENGE_MISMATCH"
SCORE_TYPE_DEVICE = "DEVICE_MISMATCH"

LEGACY_SCORE_MAP = {
    "intra_repeatability": SCORE_TYPE_GENUINE,
    "inter_challenge": SCORE_TYPE_CHALLENGE,
    "inter_device": SCORE_TYPE_DEVICE,
}


@dataclass
class GreenScoreBundle:
    genuine: pd.DataFrame
    challenge_mismatch: pd.DataFrame
    device_mismatch: pd.DataFrame
    source: str
    representation: str = REPRESENTATION


@dataclass
class ClassificationMetrics:
    auc_challenge: float
    eer_challenge: float
    eer_challenge_threshold: float
    auc_device: float
    eer_device: float
    eer_device_threshold: float
    rg_challenge: float
    rg_device: float
    rg_min: float
    q05_genuine: float
    q95_challenge: float
    q95_device: float
    median_genuine: float
    median_challenge: float
    median_device: float


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_hash(path: Path | None) -> str | None:
    if path is None or not Path(path).is_file():
        return None
    return _sha256_file(Path(path))


def load_legacy_green_scores(metrics_dir: Path) -> GreenScoreBundle:
    """Load the score CSVs written by ``experiment2.cli run`` and map them to the manuscript names."""
    metrics_dir = Path(metrics_dir)
    genuine = pd.read_csv(metrics_dir / "intra_repeatability_scores.csv")
    challenge = pd.read_csv(metrics_dir / "inter_challenge_scores.csv")
    device = pd.read_csv(metrics_dir / "inter_device_scores.csv")

    def _remap(df: pd.DataFrame, expected_legacy: str) -> pd.DataFrame:
        out = df.copy()
        if not (out["score_type"] == expected_legacy).all():
            bad = sorted(out["score_type"].unique())
            raise ValueError(f"Unexpected score_type in legacy CSV: {bad}")
        out["score_type"] = LEGACY_SCORE_MAP[expected_legacy]
        out["score"] = out["ncc"].astype(float)
        out["representation"] = REPRESENTATION
        out["score_direction"] = "HIGHER_IS_MORE_GENUINE"
        if expected_legacy == "intra_repeatability":
            out["score_genuine"] = out["score"]
        elif expected_legacy == "inter_challenge":
            out["score_challenge_mismatch"] = out["score"]
        else:
            out["score_device_mismatch"] = out["score"]
        return out

    return GreenScoreBundle(
        genuine=_remap(genuine, "intra_repeatability"),
        challenge_mismatch=_remap(challenge, "inter_challenge"),
        device_mismatch=_remap(device, "inter_device"),
        source=str(metrics_dir),
        representation=REPRESENTATION,
    )


def scores_array(df: pd.DataFrame, column: str = "score") -> np.ndarray:
    return np.asarray(df[column], dtype=np.float64)


def compute_green_classification(bundle: GreenScoreBundle) -> ClassificationMetrics:
    g = scores_array(bundle.genuine)
    c = scores_array(bundle.challenge_mismatch)
    d = scores_array(bundle.device_mismatch)
    rg_c = robust_gap(g, c)
    rg_d = robust_gap(g, d)
    eer_c, thr_c = equal_error_rate_with_threshold(g, c)
    eer_d, thr_d = equal_error_rate_with_threshold(g, d)
    return ClassificationMetrics(
        auc_challenge=auc_roc(g, c),
        eer_challenge=eer_c,
        eer_challenge_threshold=thr_c,
        auc_device=auc_roc(g, d),
        eer_device=eer_d,
        eer_device_threshold=thr_d,
        rg_challenge=rg_c,
        rg_device=rg_d,
        rg_min=float(min(rg_c, rg_d)),
        q05_genuine=float(np.percentile(g, 5)),
        q95_challenge=float(np.percentile(c, 95)),
        q95_device=float(np.percentile(d, 95)),
        median_genuine=float(np.median(g)),
        median_challenge=float(np.median(c)),
        median_device=float(np.median(d)),
    )


def per_device_tail_gaps(bundle: GreenScoreBundle) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    devices = sorted(bundle.genuine["device_id_a"].unique())
    for device in devices:
        g = scores_array(bundle.genuine[bundle.genuine["device_id_a"] == device])
        c = scores_array(
            bundle.challenge_mismatch[bundle.challenge_mismatch["device_id_a"] == device]
        )
        # Device mismatch: pairs involving this device
        ddf = bundle.device_mismatch
        d = scores_array(
            ddf[(ddf["device_id_a"] == device) | (ddf["device_id_b"] == device)]
        )
        if g.size == 0 or c.size == 0 or d.size == 0:
            continue
        q05_g = float(np.percentile(g, 5))
        q95_c = float(np.percentile(c, 95))
        q95_d = float(np.percentile(d, 95))
        rg_c = q05_g - q95_c
        rg_d = q05_g - q95_d
        rg_min = float(min(rg_c, rg_d))
        rows.append(
            {
                "device_id": device,
                "n_genuine": int(g.size),
                "n_challenge_mismatch": int(c.size),
                "n_device_mismatch": int(d.size),
                "q05_genuine": q05_g,
                "q95_challenge": q95_c,
                "q95_device": q95_d,
                "rg_challenge": rg_c,
                "rg_device": rg_d,
                "rg_min": rg_min,
                "worst_negative_q95": float(max(q95_c, q95_d)),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("rg_min", ascending=True).reset_index(drop=True)


def ecdf_table(values: np.ndarray, score_type: str) -> pd.DataFrame:
    v = np.sort(np.asarray(values, dtype=np.float64))
    n = v.size
    if n == 0:
        return pd.DataFrame(columns=["score", "ecdf", "score_type"])
    return pd.DataFrame(
        {
            "score": v,
            "ecdf": (np.arange(1, n + 1, dtype=np.float64) / n),
            "score_type": score_type,
        }
    )


def build_score_ecdf(bundle: GreenScoreBundle) -> pd.DataFrame:
    parts = [
        ecdf_table(scores_array(bundle.genuine), SCORE_TYPE_GENUINE),
        ecdf_table(scores_array(bundle.challenge_mismatch), SCORE_TYPE_CHALLENGE),
        ecdf_table(scores_array(bundle.device_mismatch), SCORE_TYPE_DEVICE),
    ]
    return pd.concat(parts, ignore_index=True)


def roc_curve(genuine: np.ndarray, impostor: np.ndarray, n_thresholds: int = 501) -> pd.DataFrame:
    g = np.asarray(genuine, dtype=np.float64)
    i = np.asarray(impostor, dtype=np.float64)
    lo = float(min(g.min(), i.min()))
    hi = float(max(g.max(), i.max()))
    thresholds = np.linspace(lo, hi, n_thresholds)
    rows = []
    for t in thresholds:
        tpr = float(np.mean(g >= t))
        fpr = float(np.mean(i >= t))
        fnr = 1.0 - tpr
        rows.append({"threshold": float(t), "tpr": tpr, "fpr": fpr, "fnr": fnr})
    return pd.DataFrame(rows)


def retrieval_metrics(
    ranks: np.ndarray,
    *,
    candidate_count: int,
    top_k: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    ranks = np.asarray(ranks, dtype=np.float64)
    out: dict[str, float] = {
        "candidate_count": float(candidate_count),
        "n_queries": float(ranks.size),
        "median_rank": float(np.median(ranks)) if ranks.size else float("nan"),
        "mrr": float(np.mean(1.0 / ranks)) if ranks.size else float("nan"),
    }
    for k in top_k:
        out[f"top{k}"] = float(np.mean(ranks <= k)) if ranks.size else float("nan")
    return out


def stratified_bootstrap_ci(
    values_by_key: dict[Any, np.ndarray],
    statistic_fn,
    *,
    n_boot: int = 5000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Bootstrap over primary keys (devices), resampling within-key scores with replacement."""
    rng = np.random.default_rng(seed)
    keys = list(values_by_key.keys())
    if not keys:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    point = float(statistic_fn(np.concatenate([values_by_key[k] for k in keys])))
    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(keys, size=len(keys), replace=True)
        sample_parts = []
        for k in chosen:
            arr = values_by_key[k]
            if arr.size == 0:
                continue
            sample_parts.append(rng.choice(arr, size=arr.size, replace=True))
        if not sample_parts:
            continue
        boots.append(float(statistic_fn(np.concatenate(sample_parts))))
    if not boots:
        return {"point": point, "ci_low": float("nan"), "ci_high": float("nan")}
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return {"point": point, "ci_low": lo, "ci_high": hi, "n_boot": float(n_boot)}


def metrics_to_dict(m: ClassificationMetrics) -> dict[str, float]:
    return {
        "auc_challenge": m.auc_challenge,
        "eer_challenge": m.eer_challenge,
        "auc_device": m.auc_device,
        "eer_device": m.eer_device,
        "rg_challenge": m.rg_challenge,
        "rg_device": m.rg_device,
        "rg_min": m.rg_min,
        "q05_genuine": m.q05_genuine,
        "q95_challenge": m.q95_challenge,
        "q95_device": m.q95_device,
        "median_genuine": m.median_genuine,
        "median_challenge": m.median_challenge,
        "median_device": m.median_device,
    }
