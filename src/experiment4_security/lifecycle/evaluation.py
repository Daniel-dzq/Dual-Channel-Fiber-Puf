"""Score-group metrics for green/red lifecycle analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from puf_common.metrics import (
    auc_roc,
    d_prime,
    equal_error_rate,
    equal_error_rate_with_threshold,
    margin,
    robust_gap,
)


def summarize_scores(genuine: np.ndarray, impostor: np.ndarray) -> dict[str, Any]:
    g = np.asarray(genuine, dtype=float)
    i = np.asarray(impostor, dtype=float)
    g = g[np.isfinite(g)]
    i = i[np.isfinite(i)]
    eer, thr = equal_error_rate_with_threshold(g, i) if g.size and i.size else (float("nan"), float("nan"))
    if g.size and i.size:
        frr = float(np.mean(g < thr))
        far = float(np.mean(i >= thr))
    else:
        frr = far = float("nan")
    return {
        "count_genuine": int(g.size),
        "count_impostor": int(i.size),
        "q5_genuine": float(np.nanpercentile(g, 5)) if g.size else float("nan"),
        "median_genuine": float(np.nanmedian(g)) if g.size else float("nan"),
        "q95_genuine": float(np.nanpercentile(g, 95)) if g.size else float("nan"),
        "q5_impostor": float(np.nanpercentile(i, 5)) if i.size else float("nan"),
        "median_impostor": float(np.nanmedian(i)) if i.size else float("nan"),
        "q95_impostor": float(np.nanpercentile(i, 95)) if i.size else float("nan"),
        "median_margin": float(margin(g, i)) if g.size and i.size else float("nan"),
        "robust_gap": float(robust_gap(g, i)) if g.size and i.size else float("nan"),
        "auc": float(auc_roc(g, i)) if g.size and i.size else float("nan"),
        "eer": float(eer),
        "far": far,
        "frr": frr,
        "d_prime": float(d_prime(g, i)) if g.size and i.size else float("nan"),
        "threshold_eer": float(thr),
    }


def group_pair_metrics(scores: pd.DataFrame, genuine_group: str, impostor_group: str) -> dict[str, Any]:
    g = scores.loc[scores["group"] == genuine_group, "score"].to_numpy()
    i = scores.loc[scores["group"] == impostor_group, "score"].to_numpy()
    out = summarize_scores(g, i)
    out["genuine_group"] = genuine_group
    out["impostor_group"] = impostor_group
    return out
