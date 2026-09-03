"""Order-effect vs challenge-effect QC (no plots; tables/JSON/MD only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from experiment00.cache import dump_json
from experiment00.config import Experiment00Config
from experiment00.templates import GreenTemplates
from puf_common.ncc import zero_mean_ncc


def _ncc(a, b, mask):
    return float(zero_mean_ncc(a, b, mask=mask))


def _spatial_row(spatial: pd.DataFrame | None, L, f, ch) -> pd.Series | None:
    if spatial is None or spatial.empty:
        return None
    ch_col = "challenge_id" if "challenge_id" in spatial.columns else "challenge"
    if ch_col not in spatial.columns:
        return None
    sp = spatial.loc[
        (spatial.length_cm == L)
        & (spatial.fiber_id == f)
        & (spatial["round"] == "A")
        & (spatial[ch_col] == ch)
    ]
    if sp.empty:
        return None
    return sp.iloc[0]


def compute_order_and_challenge_effects(
    *,
    cfg: Experiment00Config,
    templates: dict[tuple[int, str, str, str], GreenTemplates],
    mask: np.ndarray | None,
    spatial: pd.DataFrame | None,
    block_scores: pd.DataFrame | None,
    out_dir: Path,
) -> dict:
    """Aggregate QC by acquisition_position and by challenge_id separately."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lengths = sorted({k[0] for k in templates}) or cfg.dataset.expected_lengths_cm
    fibers = sorted({k[1] for k in templates}) or cfg.dataset.expected_fibers
    challenges = list(cfg.dataset.expected_challenges)

    rows = []
    for L in tqdm(lengths, desc="Order-effect QC", unit="len"):
        for f in fibers:
            for ch in challenges:
                a = templates.get((L, f, "A", ch))
                b = templates.get((L, f, "B", ch))
                if a is None or b is None:
                    continue
                residual = a.video_detail - a.video_detail_cm
                if mask is not None and mask.shape == residual.shape:
                    res_e = float(np.mean(residual[mask.astype(bool)] ** 2))
                else:
                    res_e = float(np.mean(residual**2))
                row = {
                    "length_cm": L,
                    "fiber_id": f,
                    "challenge_id": ch,
                    "sequence_id": a.sequence_id,
                    "acquisition_position": a.acquisition_position,
                    "order_source": a.order_source,
                    "ncc_raw": _ncc(a.video_raw, b.video_raw, mask),
                    "ncc_detail": _ncc(a.video_detail, b.video_detail, mask),
                    "ncc_detail_cm": _ncc(a.video_detail_cm, b.video_detail_cm, mask),
                    "mean_intensity_A": float(np.mean(a.video_raw)),
                    "common_mode_residual_energy": res_e,
                }
                sp = _spatial_row(spatial, L, f, ch)
                if sp is not None:
                    row["snr"] = float(sp["snr"]) if "snr" in sp.index else np.nan
                    row["saturation_fraction"] = (
                        float(sp["saturation_fraction"]) if "saturation_fraction" in sp.index else np.nan
                    )
                if block_scores is not None and not block_scores.empty:
                    bs = block_scores.loc[
                        (block_scores.length_cm == L)
                        & (block_scores.fiber_id == f)
                        & (block_scores["challenge"] == ch)
                    ]
                    row["block_repeatability_median"] = (
                        float(bs["score"].median()) if not bs.empty else np.nan
                    )
                rows.append(row)

    df = pd.DataFrame(rows)

    def _agg(frame: pd.DataFrame, by: str) -> pd.DataFrame:
        if frame.empty or by not in frame.columns:
            return pd.DataFrame()
        metrics = [
            c
            for c in [
                "ncc_raw",
                "ncc_detail",
                "ncc_detail_cm",
                "mean_intensity_A",
                "snr",
                "saturation_fraction",
                "block_repeatability_median",
                "common_mode_residual_energy",
            ]
            if c in frame.columns
        ]
        return frame.groupby(by, as_index=False)[metrics].median(numeric_only=True)

    order_qc = _agg(df, "acquisition_position")
    chal_qc = _agg(df, "challenge_id")
    order_qc.to_csv(out_dir / "order_effect_qc.csv", index=False)
    chal_qc.to_csv(out_dir / "challenge_effect_qc.csv", index=False)

    model = {
        "note": (
            "acquisition_position is chronological within a round under the fiber "
            "sequence mapping; it is not a challenge identity label. "
            "challenge_id aggregates isolate challenge effects separately."
        ),
        "order_source": "fiber_sequence_mapping",
        "n_pairs": int(len(df)),
        "order_effect_medians": order_qc.to_dict(orient="records") if not order_qc.empty else [],
        "challenge_effect_medians": chal_qc.to_dict(orient="records") if not chal_qc.empty else [],
    }
    if not df.empty and df["acquisition_position"].notna().any():
        x = df["acquisition_position"].to_numpy(float)
        y = df["ncc_detail_cm"].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() > 3:
            xr = pd.Series(x[m]).rank().to_numpy()
            yr = pd.Series(y[m]).rank().to_numpy()
            model["spearman_detail_cm_vs_position"] = float(np.corrcoef(xr, yr)[0, 1])

    dump_json(out_dir / "order_effect_model.json", model)
    lines = [
        "# Order-effect and challenge-effect QC\n",
        f"- pairs: {model['n_pairs']}",
        f"- order_source: {model['order_source']}",
        f"- spearman(detail_cm, acquisition_position): {model.get('spearman_detail_cm_vs_position')}",
        "",
        "acquisition_position aggregates early-vs-late recording behaviour.",
        "challenge_id aggregates isolate challenge identity effects.",
        "Do not conflate the two.",
    ]
    (out_dir / "order_effect_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return model
