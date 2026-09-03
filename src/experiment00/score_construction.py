"""Score construction: S_intra, S_inter_challenge, S_inter_device, S_block.

Primary genuine pairs join on (length_cm, fiber_id, challenge_id), never on
acquisition_position or input-list index.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from experiment00.acquisition import ORDER_SOURCE, resolve_acquisition_meta
from experiment00.config import Experiment00Config
from experiment00.templates import GreenTemplates
from puf_common.ncc import zero_mean_ncc


def _ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None) -> float:
    return float(zero_mean_ncc(a, b, mask=mask))


def _acq_cols(cfg: Experiment00Config | None, fiber_id: str, challenge_id: str) -> dict:
    if cfg is None:
        meta = resolve_acquisition_meta(
            fiber_id=fiber_id, challenge_id=challenge_id, illumination="green"
        )
    else:
        meta = resolve_acquisition_meta(
            fiber_id=fiber_id,
            challenge_id=challenge_id,
            illumination="green",
            assignment=cfg.acquisition.sequence_assignment,
            sequences=cfg.acquisition.challenge_sequences,
        )
    return {
        "challenge_id": challenge_id,
        "sequence_id": meta.sequence_id,
        "acquisition_position": meta.acquisition_position,
        "acquisition_position_zero_based": meta.acquisition_position_zero_based,
        "order_source": meta.order_source or ORDER_SOURCE,
    }


def block_repeatability_scores(
    templates: dict[tuple[int, str, str, str], GreenTemplates],
    mask: np.ndarray | None,
    cfg: Experiment00Config | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    for (L, f, rnd, ch), t in templates.items():
        blocks = t.blocks_detail_cm
        if len(blocks) < 2:
            continue
        acq = _acq_cols(cfg, f, ch)
        for i, j in combinations(range(len(blocks)), 2):
            rows.append(
                {
                    "length_cm": L,
                    "fiber_id": f,
                    "round": rnd,
                    "challenge": ch,
                    **acq,
                    "block_i": i + 1,
                    "block_j": j + 1,
                    "score_type": "S_block",
                    "score": _ncc(blocks[i], blocks[j], mask),
                }
            )
    return pd.DataFrame(rows)


def construct_pair_scores(
    templates: dict[tuple[int, str, str, str], GreenTemplates],
    mask: np.ndarray | None,
    lengths: list[int],
    fibers: list[str],
    challenges: list[str],
    cfg: Experiment00Config | None = None,
) -> pd.DataFrame:
    """Build primary pair scores via explicit challenge_id key joins.

    Enrollment=A, Query=B. Genuine pairs are A_Cxx ↔ B_Cxx for the same fiber.
    """
    rows: list[dict] = []

    def get(L, f, rnd, ch):
        return templates.get((L, f, rnd, ch))

    for L in tqdm(lengths, desc="Pair scores", unit="len"):
        for f in fibers:
            for ch in challenges:
                a = get(L, f, "A", ch)
                b = get(L, f, "B", ch)
                if a is None or b is None:
                    continue
                acq = _acq_cols(cfg, f, ch)
                rows.append(
                    {
                        "length_cm": L,
                        "score_type": "S_intra",
                        "fiber_id": f,
                        "challenge": ch,
                        "fiber_id_b": f,
                        "challenge_b": ch,
                        "challenge_id_b": ch,
                        **acq,
                        "score": _ncc(a.video_detail_cm, b.video_detail_cm, mask),
                    }
                )
        for f in fibers:
            for c1, c2 in combinations(challenges, 2):
                a1, b1 = get(L, f, "A", c1), get(L, f, "B", c1)
                a2, b2 = get(L, f, "A", c2), get(L, f, "B", c2)
                if None in (a1, b1, a2, b2):
                    continue
                s = 0.5 * (
                    _ncc(a1.video_detail_cm, b2.video_detail_cm, mask)
                    + _ncc(a2.video_detail_cm, b1.video_detail_cm, mask)
                )
                acq = _acq_cols(cfg, f, c1)
                rows.append(
                    {
                        "length_cm": L,
                        "score_type": "S_inter_challenge",
                        "fiber_id": f,
                        "challenge": c1,
                        "fiber_id_b": f,
                        "challenge_b": c2,
                        "challenge_id_b": c2,
                        **acq,
                        "score": float(s),
                    }
                )
        for f1, f2 in combinations(fibers, 2):
            for ch in challenges:
                a1, b1 = get(L, f1, "A", ch), get(L, f1, "B", ch)
                a2, b2 = get(L, f2, "A", ch), get(L, f2, "B", ch)
                if None in (a1, b1, a2, b2):
                    continue
                s = 0.5 * (
                    _ncc(a1.video_detail_cm, b2.video_detail_cm, mask)
                    + _ncc(a2.video_detail_cm, b1.video_detail_cm, mask)
                )
                acq = _acq_cols(cfg, f1, ch)
                rows.append(
                    {
                        "length_cm": L,
                        "score_type": "S_inter_device",
                        "fiber_id": f1,
                        "challenge": ch,
                        "fiber_id_b": f2,
                        "challenge_b": ch,
                        "challenge_id_b": ch,
                        **acq,
                        "score": float(s),
                    }
                )
    return pd.DataFrame(rows)


def challenge_identity_ab_matrix(
    templates: dict[tuple[int, str, str, str], GreenTemplates],
    *,
    length_cm: int,
    fiber_id: str,
    mask: np.ndarray | None,
    challenges: list[str],
) -> pd.DataFrame:
    """8x8 Round-A vs Round-B NCC matrix indexed by challenge_id (not position)."""
    mat = np.full((len(challenges), len(challenges)), np.nan, dtype=float)
    for i, ca in enumerate(challenges):
        for j, cb in enumerate(challenges):
            a = templates.get((length_cm, fiber_id, "A", ca))
            b = templates.get((length_cm, fiber_id, "B", cb))
            if a is None or b is None:
                continue
            mat[i, j] = _ncc(a.video_detail_cm, b.video_detail_cm, mask)
    return pd.DataFrame(mat, index=challenges, columns=challenges)


def expected_pair_counts(n_fibers: int = 5, n_challenges: int = 8) -> dict[str, int]:
    from math import comb

    return {
        "S_intra": n_fibers * n_challenges,
        "S_inter_challenge": n_fibers * comb(n_challenges, 2),
        "S_inter_device": comb(n_fibers, 2) * n_challenges,
    }
