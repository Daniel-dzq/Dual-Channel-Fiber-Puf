"""Green repeatability preflight and challenge-label permutation diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from experiment00.cache import dump_json
from experiment00.config import Experiment00Config
from experiment00.score_construction import challenge_identity_ab_matrix
from experiment00.templates import GreenTemplates
from puf_common.ncc import zero_mean_ncc


def run_green_repeatability_preflight(
    *,
    cfg: Experiment00Config,
    templates: dict[tuple[int, str, str, str], GreenTemplates],
    mask: np.ndarray | None,
    out_dir: Path,
    length_cm: int | None = None,
    fiber_id: str | None = None,
) -> dict:
    """Pilot/preflight: challenge-identity A/B matrix and permutation diagnostic.

    Planned P1-P5 acquisition sequences do NOT constitute CHALLENGE_LABEL_MISMATCH.
    Matrices are indexed by challenge_id; expected optimal permutation is identity
    when labels are correct.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    L = length_cm if length_cm is not None else cfg.dataset.expected_lengths_cm[0]
    f = fiber_id if fiber_id is not None else cfg.dataset.expected_fibers[0]
    challenges = list(cfg.dataset.expected_challenges)

    mat = challenge_identity_ab_matrix(
        templates, length_cm=L, fiber_id=f, mask=mask, challenges=challenges
    )
    mat.to_csv(out_dir / "challenge_identity_ab_matrix.csv")

    diag = np.diag(mat.to_numpy(float))
    diag_median = float(np.nanmedian(diag)) if np.isfinite(diag).any() else float("nan")

    # Greedy permutation maximizing diagonal sum (diagnostic only)
    M = mat.to_numpy(float).copy()
    used_cols = set()
    perm = []
    for i in range(len(challenges)):
        row = M[i].copy()
        for j in used_cols:
            row[j] = -np.inf
        jbest = int(np.nanargmax(row)) if np.isfinite(row).any() else i
        used_cols.add(jbest)
        perm.append(jbest)
    is_identity = perm == list(range(len(challenges)))

    report = {
        "length_cm": L,
        "fiber_id": f,
        "matrix_index": "challenge_id",
        "diagonal_median_ncc_detail_cm": diag_median,
        "per_challenge_diagonal": {
            challenges[i]: (float(diag[i]) if np.isfinite(diag[i]) else None)
            for i in range(len(challenges))
        },
        "optimal_permutation_identity": is_identity,
        "optimal_permutation": [challenges[j] for j in perm],
        "CHALLENGE_LABEL_MISMATCH": False,
        "mismatch_note": (
            "A non-identity optimal permutation cannot be explained merely by "
            "F02-F05 using different acquisition orders (P2-P5). Planned sequence "
            "assignment must not trigger CHALLENGE_LABEL_MISMATCH. Only after "
            "challenge_id metadata are correctly parsed would a non-identity "
            "permutation suggest a display/recording/filename-label error."
        ),
        "sequence_assignment_is_not_label_mismatch": True,
    }
    if not is_identity:
        # Still do not auto-flag mismatch solely due to sequence design
        report["CHALLENGE_LABEL_MISMATCH"] = False
        report["permutation_warning"] = (
            "Non-identity greedy permutation observed; investigate only after "
            "confirming challenge_id joins — not attributable to P1-P5 design alone."
        )

    dump_json(out_dir / "green_repeatability_preflight.json", report)
    lines = [
        "# Green repeatability preflight\n",
        f"- length_cm: {L}",
        f"- fiber_id: {f}",
        f"- diagonal_median detail_cm NCC: {diag_median}",
        f"- optimal_permutation_identity: {is_identity}",
        f"- CHALLENGE_LABEL_MISMATCH: {report['CHALLENGE_LABEL_MISMATCH']}",
        f"- note: {report['mismatch_note']}",
    ]
    (out_dir / "green_repeatability_preflight.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return report
