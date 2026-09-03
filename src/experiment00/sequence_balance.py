"""Sequence balance checks: one P1-P5 fiber per length (predefined design)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiment00.cache import dump_json
from experiment00.config import Experiment00Config


def run_sequence_balance_check(
    inventory: pd.DataFrame,
    cfg: Experiment00Config,
    out_dir: Path | None = None,
) -> dict:
    """Verify each length has F01..F05 mapped to P1..P5 exactly once."""
    assign = cfg.acquisition.sequence_assignment
    expected_pairs = [(f, assign[f]) for f in cfg.dataset.expected_fibers]
    rows = []
    ok = True
    if inventory.empty:
        green = inventory
    else:
        mask = inventory["illumination"] == "green"
        if "parse_status" in inventory.columns:
            mask = mask & (inventory["parse_status"] == "ok")
        green = inventory.loc[mask]

    for L in cfg.dataset.expected_lengths_cm:
        sub = green.loc[green["length_cm"] == L] if not green.empty else green
        fibers_present = sorted(sub["fiber_id"].dropna().unique().tolist()) if not sub.empty else []
        seq_by_fiber = {}
        if not sub.empty and "sequence_id" in sub.columns:
            for f, g in sub.groupby("fiber_id"):
                sids = sorted(g["sequence_id"].dropna().unique().tolist())
                seq_by_fiber[f] = sids[0] if len(sids) == 1 else sids
        for fiber, exp_seq in expected_pairs:
            got = seq_by_fiber.get(fiber)
            pass_f = got == exp_seq
            if not pass_f:
                ok = False
            rows.append(
                {
                    "length_cm": L,
                    "fiber_id": fiber,
                    "expected_sequence_id": exp_seq,
                    "observed_sequence_id": got,
                    "status": "PASS" if pass_f else "FAIL",
                }
            )
        # Exactly one of each P1-P5
        observed_seqs = [
            seq_by_fiber[f] for f in cfg.dataset.expected_fibers if f in seq_by_fiber
        ]
        uniq = [s for s in observed_seqs if isinstance(s, str)]
        bal = sorted(uniq) == sorted(assign.values()) and len(uniq) == 5
        if not bal:
            ok = False
        rows.append(
            {
                "length_cm": L,
                "fiber_id": "ALL",
                "expected_sequence_id": "P1-P5 once each",
                "observed_sequence_id": ",".join(uniq),
                "status": "PASS" if bal else "FAIL",
            }
        )

    # Position distribution note (partial counterbalance, not claimed Latin square)
    pos_rows = []
    seqs = cfg.acquisition.challenge_sequences
    for sid, seq in seqs.items():
        for pos, ch in enumerate(seq, start=1):
            pos_rows.append(
                {
                    "sequence_id": sid,
                    "acquisition_position": pos,
                    "challenge_id": ch,
                }
            )
    design = pd.DataFrame(pos_rows)
    # Count how often each challenge appears at each position across P1-P5
    counts = (
        design.groupby(["challenge_id", "acquisition_position"]).size().reset_index(name="n_sequences")
    )

    report = {
        "ok": ok,
        "design_description": (
            "Predefined partially counterbalanced order design: "
            "F01->P1 ... F05->P5, identical across all lengths. "
            "Not claimed as a complete Latin square."
        ),
        "expected_per_length": {f: s for f, s in expected_pairs},
        "n_fail": sum(1 for r in rows if r["status"] == "FAIL"),
        "challenge_position_counts": counts.to_dict(orient="records"),
    }
    df = pd.DataFrame(rows)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "sequence_balance_check.csv", index=False)
        dump_json(out_dir / "sequence_balance_check.json", report)
    return {"ok": ok, "table": df, "report": report}
