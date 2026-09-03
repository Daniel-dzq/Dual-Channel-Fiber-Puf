#!/usr/bin/env python3
"""Supplementary Note 7.1 - threshold development on the eight-challenge remount dataset (T_G, n_req).

Lightweight mode recomputes the corrected operating point from the green pair-score table
shipped in ``data/threshold_development/green_pair_scores_corrected.csv``:

* T_G is the equal-error-rate threshold (501-point sweep) of the development devices
  F01-F05 for same-device/same-state/same-challenge (Round A vs Round B) against
  same-device/same-state/different-challenge scores;
* n_req (k of 8) is the session rule selected on the development devices;
* session acceptance for k = 8, 7, 6 and the joint (red identity AND green credential)
  remount events (30 in total, 20 on the evaluation devices F06-F15).

The 16 S1 recordings listed in the public ``data_quality_exclusions.csv``
(``ACCIDENTAL_S1_COPY_OF_9CM``) are excluded from the independent S1 analysis; their raw
files remain in the dataset. The Exp. 3 ``threshold_development/tau_G.json`` value in the
analysis-ready package (0.1653) is a different pipeline's development EER threshold and is
not the Supplementary Note 7.1 T_G.

Raw mode (decode ``raw_threshold_development.zip``):
``python -m experiment4_security.lifecycle.cli --config configs/lifecycle_threshold_development.yaml --data-root <zenodo>``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiment4_security.lifecycle.thresholds import select_session_rule, select_tau
from puf_common.checks import check, open_dataset, report, reproduction_parser
from puf_common.metrics import auc_roc, equal_error_rate_with_threshold

REPO = Path(__file__).resolve().parents[1]
DEVELOPMENT = [f"F{i:02d}" for i in range(1, 6)]
EVALUATION = [f"F{i:02d}" for i in range(6, 16)]
GENUINE, IMPOSTOR = "same_device_same_state_same_challenge", "same_device_same_state_diff_challenge"

EXPECTED = {"T_G": 0.12899641700197656, "n_req": 7, "sessions": {8: "9/15", 7: "12/15", 6: "12/15"}, "joint": "26/30", "evaluation": "19/20", "n_excluded_S1": 16, "exp3_tau_G_not_S7_1": 0.1653272940550798}


def development_operating_point(green: pd.DataFrame) -> dict:
    dev = green[green.device_id_a.isin(DEVELOPMENT) & green.device_id_b.isin(DEVELOPMENT)]
    genuine = dev.loc[dev.group == GENUINE, "score"].to_numpy(float)
    impostor = dev.loc[dev.group == IMPOSTOR, "score"].to_numpy(float)
    tau_g = select_tau(genuine, impostor)
    rule = select_session_rule(dev, tau_g)
    eer, _ = equal_error_rate_with_threshold(genuine, impostor)
    return {"T_G": tau_g, "n_req": int(rule["selected_k_of_8"]), "candidates": rule["candidates"], "development_auc": auc_roc(genuine, impostor), "development_eer": eer, "n_genuine": len(genuine), "n_impostor": len(impostor)}


def main() -> int:
    args = reproduction_parser(__doc__.splitlines()[0]).parse_args()
    ds, out = open_dataset(args, "threshold_development")

    green = pd.read_csv(REPO / "data/threshold_development/green_pair_scores_corrected.csv")
    events = pd.read_csv(REPO / "data/threshold_development/lifecycle_events_corrected.csv")
    exclusions = ds.read_csv("data_quality_exclusions.csv")
    frozen = ds.read_json("analysis_ready_data/fiber_id_9d/lifecycle/threshold_development.json")
    exp3_tau = ds.read_json("analysis_ready_data/threshold_development/tau_G.json")["tau_G"]

    op = development_operating_point(green)
    sessions = {c["k_of_8"]: f"{round(c['session_pass_rate'] * c['n_events'])}/{c['n_events']}" for c in op["candidates"]}
    joint = f"{int(events.authenticated_reenrollment_success.sum())}/{len(events)}"
    eval_events = events[events.device_id.isin(EVALUATION)]
    evaluation = f"{int(eval_events.authenticated_reenrollment_success.sum())}/{len(eval_events)}"
    n_excluded = int((exclusions.anomaly == "ACCIDENTAL_S1_COPY_OF_9CM").sum())

    pd.Series({**op, "sessions": sessions, "joint_events": joint, "evaluation_events": evaluation, "n_excluded_S1_records": n_excluded}).to_json(out / "threshold_development_summary.json", indent=2)
    events.to_csv(out / "lifecycle_events.csv", index=False)

    checks = [
        check("T_G (development EER threshold)", op["T_G"], EXPECTED["T_G"], 0),
        check("T_G equals frozen tau_G_frozen in the public dataset", op["T_G"], float(frozen["tau_G_frozen"]), 0),
        check("n_req (k of 8)", op["n_req"], EXPECTED["n_req"], 0),
        check("n_req equals frozen n_req_frozen in the public dataset", op["n_req"], int(frozen["n_req_frozen"]), 0),
        *[check(f"development session acceptance k={k}", sessions[k], v) for k, v in EXPECTED["sessions"].items()],
        check("joint current-state remount events", joint, EXPECTED["joint"]),
        check("evaluation-device joint events", evaluation, EXPECTED["evaluation"]),
        check("joint rate equals frozen authenticated_reenrollment_success", round(float(events.authenticated_reenrollment_success.mean()), 12), round(float(frozen["lifecycle_metrics"]["authenticated_reenrollment_success"]), 12), 0),
        check("excluded invalid S1 records in data_quality_exclusions.csv", n_excluded, EXPECTED["n_excluded_S1"], 0),
        check("Exp. 3 tau_G.json is a different quantity (not T_G)", float(exp3_tau), EXPECTED["exp3_tau_G_not_S7_1"], 0),
    ]
    return 0 if report(checks, out, "Threshold-development (Supplementary Note 7.1) authority checks") else 1


if __name__ == "__main__":
    raise SystemExit(main())
