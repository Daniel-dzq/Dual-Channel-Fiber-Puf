"""Leakage and negative-control audits.

Every check here either PASSES silently (recorded) or raises/records a hard
failure -- nothing is allowed to "fail soft" and continue producing Track
A/B/C results as if nothing happened. `run.py` is expected to abort (or at
minimum prominently flag) Track A/B/C if any hard-leakage check fails, per
Section 24 (stage 8: Track B / Track C run only after Track A passes the leakage audit).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from experiment4_security.ml_attack.representations import RepresentationBundle
from puf_common.ncc import zero_mean_ncc


@dataclass
class LeakageAuditResult:
    checks: list[dict[str, Any]] = field(default_factory=list)
    overall_status: str = "PASS"

    def add(self, name: str, passed: bool, detail: dict[str, Any]) -> None:
        self.checks.append({"check": name, "passed": bool(passed), **detail})
        if not passed:
            self.overall_status = "FAIL"

    def to_dict(self) -> dict[str, Any]:
        return {"overall_status": self.overall_status, "checks": self.checks}


def audit_representation_fit(
    result: LeakageAuditResult,
    *,
    context: str,
    bundle: RepresentationBundle,
    validation_ids: set[str],
    test_ids: set[str],
    held_out_state_ids: set[str],
) -> None:
    fit_ids = set(bundle.pca_fit_sample_ids) | set(bundle.normalization_fit_sample_ids) | set(bundle.common_fit_sample_ids)
    overlap_val = fit_ids & validation_ids
    overlap_test = fit_ids & test_ids
    overlap_held_out = fit_ids & held_out_state_ids
    result.add(
        f"{context}:pca_normalization_common_template_no_validation_leakage",
        passed=(len(overlap_val) == 0),
        detail={"overlap_sample_ids": sorted(overlap_val)[:10], "n_overlap": len(overlap_val)},
    )
    result.add(
        f"{context}:pca_normalization_common_template_no_test_leakage",
        passed=(len(overlap_test) == 0),
        detail={"overlap_sample_ids": sorted(overlap_test)[:10], "n_overlap": len(overlap_test)},
    )
    result.add(
        f"{context}:pca_normalization_common_template_no_held_out_state_leakage",
        passed=(len(overlap_held_out) == 0),
        detail={"overlap_sample_ids": sorted(overlap_held_out)[:10], "n_overlap": len(overlap_held_out)},
    )


def audit_video_frame_disjoint(
    result: LeakageAuditResult, *, context: str, train_video_paths: set[str], test_video_paths: set[str]
) -> None:
    overlap = train_video_paths & test_video_paths
    result.add(
        f"{context}:no_video_shared_between_train_and_test",
        passed=(len(overlap) == 0),
        detail={"overlap_paths": sorted(overlap)[:5], "n_overlap": len(overlap)},
    )


def audit_challenge_split_consistency(result: LeakageAuditResult, manifest: pd.DataFrame) -> None:
    """Confirm challenge_id -> split is single-valued across every
    (state, round) it appears in."""
    if "split" not in manifest.columns:
        result.add("challenge_split:single_valued_per_challenge_id", passed=False, detail={"error": "no split column present"})
        return
    grouped = manifest.groupby("challenge_id")["split"].nunique()
    violations = grouped[grouped > 1]
    result.add(
        "challenge_split:single_valued_per_challenge_id",
        passed=bool((grouped <= 1).all()),
        detail={"violating_challenge_ids": violations.index.tolist()[:10], "n_violations": int((grouped > 1).sum())},
    )


def audit_state_not_in_fit(
    result: LeakageAuditResult, *, context: str, fit_state_ids: set[str], held_out_state_id: str
) -> None:
    result.add(
        f"{context}:held_out_state_excluded_from_fit_states",
        passed=(held_out_state_id not in fit_state_ids),
        detail={"held_out_state_id": held_out_state_id, "fit_state_ids": sorted(fit_state_ids)},
    )


def run_permutation_negative_control(
    *,
    train_features: dict[str, np.ndarray],
    train_eval_vectors: dict[str, np.ndarray],
    test_features: dict[str, np.ndarray],
    test_eval_vectors: dict[str, np.ndarray],
    fit_ridge_fn,
    mean_baseline_test_ncc: float,
    seed: int,
    suspicion_margin: float = 0.05,
) -> dict[str, Any]:
    """Shuffle the challenge_id <-> response correspondence within TRAIN,
    refit, and evaluate on the (unshuffled) TEST set. A real, non-leaking
    pipeline should see permuted-label test performance collapse to
    approximately the mean-response baseline, since the shuffled labels
    destroy any genuine challenge -> response mapping. If permuted
    performance remains well above the mean baseline, that indicates
    leakage (e.g. a representation/common-template contaminated with test
    information, or a model that is secretly keying off of something other
    than the challenge content).
    """
    rng = np.random.default_rng(seed)
    train_ids = sorted(train_features.keys() & train_eval_vectors.keys())
    shuffled_targets = train_ids.copy()
    rng.shuffle(shuffled_targets)
    permuted_eval_vectors = {cid: train_eval_vectors[shuffled_targets[i]] for i, cid in enumerate(train_ids)}

    fit_out = fit_ridge_fn(train_features, permuted_eval_vectors)
    predictor = fit_out["predictor"]

    test_ids = sorted(test_features.keys() & test_eval_vectors.keys())
    scores = [float(zero_mean_ncc(predictor.predict(test_features[cid]), test_eval_vectors[cid])) for cid in test_ids]
    permuted_test_ncc = float(np.mean(scores)) if scores else float("nan")

    suspicious = bool(np.isfinite(permuted_test_ncc) and (permuted_test_ncc - mean_baseline_test_ncc) > suspicion_margin)
    return {
        "permuted_test_ncc_mean": permuted_test_ncc,
        "mean_baseline_test_ncc": mean_baseline_test_ncc,
        "margin_used": suspicion_margin,
        "run_status": "LEAKAGE_SUSPECTED" if suspicious else "OK",
    }
