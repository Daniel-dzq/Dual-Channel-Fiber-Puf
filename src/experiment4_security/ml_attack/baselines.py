"""Baselines for enrollment-database disclosure (not physical cloning).

C0_mean_response: ignore the challenge; always emit the mean enrollment template.
C1_exact_template_replay: replay the stored Round-A enrollment template T_s,c.
  This is a digital database-replay benchmark, not a physical clone of the fiber.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MeanBaselinePredictor:
    mean_vector: np.ndarray
    model_name: str = "C0_mean_response"
    n_parameters: int = 0

    def predict(self, challenge_binary_vector: np.ndarray) -> np.ndarray:
        return self.mean_vector

    def predict_by_challenge_id(self, challenge_id: str) -> np.ndarray:
        return self.mean_vector


def fit_mean_baseline(train_eval_vectors: dict[str, np.ndarray]) -> MeanBaselinePredictor:
    stack = np.stack(list(train_eval_vectors.values()), axis=0).astype(np.float64)
    mean_vec = stack.mean(axis=0).astype(np.float32)
    return MeanBaselinePredictor(mean_vector=mean_vec, n_parameters=int(mean_vec.size))


@dataclass
class ExactTemplateReplayPredictor:
    """C1: exact stolen enrollment-template replay (database dump attack)."""

    templates_by_challenge: dict[str, np.ndarray]
    model_name: str = "C1_exact_template_replay"

    @property
    def n_parameters(self) -> int:
        if not self.templates_by_challenge:
            return 0
        return int(sum(v.size for v in self.templates_by_challenge.values()))

    def predict_by_challenge_id(self, challenge_id: str) -> np.ndarray:
        return self.templates_by_challenge[challenge_id]

    def predict(self, challenge_binary_vector: np.ndarray) -> np.ndarray:
        raise RuntimeError(
            "ExactTemplateReplayPredictor requires challenge_id lookup via "
            "predict_by_challenge_id; binary-vector predict is not defined."
        )


def fit_exact_template_replay(enrollment_templates: dict[str, np.ndarray]) -> ExactTemplateReplayPredictor:
    return ExactTemplateReplayPredictor(
        templates_by_challenge={cid: v.astype(np.float32) for cid, v in enrollment_templates.items()}
    )


# Backwards-compat aliases used by leftover imports during migration.
NearestChallengeBaselinePredictor = ExactTemplateReplayPredictor  # type: ignore


def fit_nearest_challenge_baseline(
    train_challenge_binary: dict[str, np.ndarray], train_eval_vectors: dict[str, np.ndarray]
) -> ExactTemplateReplayPredictor:
    """Deprecated unseen-challenge baseline; maps to exact replay for registered bank."""
    return fit_exact_template_replay(train_eval_vectors)
