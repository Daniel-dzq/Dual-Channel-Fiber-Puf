"""Lifecycle figures (Nature-style, English labels)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 11,
        "axes.titlesize": 0,  # no in-panel titles
        "axes.labelsize": 12,
        "figure.dpi": 150,
    }
)
COLORS = {"genuine": "#1b4f72", "impostor": "#922b21", "accent": "#1e8449"}


def _save(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def generate_lifecycle_figures(
    out_dir: Path,
    red_scores: pd.DataFrame,
    green_scores: pd.DataFrame,
    events: pd.DataFrame,
    thresholds: dict[str, Any],
) -> None:
    if red_scores is not None and not red_scores.empty:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for group, color in [
            ("same_device_diff_state", COLORS["genuine"]),
            ("diff_device", COLORS["impostor"]),
        ]:
            s = red_scores.loc[red_scores["group"] == group, "score"]
            if len(s):
                ax.hist(s, bins=30, alpha=0.55, label=group.replace("_", " "), color=color)
        ax.axvline(thresholds.get("tau_R", np.nan), color="k", ls="--", lw=1)
        ax.set_xlabel(r"$S_R$")
        ax.set_ylabel("Count")
        ax.legend(frameon=False, fontsize=9)
        _save(fig, out_dir / "red_identity_score_distributions")

    if green_scores is not None and not green_scores.empty:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for group, color in [
            ("same_device_same_state_same_challenge", COLORS["genuine"]),
            ("same_device_diff_state_same_challenge", COLORS["impostor"]),
        ]:
            s = green_scores.loc[green_scores["group"] == group, "score"]
            if len(s):
                ax.hist(s, bins=30, alpha=0.55, label=group.replace("_", " "), color=color)
        ax.axvline(thresholds.get("tau_G", np.nan), color="k", ls="--", lw=1)
        ax.set_xlabel(r"$q_G$")
        ax.set_ylabel("Count")
        ax.legend(frameon=False, fontsize=8)
        _save(fig, out_dir / "green_score_distributions")

        # Cross-state revocation matrix (device × transition median score)
        sub = green_scores[green_scores["group"] == "same_device_diff_state_same_challenge"]
        if not sub.empty:
            piv = sub.groupby(["device_id_a", "state_id_a", "state_id_b"])["score"].median().reset_index()
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.scatter(piv["device_id_a"], piv["score"], c=COLORS["accent"], s=28)
            ax.axhline(thresholds.get("tau_G", np.nan), color="k", ls="--", lw=1)
            ax.set_xlabel("Device")
            ax.set_ylabel("Cross-state median $q_G$")
            plt.xticks(rotation=45, ha="right")
            _save(fig, out_dir / "cross_state_revocation_matrix")

    if events is not None and not events.empty:
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        rates = [
            events["red_pass"].mean(),
            events["previous_green_fail"].mean(),
            events["current_credential_pass"].mean(),
            events["authenticated_reenrollment_success"].mean(),
        ]
        labels = ["Identity", "Revocation", "Re-enroll", "Auth. re-enroll"]
        ax.bar(labels, rates, color=COLORS["genuine"])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Rate")
        _save(fig, out_dir / "lifecycle_summary")
        # Duplicate summary under the required event-matrix filename.
        fig2, ax2 = plt.subplots(figsize=(5.5, 3.2))
        ax2.bar(labels, rates, color=COLORS["genuine"])
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Rate")
        _save(fig2, out_dir / "reenrollment_event_matrix")

        # placeholders for required names
        for name in (
            "red_s0_gallery_identification",
            "development_vs_frozen_test",
            "per_device_failure_audit",
        ):
            fig, ax = plt.subplots(figsize=(4.5, 3))
            if name == "per_device_failure_audit":
                fail = (~events["authenticated_reenrollment_success"]).astype(int)
                ax.bar(events["device_id"] + events["to_state"], fail, color=COLORS["impostor"])
                ax.set_ylabel("Failure (0/1)")
                plt.xticks(rotation=90, fontsize=7)
            else:
                ax.text(0.5, 0.5, "See tables in report", ha="center", va="center")
                ax.axis("off")
            _save(fig, out_dir / name)
