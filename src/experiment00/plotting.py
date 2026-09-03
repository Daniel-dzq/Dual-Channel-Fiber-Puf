"""Nature-inspired plotting for Experiment 00 (English-only; no rainbow)."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

from experiment00.config import Experiment00Config
from experiment00.length_selection import SelectionDecision
from experiment00.templates import GreenTemplates
from puf_common.acf_features import normalized_acf_2d, radial_acf_profile
from puf_common.psd_features import compute_psd_2d, radial_average_psd

C = {
    "intra": "#3C5488",
    "inter_challenge": "#00A087",
    "inter_device": "#E64B35",
    "red": "#DC0000",
    "aux": "#4DBBD5",
    "violet": "#8491B4",
    "neutral": "#4A4A4A",
}
LENGTH_COLORS = {
    7: "#DEEBF7",
    9: "#9ECAE1",
    11: "#4292C6",
    13: "#2171B5",
    15: "#084594",
}


def apply_style(cfg: Experiment00Config) -> None:
    mpl.rcParams.update(
        {
            "font.family": cfg.plotting.font_family,
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "figure.dpi": 120,
            "savefig.dpi": cfg.plotting.dpi,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def panel_label(ax, lab: str, x: float = -0.12, y: float = 1.05) -> None:
    if lab:
        ax.text(x, y, lab, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom")


def save_fig(fig, stem: Path, cfg: Experiment00Config) -> list[str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, flag in (
        ("png", cfg.plotting.save_png),
        ("pdf", cfg.plotting.save_pdf),
        ("svg", cfg.plotting.save_svg),
    ):
        if flag:
            p = stem.with_suffix(f".{ext}")
            fig.savefig(p, bbox_inches="tight", facecolor="white")
            paths.append(str(p))
    plt.close(fig)
    return paths


def select_representative_template(
    templates: dict[tuple[int, str, str, str], GreenTemplates],
    spatial: pd.DataFrame,
    length_cm: int,
) -> GreenTemplates | None:
    """Deterministic median-nearest rule (never cherry-pick best NCC)."""
    cands = [
        t
        for (L, f, rnd, ch), t in templates.items()
        if L == length_cm and rnd == "A" and ch == "C01"
    ]
    if not cands and templates:
        cands = [t for (L, *_), t in templates.items() if L == length_cm]
    if not cands:
        return None
    if spatial is None or spatial.empty or "acf_width_px" not in spatial.columns:
        return sorted(cands, key=lambda t: (t.fiber_id, t.challenge))[0]
    sub = spatial.loc[(spatial.length_cm == length_cm) & (spatial["round"] == "A")]
    if sub.empty:
        return sorted(cands, key=lambda t: (t.fiber_id, t.challenge))[0]
    med = float(sub["acf_width_px"].median())
    # Map fiber to acf width at C01 if possible
    best = None
    best_d = 1e18
    for t in cands:
        row = sub.loc[(sub.fiber_id == t.fiber_id) & (sub.challenge == t.challenge)]
        if row.empty:
            row = sub.loc[sub.fiber_id == t.fiber_id]
        if row.empty:
            continue
        d = abs(float(row["acf_width_px"].iloc[0]) - med)
        if d < best_d:
            best_d = d
            best = t
    return best or sorted(cands, key=lambda t: (t.fiber_id, t.challenge))[0]


def _imshow_fixed(ax, img, vmin, vmax, cmap="gray"):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])


def generate_panel_b_and_s10(
    cfg: Experiment00Config,
    fig_dir: Path,
    templates: dict,
    spatial: pd.DataFrame,
    mask: np.ndarray | None,
    lengths: list[int],
) -> dict[str, list[str]]:
    """Build Main Figure panel-b strip assets and supplementary S10."""
    apply_style(cfg)
    paths: dict[str, list[str]] = {"panel_b": [], "S10": []}
    mbool = mask.astype(bool) if mask is not None else None

    # Shared display limits from all selected representatives (fair comparison)
    reps = []
    for L in lengths:
        t = select_representative_template(templates, spatial, L)
        if t is not None:
            reps.append((L, t))
    if not reps:
        return paths

    raw_stack = np.stack([t.video_raw for _, t in reps], axis=0)
    vmin_raw = float(np.nanpercentile(raw_stack, 2))
    vmax_raw = float(np.nanpercentile(raw_stack, 98))

    # Panel b composite: rows = lengths, cols = raw | ACF | radial PSD
    n = len(reps)
    fig, axes = plt.subplots(n, 3, figsize=(9.5, 2.2 * n))
    if n == 1:
        axes = np.array([axes])
    for i, (L, t) in enumerate(tqdm(reps, desc="Figure panel b", unit="len")):
        _imshow_fixed(axes[i, 0], t.video_raw, vmin_raw, vmax_raw)
        axes[i, 0].set_ylabel(f"{L} cm", fontsize=11)
        if i == 0 and cfg.plotting.add_panel_labels:
            panel_label(axes[i, 0], "b", x=-0.08, y=1.05)
        if mbool is not None and mbool.shape == t.video_raw.shape:
            acf = normalized_acf_2d(t.video_raw, mbool, max_lag=24)
            _imshow_fixed(axes[i, 1], acf, -0.2, 1.0)
            psd, fx, fy = compute_psd_2d(t.video_raw, mbool)
            centers, vals = radial_average_psd(psd, fx, fy)
            axes[i, 2].plot(centers, vals, color=LENGTH_COLORS.get(L, C["neutral"]), lw=1.5)
            axes[i, 2].set_yticks([])
        else:
            axes[i, 1].axis("off")
            axes[i, 2].axis("off")
        if i == n - 1:
            axes[i, 0].set_xlabel("Raw green")
            axes[i, 1].set_xlabel("2D ACF")
            axes[i, 2].set_xlabel("Radial PSD")
    fig.tight_layout()
    paths["panel_b"] = save_fig(fig, fig_dir / "main_figure_00_panel_b", cfg)

    # S10: all devices ACF/PSD curves for each length
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.5), sharex=False)
    if cfg.plotting.add_panel_labels:
        panel_label(axes[0], "a")
        panel_label(axes[1], "b")
    for L, t in tqdm(reps, desc="Figure S10", unit="len"):
        if mbool is None or mbool.shape != t.video_raw.shape:
            continue
        acf = normalized_acf_2d(t.video_raw, mbool, max_lag=32)
        rad = radial_acf_profile(acf)
        axes[0].plot(np.arange(len(rad)), rad, color=LENGTH_COLORS.get(L, C["neutral"]), label=f"{L} cm", lw=1.4)
        psd, fx, fy = compute_psd_2d(t.video_raw, mbool)
        centers, vals = radial_average_psd(psd, fx, fy)
        axes[1].plot(centers, vals, color=LENGTH_COLORS.get(L, C["neutral"]), label=f"{L} cm", lw=1.4)
    axes[0].set_ylabel("Radial ACF")
    axes[1].set_ylabel("Radial PSD")
    axes[1].set_xlabel("Spatial frequency")
    axes[0].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    axes[1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    paths["S10"] = save_fig(fig, fig_dir / "supp_S10_acf_psd_by_length", cfg)
    return paths


def generate_all_figures(
    cfg: Experiment00Config,
    run_dir: Path,
    pairs: pd.DataFrame,
    length_metrics: pd.DataFrame,
    boot_sel: pd.DataFrame,
    spatial: pd.DataFrame,
    decision: SelectionDecision,
    templates: dict | None = None,
    mask: np.ndarray | None = None,
) -> dict[str, list[str]]:
    apply_style(cfg)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    all_paths: dict[str, list[str]] = {}

    lengths = cfg.dataset.expected_lengths_cm
    if templates:
        extra = generate_panel_b_and_s10(cfg, fig_dir, templates, spatial, mask, lengths)
        all_paths.update(extra)

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.0))

    ax = axes[0, 0]
    ax.axis("off")
    if cfg.plotting.add_panel_labels:
        panel_label(ax, "a", x=0.0, y=1.0)
    ax.text(
        0.02,
        0.95,
        "7-15 cm lengths\n-> 5 fibers / length\n-> Enrollment A / Query B\n"
        "-> red before/after\n-> physical stats + PUF scores\n-> operating-window selection",
        va="top",
        ha="left",
        fontsize=11,
        color=C["neutral"],
        family=cfg.plotting.font_family,
    )

    # Panel b: embed note pointing to dedicated panel_b file if generated
    ax = axes[0, 1]
    ax.axis("off")
    if cfg.plotting.add_panel_labels:
        panel_label(ax, "b", x=0.0, y=1.0)
    if all_paths.get("panel_b"):
        ax.text(0.02, 0.55, "See main_figure_00_panel_b\n(raw / ACF / PSD by length)", fontsize=11, color=C["neutral"])
    else:
        ax.text(0.02, 0.55, "Panel b pending templates", fontsize=11, color=C["neutral"])

    ax = axes[0, 2]
    if cfg.plotting.add_panel_labels:
        panel_label(ax, "c")
    if not pairs.empty:
        plot_df = pairs.copy()
        sns.boxplot(
            data=plot_df,
            x="length_cm",
            y="score",
            hue="score_type",
            palette={
                "S_intra": C["intra"],
                "S_inter_challenge": C["inter_challenge"],
                "S_inter_device": C["inter_device"],
            },
            ax=ax,
            fliersize=1.5,
            linewidth=0.8,
        )
        ax.set_xlabel("Length (cm)")
        ax.set_ylabel("NCC (detail_cm)")
        ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))

    ax = axes[1, 0]
    if cfg.plotting.add_panel_labels:
        panel_label(ax, "d")
    if not spatial.empty and "acf_width_px" in spatial.columns:
        for metr, color in [
            ("acf_width_px", C["aux"]),
            ("n_eff", C["violet"]),
            ("speckle_contrast", C["inter_challenge"]),
        ]:
            if metr not in spatial.columns:
                continue
            g = spatial.groupby("length_cm")[metr].median()
            ax.plot(g.index, g.values, marker="o", color=color, lw=1.5, label=metr)
        ax.set_xlabel("Length (cm)")
        ax.set_ylabel("Spatial metric (median)")
        ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    if cfg.plotting.add_panel_labels:
        panel_label(ax, "e")
    if not length_metrics.empty and "n_eff_median" in length_metrics.columns:
        x = length_metrics["n_eff_median"].to_numpy(float)
        y = length_metrics["robust_gap_min"].to_numpy(float)
        Ls = length_metrics["length_cm"].to_numpy(int)
        for xi, yi, L in zip(x, y, Ls):
            ax.scatter([xi], [yi], s=80, color=LENGTH_COLORS.get(int(L), C["neutral"]), zorder=3)
            ax.text(xi, yi, f" {L} cm", fontsize=9, va="center")
        ax.set_xlabel(r"$N_{\mathrm{eff}}$ (approx.)")
        ax.set_ylabel(r"min robust gap")

    ax = axes[1, 2]
    if cfg.plotting.add_panel_labels:
        panel_label(ax, "f")
    if not boot_sel.empty:
        ax.bar(
            boot_sel["length_cm"].astype(str),
            boot_sel["selection_probability"],
            color=[LENGTH_COLORS.get(int(L), C["neutral"]) for L in boot_sel["length_cm"]],
            edgecolor=C["neutral"],
            linewidth=0.6,
        )
        ax.set_xlabel("Length (cm)")
        ax.set_ylabel("Bootstrap selection probability")
        ax.set_ylim(0, 1.05)
        if decision.selected_length_cm is not None:
            ax.text(
                0.98,
                0.98,
                f"Selected: {decision.selected_length_cm} cm\n({decision.decision_type})",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                color=C["neutral"],
            )

    fig.tight_layout()
    all_paths["main_figure_00"] = save_fig(fig, fig_dir / "main_figure_00", cfg)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    if not length_metrics.empty:
        ax.plot(
            length_metrics["length_cm"],
            length_metrics["robust_gap_min"],
            marker="o",
            color=C["intra"],
            lw=1.8,
        )
        ax.set_xlabel("Length (cm)")
        ax.set_ylabel("robust_gap_min")
    all_paths["S12_context"] = save_fig(fig, fig_dir / "supp_S12_bootstrap_selection_context", cfg)
    return all_paths
