"""Shared figure layout, fonts and colour semantics for the fixed-state analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# --- Colour semantics (shared with Exp00 / future Exp03 lifecycle) ---
COLOR_GENUINE = "#3C5488"
COLOR_CHALLENGE = "#00A087"
COLOR_DEVICE = "#E64B35"
COLOR_RED_SAME = "#B2182B"
COLOR_RED_DIFF = "#7F7F7F"
COLOR_AUX_BLUEGREY = "#8491B4"
COLOR_AUX_CYAN = "#4DBBD5"
COLOR_HIGHLIGHT_BG = "#EEF3F8"
COLOR_BG = "#FFFFFF"
COLOR_TEXT = "#202020"
COLOR_GRID = "#D9D9D9"

NCC_CMAP = LinearSegmentedColormap.from_list(
    "chapter4_ncc_diverging",
    ["#2166AC", "#F7F7F7", "#B2182B"],
    N=256,
)
NCC_VMIN = -0.2
NCC_VCENTER = 0.0
NCC_VMAX = 1.0

MM = 1 / 25.4
FIG3_WIDTH_IN = 178 * MM  # 7.01 in
FIG3_HEIGHT_IN = 6.2

REPRESENTATION = "fullres_detail_cm_fixed_state"


def ncc_norm() -> TwoSlopeNorm:
    return TwoSlopeNorm(vmin=NCC_VMIN, vcenter=NCC_VCENTER, vmax=NCC_VMAX)


def apply_chapter4_style() -> dict[str, Any]:
    """Configure Arial (Liberation Sans fallback) and editable PDF/SVG text."""
    families = [f.name for f in mpl.font_manager.fontManager.ttflist]
    if "Arial" in families:
        font_family = "Arial"
        font_status = "Arial"
    elif "Liberation Sans" in families:
        font_family = "Liberation Sans"
        font_status = "Liberation Sans (fallback)"
    else:
        font_family = "DejaVu Sans"
        font_status = "DejaVu Sans (fallback — Arial unavailable)"

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font_family, "Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.9,
            "axes.edgecolor": COLOR_TEXT,
            "axes.labelcolor": COLOR_TEXT,
            "xtick.color": COLOR_TEXT,
            "ytick.color": COLOR_TEXT,
            "text.color": COLOR_TEXT,
            "figure.facecolor": COLOR_BG,
            "axes.facecolor": COLOR_BG,
            "savefig.facecolor": COLOR_BG,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
        }
    )
    return {"font_family": font_family, "font_status": font_status}


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.9)
        ax.spines[spine].set_color(COLOR_TEXT)
    ax.tick_params(width=0.8, length=3.5, colors=COLOR_TEXT)


def panel_label(ax: plt.Axes, letter: str, *, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight="bold",
        ha="left",
        va="top",
        color=COLOR_TEXT,
        clip_on=False,
    )


def save_figure(fig: plt.Figure, stem: Path, *, dpi_raster: int = 600) -> list[Path]:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ext, kwargs in (
        (".pdf", {}),
        (".svg", {}),
        (".png", {"dpi": 200}),
        (".tiff", {"dpi": dpi_raster, "pil_kwargs": {"compression": "tiff_lzw"}}),
    ):
        out = stem.with_suffix(ext)
        try:
            fig.savefig(out, bbox_inches="tight", pad_inches=0.02, **kwargs)
            paths.append(out)
        except TypeError:
            # Older matplotlib may not accept pil_kwargs
            if ext == ".tiff":
                fig.savefig(out, dpi=dpi_raster, bbox_inches="tight", pad_inches=0.02)
                paths.append(out)
            else:
                raise
    return paths


def save_preview_half(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = fig.get_size_inches()
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    return path
