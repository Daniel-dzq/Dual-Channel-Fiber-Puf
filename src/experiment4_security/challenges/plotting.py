"""Challenge library contact sheet."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def render_contact_sheet(
    patterns_dir: Path,
    png_path: Path,
    pdf_path: Path,
    *,
    n: int = 128,
    cols: int = 16,
) -> None:
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.7, rows * 0.55))
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i >= n:
            continue
        p = patterns_dir / f"C{i + 1:03d}.png"
        if not p.exists():
            continue
        img = np.asarray(Image.open(p).convert("L"))
        # Show active crop for readability
        ax.imshow(img[128:640, 256:768], cmap="gray", vmin=0, vmax=255)
    fig.tight_layout(pad=0.1)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
