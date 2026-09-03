"""Channel data-leakage / cross-talk audit for wavelength-matched RGB decode.

Checks whether the wavelength-labeled video actually concentrates energy in the
expected BGR plane, and whether gray/wrong-channel decoding would leak the
other wavelength's encoding into the scientific representation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from experiment02b.video_io import color_label_to_channel, extract_analysis_channel
from puf_common.ncc import zero_mean_ncc


def _plane_stats(bgr: np.ndarray) -> dict[str, float]:
    bgr = np.asarray(bgr, dtype=np.float64)
    if bgr.ndim != 3 or bgr.shape[2] < 3:
        raise ValueError(f"Expected BGR HxWx3, got {bgr.shape}")
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    return {
        "mean_B": float(np.mean(b)),
        "mean_G": float(np.mean(g)),
        "mean_R": float(np.mean(r)),
        "std_B": float(np.std(b)),
        "std_G": float(np.std(g)),
        "std_R": float(np.std(r)),
        "energy_B": float(np.mean(b**2)),
        "energy_G": float(np.mean(g**2)),
        "energy_R": float(np.mean(r**2)),
    }


def audit_bgr_sample(
    bgr: np.ndarray,
    color_label: str,
    *,
    dominance_ratio_warn: float = 1.25,
    dominance_ratio_fail: float = 1.05,
    gray_ncc_warn: float = 0.98,
) -> dict[str, Any]:
    """Audit one BGR frame sample.

    Leakage / QC notions:
    1. Expected-channel dominance: for green videos, mean_G should exceed mean_R/B;
       for red videos, mean_R should exceed mean_G/B.
    2. Wrong-channel residual: energy still present in non-target planes (chroma
       upsampling / white balance / mixed illumination).
    3. Gray leakage proxy: NCC(selected_plane, gray) very high AND wrong-plane
       energy large → gray decode would mix off-wavelength content.
    """
    channel_name, bgr_index, wavelength_nm = color_label_to_channel(color_label)
    stats = _plane_stats(bgr)
    selected = extract_analysis_channel(bgr, color_label).astype(np.float64)
    # OpenCV-style gray from BGR
    gray = (
        0.114 * bgr[..., 0] + 0.587 * bgr[..., 1] + 0.299 * bgr[..., 2]
    ).astype(np.float64)

    if channel_name == "green":
        target_mean = stats["mean_G"]
        other_means = [stats["mean_R"], stats["mean_B"]]
        wrong_energy = stats["energy_R"] + stats["energy_B"]
        target_energy = stats["energy_G"]
        wrong_channel_for_test = extract_analysis_channel(bgr, "red")
    else:
        target_mean = stats["mean_R"]
        other_means = [stats["mean_G"], stats["mean_B"]]
        wrong_energy = stats["energy_G"] + stats["energy_B"]
        target_energy = stats["energy_R"]
        wrong_channel_for_test = extract_analysis_channel(bgr, "green")

    max_other = float(max(other_means)) if other_means else float("nan")
    dominance = float(target_mean / (max_other + 1e-6))
    target_energy_fraction = float(target_energy / (target_energy + wrong_energy + 1e-12))
    ncc_vs_gray = float(zero_mean_ncc(selected, gray))
    ncc_vs_wrong = float(zero_mean_ncc(selected, wrong_channel_for_test.astype(np.float64)))

    reasons: list[str] = []
    status = "PASS"
    if dominance < dominance_ratio_fail:
        status = "FAIL"
        reasons.append(
            f"expected_channel_not_dominant dominance={dominance:.3f}<{dominance_ratio_fail}"
        )
    elif dominance < dominance_ratio_warn:
        status = "WARN"
        reasons.append(
            f"weak_expected_channel_dominance dominance={dominance:.3f}<{dominance_ratio_warn}"
        )

    if target_energy_fraction < 0.45:
        if status != "FAIL":
            status = "WARN"
        reasons.append(
            f"low_target_energy_fraction={target_energy_fraction:.3f} "
            "(possible cross-talk / white-balance leakage)"
        )

    # If gray is almost identical to selected BUT wrong planes carry substantial energy,
    # gray decode would silently mix leakage into the scientific stream.
    off_frac = 1.0 - target_energy_fraction
    if ncc_vs_gray >= gray_ncc_warn and off_frac >= 0.20:
        if status != "FAIL":
            status = "WARN"
        reasons.append(
            "gray_decode_would_mix_off_channel_energy "
            f"(ncc_selected_vs_gray={ncc_vs_gray:.3f}, off_energy_frac={off_frac:.3f})"
        )

    return {
        "color_label": str(color_label).lower(),
        "wavelength_nm": wavelength_nm,
        "analysis_channel": channel_name,
        "analysis_channel_bgr_index": bgr_index,
        **stats,
        "expected_channel_dominance_ratio": dominance,
        "target_energy_fraction": target_energy_fraction,
        "off_channel_energy_fraction": off_frac,
        "ncc_selected_vs_gray": ncc_vs_gray,
        "ncc_selected_vs_wrong_wavelength_channel": ncc_vs_wrong,
        "channel_leakage_status": status,
        "channel_leakage_reasons": reasons,
        "gray_decode_forbidden": True,
    }
