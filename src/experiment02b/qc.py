"""Per-video quality control status (PASS / WARN / FAIL)."""

from __future__ import annotations

from typing import Any

from experiment02b.config import Experiment02bConfig


def evaluate_qc(row: dict[str, Any], cfg: Experiment02bConfig) -> dict[str, Any]:
    reasons: list[str] = []
    warn: list[str] = []
    q = cfg.qc
    a = cfg.analysis

    retained = int(row.get("retained_frame_count") or 0)
    if retained < int(q.minimum_retained_frames):
        warn.append(f"retained_frames<{q.minimum_retained_frames}")

    sat = float(row.get("saturation_fraction") or 0.0)
    if sat >= float(q.saturation_warning_fraction):
        warn.append("high_saturation_fraction")

    sbr = float(row.get("signal_to_background_ratio") or float("nan"))
    if sbr == sbr and sbr < float(q.minimum_signal_to_background_ratio):
        warn.append("low_signal_to_background_ratio")

    cov = float(row.get("roi_coverage") or float("nan"))
    if cov == cov and cov < float(q.minimum_roi_coverage):
        warn.append("low_roi_coverage")

    if row.get("crop_ok") is False:
        reasons.append(str(row.get("crop_fail_reason") or "crop_out_of_bounds"))

    if row.get("decode_ok") is False:
        reasons.append("decode_failed")

    leak_status = str(row.get("channel_leakage_status") or "")
    if leak_status == "FAIL":
        reasons.append("channel_leakage_fail")
    elif leak_status == "WARN":
        warn.append("channel_leakage_warn")

    dark = float(row.get("dark_fraction") or 0.0)
    if dark >= float(q.dark_fraction_warning):
        warn.append("high_dark_fraction")

    # centroid near border of full frame
    cx = row.get("centroid_x")
    cy = row.get("centroid_y")
    w = row.get("width")
    h = row.get("height")
    margin = float(q.max_centroid_border_margin_px)
    if cx is not None and cy is not None and w and h:
        if (
            float(cx) < margin
            or float(cy) < margin
            or float(cx) > float(w) - margin
            or float(cy) > float(h) - margin
        ):
            warn.append("centroid_near_border")

    if reasons:
        status = "FAIL"
    elif warn:
        status = "WARN"
    else:
        status = "PASS"

    return {
        "qc_status": status,
        "qc_reasons": reasons + warn,
        "qc_fail_reasons": reasons,
        "qc_warn_reasons": warn,
    }
