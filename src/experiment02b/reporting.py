"""Neutral analysis reports (no marketing language; no auto success claims)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from experiment02b.config import Experiment02bConfig


def write_reports(cfg: Experiment02bConfig) -> dict[str, Any]:
    out = Path(cfg.paths.output_dir)
    man = out / "manifest" / "inventory_audit.json"
    audit = json.loads(man.read_text()) if man.is_file() else {}
    per_video_p = out / "metrics" / "per_video_metrics.csv"
    per_video = pd.read_csv(per_video_p) if per_video_p.is_file() else pd.DataFrame()
    qc_p = out / "qc" / "video_qc.csv"
    qc = pd.read_csv(qc_p) if qc_p.is_file() else pd.DataFrame()
    contrasts_p = out / "statistics" / "factorial_contrasts.csv"
    contrasts = pd.read_csv(contrasts_p) if contrasts_p.is_file() else pd.DataFrame()
    summary_p = out / "statistics" / "contrast_summary.csv"
    csum = pd.read_csv(summary_p) if summary_p.is_file() else pd.DataFrame()
    stab_p = out / "metrics" / "temporal_stability_metrics.csv"
    stab = pd.read_csv(stab_p) if stab_p.is_file() else pd.DataFrame()

    attr = cfg.attribution
    causal_ok = attr.causal_factorial_interpretation_valid

    summary = {
        "experiment_id": cfg.experiment_id,
        "inventory": audit,
        "n_videos_metrics": int(len(per_video)),
        "qc_counts": {
            "PASS": int((qc["qc_status"] == "PASS").sum()) if not qc.empty and "qc_status" in qc else None,
            "WARN": int((qc["qc_status"] == "WARN").sum()) if not qc.empty and "qc_status" in qc else None,
            "FAIL": int((qc["qc_status"] == "FAIL").sum()) if not qc.empty and "qc_status" in qc else None,
        },
        "attribution": {
            "fixed_input_pattern_confirmed": attr.fixed_input_pattern_confirmed,
            "same_camera_geometry_confirmed": attr.same_camera_geometry_confirmed,
            "same_package_state_confirmed": attr.same_package_state_confirmed,
            "comparable_polarization_confirmed": attr.comparable_polarization_confirmed,
            "comparable_optical_path_documented": attr.comparable_optical_path_documented,
            "causal_factorial_interpretation_valid": causal_ok,
        },
        "primary_endpoints": {
            "acf_r50_px": "radial ACF first crossing of 0.5 (radius, linear interpolation)",
            "acf_fwhm_px": "2 × acf_r50_px",
            "psd_centroid_cyc_per_px": "sum(f·P)/sum(P) with DC excluded; P normalized",
        },
        "decode_policy": "BGR color decode; green→G plane, red→R plane; gray/luma forbidden; channel leakage audited",
        "common_mode": "disabled (no detail_cm)",
        "plots_stage": "disabled — publication figures deferred",
    }
    (out / "reports" / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines: list[str] = []
    lines.append("# Experiment 02b — Analysis Report\n")
    lines.append("Neutral mechanism-control report. No authentication metrics.\n")

    lines.append("## 1. Data completeness\n")
    lines.append(f"- videos_root: `{cfg.paths.videos_root}`")
    lines.append(f"- inventory ok: `{audit.get('ok')}`")
    lines.append(f"- parsed: {audit.get('n_parsed')} / expected 60")
    lines.append(f"- missing keys: {len(audit.get('missing_keys') or [])}")
    lines.append(f"- duplicate keys: {len(audit.get('duplicate_keys') or [])}")
    lines.append(f"- unparsed files: {len(audit.get('unparsed_files') or [])}\n")

    lines.append("## 2. Video encoding and time sampling\n")
    lines.append("- Analysis window: drop first/last 5 s; ≤60 frames uniform in time; ≥24 retained preferred.")
    lines.append("- Frame indices are never used as the scientific clock; timestamps drive sampling.")
    if not per_video.empty and "decode_backend" in per_video.columns:
        lines.append(f"- Decode backends: {per_video['decode_backend'].value_counts().to_dict()}")
        lines.append(f"- Retained frames median: {per_video['retained_frame_count'].median() if 'retained_frame_count' in per_video else 'n/a'}")
    lines.append("")

    lines.append("## 3. Quality control\n")
    if qc.empty:
        lines.append("- QC table not available.\n")
    else:
        lines.append(f"- PASS/WARN/FAIL: {summary['qc_counts']}")
        if "qc_status" in qc.columns:
            fail = qc[qc["qc_status"] == "FAIL"]
            if not fail.empty:
                lines.append("- FAIL videos:")
                for _, r in fail.iterrows():
                    lines.append(f"  - {r.get('filename')}: {r.get('qc_reasons')}")
            warn = qc[qc["qc_status"] == "WARN"]
            if not warn.empty:
                lines.append(f"- WARN count: {len(warn)} (retained by default)")
        lines.append("")

    lines.append("## 4. Temporal stability under fixed package\n")
    lines.append("- Endpoints are descriptive; no pre-set NCC>0.95 claim.")
    if not stab.empty:
        lines.append(
            f"- repeat_to_repeat_ncc_median (median across conditions): "
            f"{stab['repeat_to_repeat_ncc_median'].median():.4f}"
            if "repeat_to_repeat_ncc_median" in stab
            else "- repeat NCC table present"
        )
    else:
        lines.append("- Stability table not available.")
    lines.append("")

    lines.append("## 5. ACF spatial correlation length\n")
    lines.append("- Primary: `acf_fwhm_px = 2 * acf_r50_px` on Hann-windowed 384×384 detail.")
    lines.append("- Smaller width is consistent with finer speckles / shorter correlation length.")
    _append_metric_block(lines, csum, contrasts, "acf_fwhm_px")

    lines.append("## 6. PSD spatial-frequency complexity\n")
    lines.append("- Primary: `psd_centroid_cyc_per_px` (DC excluded, normalized power).")
    lines.append("- ACF and PSD are Fourier duals — not independent proofs.")
    _append_metric_block(lines, csum, contrasts, "psd_centroid_cyc_per_px")

    lines.append("## 7. Wavelength main effect\n")
    lines.append("- Defined as 532 relative to 650, averaged across geometries.")
    _append_effect(lines, csum, "acf_fwhm_px", "Delta_wavelength")
    _append_effect(lines, csum, "psd_centroid_cyc_per_px", "Delta_wavelength")

    lines.append("## 8. Excitation-geometry main effect\n")
    lines.append("- Defined as lateral − axial, averaged across wavelengths.")
    lines.append("- Language: excitation geometry/pathway (not pure incidence angle).\n")
    _append_effect(lines, csum, "acf_fwhm_px", "Delta_geometry")
    _append_effect(lines, csum, "psd_centroid_cyc_per_px", "Delta_geometry")

    lines.append("## 9. Wavelength × geometry interaction\n")
    _append_effect(lines, csum, "acf_fwhm_px", "Delta_interaction")
    _append_effect(lines, csum, "psd_centroid_cyc_per_px", "Delta_interaction")

    lines.append("## 10. Beam-scale normalization sensitivity\n")
    lines.append("- See `normalized_acf_fwhm` and `normalized_psd_centroid` columns in per-video metrics.")
    lines.append("- Pre-specified; not selected post hoc for nicest plots.\n")

    lines.append("## 11. Consistency across fibers\n")
    lines.append("- With n=5 fibers, sign counts and bootstrap CIs are limited-resolution.\n")

    lines.append("## 12. Strongest conclusion currently supported\n")
    if not causal_ok:
        lines.append(
            "- `causal_factorial_interpretation_valid = false` because one or more attribution "
            "flags in config are unset. Descriptive contrasts may still be computed, but "
            "strict mechanism attribution is not claimed."
        )
    lines.append(
        "- Allowed phrasing when contrasts are coherent: "
        "\"Results support that wavelength and excitation geometry differentially affect "
        "effective spatial-frequency content and the effective modal subspace.\""
    )
    lines.append(
        "- Stronger \"geometry > wavelength\" wording is allowed only if the pre-specified "
        "evidence checklist in the experiment brief is met; code does not auto-assert this.\n"
    )

    lines.append("## 13. Conclusions not supported by this experiment\n")
    lines.append("- Does not prove true guided-mode counts or complete modal decomposition.")
    lines.append("- Does not prove \"red only low-order / green only high-order\".")
    lines.append("- Does not measure PUF authentication (AUC/EER/Top-1) or clone resistance.")
    lines.append("- Does not prove long-term stability or mechanical remount sensitivity.")
    lines.append("- ACF width is not a direct mode counter.\n")

    lines.append("## 14. Method and data limitations\n")
    lines.append("- YUV420P chroma subsampling: science uses wavelength-matched RGB plane, not gray.")
    lines.append("- Channel leakage audit flags weak expected-plane dominance / gray-mix risk.")
    lines.append("- Variable frame counts → timestamp sampling mandatory.")
    lines.append("- Background may be edge-median estimated if no dark field.")
    lines.append("- n=5 fibers → avoid p-value theater; report paired effects + bootstrap.")
    lines.append("- Pathway optics may confound \"pure angle\" claims.")
    lines.append("\nPublication figures intentionally deferred until numerical review.\n")

    (out / "reports" / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"report": str(out / "reports" / "analysis_report.md")}


def _append_metric_block(lines: list[str], csum: pd.DataFrame, contrasts: pd.DataFrame, metric: str) -> None:
    if csum.empty:
        lines.append(f"- No contrast summary for `{metric}` yet.\n")
        return
    sub = csum[csum.get("metric") == metric] if "metric" in csum.columns else pd.DataFrame()
    if sub.empty:
        lines.append(f"- No rows for `{metric}`.\n")
        return
    lines.append(f"- Metric `{metric}` contrast summary rows: {len(sub)}")
    lines.append("")


def _append_effect(lines: list[str], csum: pd.DataFrame, metric: str, effect: str) -> None:
    if csum.empty or "metric" not in csum.columns:
        lines.append("- (no data)\n")
        return
    sub = csum[(csum["metric"] == metric) & (csum["effect"] == effect)]
    if sub.empty:
        lines.append(f"- `{metric}` / `{effect}`: not available\n")
        return
    r = sub.iloc[0]
    lines.append(
        f"- `{metric}` `{effect}`: median={r.get('median_effect')} mean={r.get('mean_effect')} "
        f"n+={r.get('n_positive')} n-={r.get('n_negative')} "
        f"std_effect={r.get('standardized_paired_effect')} "
        f"bootCI=[{r.get('bootstrap_ci_low')}, {r.get('bootstrap_ci_high')}]"
    )
    lines.append("")
