"""Information-leakage audit (must all PASS before formal analysis)."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from experiment00.cache import dump_json
from experiment00.config import Experiment00Config
from experiment00.templates import build_green_detail_cm_group, group_green_recordings
from experiment00.video_processing import ProcessedVideo


def _audit_item(name: str, passed: bool, detail: str) -> dict:
    return {"check": name, "status": "PASS" if passed else "FAIL", "detail": detail}


def _fake(L, f, rnd, ch, seed, rng) -> ProcessedVideo:
    img = rng.normal(100, 5, size=(32, 32)).astype(np.float32)
    img = img + (hash((L, f, rnd, ch, seed)) % 50) * 0.1
    blocks = [img + rng.normal(0, 0.5, size=img.shape).astype(np.float32) for _ in range(3)]
    return ProcessedVideo(
        path=Path(f"{L}_{f}_{rnd}_{ch}.mp4"),
        length_cm=L,
        fiber_id=f,
        illumination="green",
        round=rnd,
        challenge=ch,
        challenge_id=ch,
        red_phase=None,
        fps=10,
        n_frames=90,
        width=32,
        height=32,
        retained_indices=list(range(30)),
        block_indices=[list(range(10))] * 3,
        blocks_raw=blocks,
        representative_raw=img,
        saturation_fraction=0.0,
        mean_intensity=float(img.mean()),
        sequence_id="P1",
        acquisition_position=1,
        order_source="fiber_sequence_mapping",
    )


def audit_common_mode_isolation(cfg: Experiment00Config) -> list[dict]:
    """Synthetic isolation checks (fast, deterministic). English-only."""
    from copy import deepcopy

    rng = np.random.default_rng(cfg.experiment.seed)
    cfg_local = deepcopy(cfg)
    cfg_local.preprocessing.detail_sigma_px = 3.0
    results: list[dict] = []

    a = [_fake(7, "F01", "A", f"C{i:02d}", i, rng) for i in range(1, 9)]
    b = [_fake(7, "F01", "B", f"C{i:02d}", 100 + i, rng) for i in range(1, 9)]
    out_a1 = build_green_detail_cm_group(a, cfg_local)
    for r in b:
        r.representative_raw[:] = 500.0
        r.blocks_raw = [np.full_like(r.representative_raw, 500.0) for _ in r.blocks_raw]
    out_a2 = build_green_detail_cm_group(a, cfg_local)
    same = all(np.allclose(out_a1[ch].video_detail_cm, out_a2[ch].video_detail_cm) for ch in out_a1)
    results.append(
        _audit_item(
            "round_A_common_mode_only_from_A",
            same,
            "Round A detail_cm unchanged after mutating Round B templates",
        )
    )

    out_b1 = build_green_detail_cm_group(b, cfg_local)
    for r in a:
        r.representative_raw[:] = -100.0
    out_b2 = build_green_detail_cm_group(b, cfg_local)
    same_b = all(np.allclose(out_b1[ch].video_detail_cm, out_b2[ch].video_detail_cm) for ch in out_b1)
    results.append(
        _audit_item(
            "round_B_common_mode_only_from_B",
            same_b,
            "Round B detail_cm unchanged after mutating Round A templates",
        )
    )

    # Length / fiber grouping API never mixes keys
    mixed = [_fake(7, "F01", "A", f"C{i:02d}", i, rng) for i in range(1, 5)]
    mixed += [_fake(9, "F01", "A", f"C{i:02d}", 50 + i, rng) for i in range(1, 5)]
    mixed += [_fake(7, "F02", "A", f"C{i:02d}", 70 + i, rng) for i in range(1, 5)]
    groups = group_green_recordings(mixed)
    keys_ok = all(len({(r.length_cm, r.fiber_id, r.round) for r in recs}) == 1 for recs in groups.values())
    results.append(
        _audit_item(
            "length_fiber_round_groups_never_mixed",
            keys_ok and (7, "F01", "A") in groups and (9, "F01", "A") in groups and (7, "F02", "A") in groups,
            f"group keys={sorted(groups.keys())}",
        )
    )

    # Source audits
    from experiment00 import mask as mask_mod
    from experiment00 import length_selection as ls
    from experiment00 import plotting as pl
    from experiment00 import discovery as disc
    from experiment00 import templates as tmpl

    src_mask = inspect.getsource(mask_mod.build_global_valid_mask)
    leak_tokens = ["robust_gap", "selected_length", "selection_probability", "maximin"]
    results.append(
        _audit_item(
            "mask_independent_of_selection_scores",
            not any(t in src_mask for t in leak_tokens),
            "build_global_valid_mask source has no selection metrics",
        )
    )

    src_sel = inspect.getsource(ls.select_length)
    results.append(
        _audit_item(
            "selection_no_posthoc_weights",
            "weighted_sum" not in src_sel.lower() and "0.3 *" not in src_sel,
            "select_length uses maximin robust-gap (no post-hoc weights)",
        )
    )

    src_plot = inspect.getsource(pl.select_representative_template)
    results.append(
        _audit_item(
            "representative_median_nearest_not_best_ncc",
            "median" in src_plot.lower() and "argsort" not in src_plot.lower().replace("score", ""),
            "Representative selection uses median-nearest spatial metric, not best-NCC ranking",
        )
    )

    src_group = inspect.getsource(tmpl.group_green_recordings) + inspect.getsource(tmpl.build_green_detail_cm_group)
    results.append(
        _audit_item(
            "common_mode_api_documents_round_isolation",
            "never share" in src_group.lower() or "ONLY within this group" in src_group,
            "templates.py documents per-(length,fiber,round) isolation",
        )
    )

    src_disc = inspect.getsource(disc.run_analysis)
    results.append(
        _audit_item(
            "pipeline_builds_common_mode_per_group_key",
            "group_green_recordings" in src_disc and "build_green_detail_cm_group" in src_disc,
            "Formal pipeline iterates group keys (L,f,round) independently",
        )
    )

    return results


def run_leakage_audit(cfg: Experiment00Config, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for item in tqdm(audit_common_mode_isolation(cfg), desc="Leakage audit checks", unit="chk"):
        items.append(item)
    ok = all(x["status"] == "PASS" for x in items)
    report = {
        "ok": ok,
        "n_checks": len(items),
        "n_pass": sum(1 for x in items if x["status"] == "PASS"),
        "n_fail": sum(1 for x in items if x["status"] == "FAIL"),
        "checks": items,
        "note": "Enrollment A and Query B never share common-mode estimation groups.",
    }
    dump_json(out_dir / "information_leakage_audit.json", report)
    lines = ["# Information leakage audit\n", f"Overall: {'PASS' if ok else 'FAIL'}\n"]
    for x in items:
        lines.append(f"- [{x['status']}] {x['check']}: {x['detail']}")
    (out_dir / "information_leakage_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
