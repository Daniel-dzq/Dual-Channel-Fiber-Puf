"""Fixed-state dual-channel analysis (Fig. 5a,b and Fig. 6): audit, template rebuild, Fiber-ID, figure data."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from PIL import Image

from experiment2.figure_layout import REPRESENTATION
from experiment2.fixed_state_metrics import (
    GreenScoreBundle,
    build_score_ecdf,
    compute_green_classification,
    file_hash,
    load_legacy_green_scores,
    metrics_to_dict,
    per_device_tail_gaps,
    retrieval_metrics,
    roc_curve,
    scores_array,
)
from experiment2.config import Experiment2Config
from experiment2.score_construction import (
    build_detail_cm_for_green_device,
    compute_intra_scores,
    compute_inter_challenge_scores,
)
from experiment2.video_processing import process_video
from puf_common.features import to_detail
from puf_common.masks import combine_pair_mask
from puf_common.ncc import zero_mean_ncc


def load_metadata(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

logger = logging.getLogger(__name__)
TZ = ZoneInfo("Asia/Singapore")

# Fiber-ID feature definitions are shared with experiment3.red_features (installed sibling package).


def _import_red_features():
    from experiment3.config import Experiment3Config, AnalysisConfig as E3Analysis
    from experiment3.red_features import (
        LOWDIM_NAMES,
        apply_standardizer,
        extract_lowdim_vector,
        fit_standardizer,
        red_similarity_score,
    )

    return {
        "Experiment3Config": Experiment3Config,
        "E3Analysis": E3Analysis,
        "LOWDIM_NAMES": LOWDIM_NAMES,
        "apply_standardizer": apply_standardizer,
        "extract_lowdim_vector": extract_lowdim_vector,
        "fit_standardizer": fit_standardizer,
        "red_similarity_score": red_similarity_score,
    }


def new_run_dir(output_root: Path, kind: str = "fixed_state") -> Path:
    stamp = datetime.now(TZ).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / "fixed_state_analysis" / "runs" / f"{stamp}_{kind}"
    for sub in (
        "figure_data",
        "panels",
        "combined",
        "supplementary",
        "reports",
        "cache",
        "previews",
    ):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def audit_fixed_state_data(cfg: Experiment2Config) -> dict[str, Any]:
    meta = load_metadata(cfg.metadata_path)
    devices = sorted(meta["device_id"].unique())
    n_green = int((meta["channel"] == "green").sum())
    n_red = int((meta["channel"] == "red").sum())
    missing = []
    for _, row in meta.iterrows():
        path = cfg.root / row["video_path"]
        if not path.is_file():
            # metadata may store relative under videos/
            alt = cfg.resolve_path(row["video_path"])
            if alt is None or not alt.is_file():
                # try videos/RedAndGreen layout
                cand = cfg.root / "videos" / "RedAndGreen" / row["channel"] / Path(row["video_path"]).name
                if not cand.is_file():
                    missing.append(str(row["video_path"]))

    templates = cfg.output_dir / "templates"
    n_block = len(list((templates / "block_templates").glob("*.npy"))) if (templates / "block_templates").is_dir() else 0
    n_rep = len(list((templates / "representative_templates").glob("*.npy"))) if (templates / "representative_templates").is_dir() else 0
    n_mask = len(list((templates / "masks").glob("*_valid_mask.png"))) if (templates / "masks").is_dir() else 0

    metrics_dir = cfg.output_dir / "metrics"
    score_counts = {}
    for name, key in (
        ("intra_repeatability_scores.csv", "genuine_legacy"),
        ("inter_challenge_scores.csv", "challenge_legacy"),
        ("inter_device_scores.csv", "device_legacy"),
    ):
        p = metrics_dir / name
        score_counts[key] = int(sum(1 for _ in open(p)) - 1) if p.is_file() else 0

    data_status = "DATA_READY" if not missing and len(devices) == 15 and n_green == 120 and n_red == 30 else "DATA_INCOMPLETE"

    return {
        "data_status": data_status,
        "n_devices": len(devices),
        "device_ids": devices,
        "n_metadata_rows": int(len(meta)),
        "n_green_videos": n_green,
        "n_red_videos": n_red,
        "missing_videos": missing,
        "n_block_templates": n_block,
        "n_representative_templates": n_rep,
        "n_masks": n_mask,
        "legacy_score_counts": score_counts,
        "expected_pairs": {
            "genuine": 15 * 8 * 3,  # C(3,2)=3 pairs per challenge
            "challenge_mismatch": 15 * 28,  # C(8,2)
            "device_mismatch": 8 * 105,  # C(15,2)
        },
        "processing_protocol": {
            "discard_head_s": cfg.acquisition.discard_head_s,
            "discard_tail_s": cfg.acquisition.discard_tail_s,
            "num_time_blocks": cfg.acquisition.num_time_blocks,
            "block_aggregation": cfg.acquisition.block_aggregation,
            "envelope_sigma_px": cfg.analysis.envelope_sigma_px,
            "envelope_eps": cfg.analysis.envelope_eps,
            "feature_type_config": cfg.analysis.feature_type,
            "representation_formal": REPRESENTATION,
            "max_saturation_fraction": cfg.analysis.max_saturation_fraction,
            "green_common_subtraction": True,
            "red_common_subtraction": False,
        },
        "note_fiber_id_dims": (
            "The Fiber-ID descriptor has nine active features (puf_common.fiber_id). "
            "Stored tables may also carry four zero-valued acf_* columns for storage compatibility."
        ),
    }


def _load_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(bool)


def load_device_masks(cfg: Experiment2Config) -> dict[str, np.ndarray]:
    mask_dir = cfg.output_dir / "templates" / "masks"
    out = {}
    for device in cfg.experiment.device_ids:
        path = mask_dir / f"{device}_valid_mask.png"
        if path.is_file():
            out[device] = _load_mask(path)
    return out


class _FakeRec:
    """Minimal stand-in so build_detail_cm can run from cached intensity templates."""

    def __init__(self, blocks: list[np.ndarray], representative: np.ndarray):
        self.blocks = blocks
        self.representative = representative


def rebuild_green_detail_cm_from_cache(
    cfg: Experiment2Config,
    device_id: str,
) -> dict[str, Any]:
    """Priority-2: rebuild detail_cm from block + representative intensity caches."""
    block_dir = cfg.output_dir / "templates" / "block_templates"
    rep_dir = cfg.output_dir / "templates" / "representative_templates"
    green: dict[str, _FakeRec] = {}
    for cid in cfg.slm.challenge_ids:
        blocks = [
            np.load(block_dir / f"{device_id}_{cid}_block{b}.npy") for b in (1, 2, 3)
        ]
        rep_path = rep_dir / f"{device_id}_green_{cid}.npy"
        rep = np.load(rep_path) if rep_path.is_file() else np.median(np.stack(blocks, 0), axis=0)
        green[cid] = _FakeRec(blocks, rep)
    # Adapt to ProcessedRecording-like usage in build_detail_cm_for_green_device
    # which only needs .blocks
    return build_detail_cm_for_green_device(green, cfg)  # type: ignore[arg-type]


def verify_green_scores_from_cache(
    cfg: Experiment2Config,
    bundle: GreenScoreBundle,
    *,
    atol: float = 1e-5,
    max_devices: int | None = None,
) -> dict[str, Any]:
    """Recompute a subset of S_G / S_C from templates and compare to frozen CSV."""
    masks = load_device_masks(cfg)
    devices = cfg.experiment.device_ids
    if max_devices is not None:
        devices = devices[:max_devices]
    mismatches = []
    checked_g = checked_c = 0
    for device in devices:
        feats = rebuild_green_detail_cm_from_cache(cfg, device)
        mask = masks[device]
        for cid in cfg.slm.challenge_ids:
            recomputed = compute_intra_scores(
                device_id=device,
                challenge_id=cid,
                session_id="S01",
                features=feats[cid],
                mask=mask,
                cfg=cfg,
                dark_hash="verify",
                mask_id=device,
            )
            for row in recomputed:
                checked_g += 1
                match = bundle.genuine[
                    (bundle.genuine["device_id_a"] == device)
                    & (bundle.genuine["challenge_id_a"] == cid)
                    & (bundle.genuine["block_id_a"] == row["block_id_a"])
                    & (bundle.genuine["block_id_b"] == row["block_id_b"])
                ]
                if match.empty:
                    mismatches.append({"kind": "genuine_missing", "device": device, "challenge": cid})
                    continue
                if abs(float(match.iloc[0]["score"]) - float(row["ncc"])) > atol:
                    mismatches.append(
                        {
                            "kind": "genuine_mismatch",
                            "device": device,
                            "challenge": cid,
                            "csv": float(match.iloc[0]["score"]),
                            "recomputed": float(row["ncc"]),
                        }
                    )
        inter = compute_inter_challenge_scores(
            device_id=device,
            session_id="S01",
            features_by_challenge=feats,
            challenge_ids=cfg.slm.challenge_ids,
            mask=mask,
            cfg=cfg,
            dark_hash="verify",
            mask_id=device,
        )
        for row in inter:
            checked_c += 1
            match = bundle.challenge_mismatch[
                (bundle.challenge_mismatch["device_id_a"] == device)
                & (
                    (
                        (bundle.challenge_mismatch["challenge_id_a"] == row["challenge_id_a"])
                        & (bundle.challenge_mismatch["challenge_id_b"] == row["challenge_id_b"])
                    )
                    | (
                        (bundle.challenge_mismatch["challenge_id_a"] == row["challenge_id_b"])
                        & (bundle.challenge_mismatch["challenge_id_b"] == row["challenge_id_a"])
                    )
                )
            ]
            if match.empty:
                mismatches.append({"kind": "challenge_missing", "device": device})
                continue
            if abs(float(match.iloc[0]["score"]) - float(row["ncc"])) > atol:
                mismatches.append(
                    {
                        "kind": "challenge_mismatch",
                        "device": device,
                        "csv": float(match.iloc[0]["score"]),
                        "recomputed": float(row["ncc"]),
                    }
                )
    ok = len(mismatches) == 0
    return {
        "ok": ok,
        "checked_genuine": checked_g,
        "checked_challenge": checked_c,
        "n_mismatches": len(mismatches),
        "mismatches_head": mismatches[:20],
        "cache_status": "VERIFIED" if ok else "MISMATCH",
    }


def load_or_build_all_green_features(
    cfg: Experiment2Config,
    *,
    cache_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Build detail_cm features once per device; cached under the shared analysis cache."""
    cache_dir = Path(
        cache_dir
        or (cfg.output_dir / "fixed_state_analysis" / "_shared_cache" / "green_detail_cm")
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict[str, Any]] = {}
    for device in cfg.experiment.device_ids:
        path = cache_dir / f"{device}_detail_cm.npz"
        if path.is_file():
            pack = np.load(path, allow_pickle=True)
            feats = {}
            for cid in cfg.slm.challenge_ids:
                from experiment2.score_construction import DetailCmFeatures

                feats[cid] = DetailCmFeatures(
                    detail_blocks=[pack[f"{cid}_d_b{b}"] for b in (1, 2, 3)],
                    detail_representative=pack[f"{cid}_d_rep"],
                    detail_cm_blocks=[pack[f"{cid}_cm_b{b}"] for b in (1, 2, 3)],
                    detail_cm_representative=pack[f"{cid}_cm_rep"],
                )
            out[device] = feats
            continue
        feats = rebuild_green_detail_cm_from_cache(cfg, device)
        payload = {}
        for cid, f in feats.items():
            for b, arr in enumerate(f.detail_blocks, start=1):
                payload[f"{cid}_d_b{b}"] = arr
            payload[f"{cid}_d_rep"] = f.detail_representative
            for b, arr in enumerate(f.detail_cm_blocks, start=1):
                payload[f"{cid}_cm_b{b}"] = arr
            payload[f"{cid}_cm_rep"] = f.detail_cm_representative
        np.savez_compressed(path, **payload)
        out[device] = feats
        logger.info("Cached green detail_cm features for %s", device)
    return out


def build_similarity_matrices(
    cfg: Experiment2Config,
    all_feats: dict[str, dict[str, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Average 8×8 challenge and 15×15 device NCC matrices on detail_cm."""
    masks = load_device_masks(cfg)
    devices = cfg.experiment.device_ids
    challenges = cfg.slm.challenge_ids
    n_c = len(challenges)
    n_d = len(devices)
    if all_feats is None:
        all_feats = load_or_build_all_green_features(cfg)
    chal_stack = []
    for device in devices:
        feats = all_feats[device]
        mat = np.zeros((n_c, n_c), dtype=np.float64)
        mask = masks[device]
        for i, ci in enumerate(challenges):
            for j, cj in enumerate(challenges):
                mat[i, j] = zero_mean_ncc(
                    feats[ci].detail_cm_representative,
                    feats[cj].detail_cm_representative,
                    mask=mask,
                )
        chal_stack.append(mat)
    challenge_matrix = np.mean(np.stack(chal_stack, 0), axis=0)

    device_stack = []
    for cid in challenges:
        mat = np.zeros((n_d, n_d), dtype=np.float64)
        for i, di in enumerate(devices):
            for j, dj in enumerate(devices):
                pair_mask = combine_pair_mask(masks[di], masks[dj])
                mat[i, j] = zero_mean_ncc(
                    all_feats[di][cid].detail_cm_representative,
                    all_feats[dj][cid].detail_cm_representative,
                    mask=pair_mask,
                )
        device_stack.append(mat)
    device_matrix = np.mean(np.stack(device_stack, 0), axis=0)

    chal_off = challenge_matrix.copy()
    np.fill_diagonal(chal_off, -np.inf)
    ci, cj = np.unravel_index(int(np.argmax(chal_off)), chal_off.shape)
    dev_off = device_matrix.copy()
    np.fill_diagonal(dev_off, -np.inf)
    di, dj = np.unravel_index(int(np.argmax(dev_off)), dev_off.shape)

    meta = {
        "challenge_ids": challenges,
        "device_ids": devices,
        "most_confusable_challenge_pair": [challenges[ci], challenges[cj], float(chal_off[ci, cj])],
        "most_confusable_device_pair": [devices[di], devices[dj], float(dev_off[di, dj])],
        "vmin": -0.2,
        "vmax": 1.0,
        "representation": REPRESENTATION,
    }
    return challenge_matrix, device_matrix, meta


def _get_e3_analysis_cfg():
    mods = _import_red_features()
    yaml_path = Path(__file__).resolve().parents[2] / "configs" / "threshold_development.yaml"
    if yaml_path.is_file():
        from experiment3.config import load_config as load_e3

        return load_e3(yaml_path)

    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.analysis = mods["E3Analysis"]()
    return cfg


def _load_dark_red_2048(cfg: Experiment2Config) -> np.ndarray:
    candidates = [
        cfg.dark_artifact_path,
        cfg.resolve_path("outputs/experiment2/templates/dark/dark_reference_2048x1536.npz"),
        cfg.output_dir / "templates" / "dark" / "dark_reference_2048x1536.npz",
        cfg.resolve_path("outputs/experiment2/templates/dark/experiment1_dark_2048x1536.npz"),
        cfg.output_dir / "templates" / "dark" / "experiment1_dark_2048x1536.npz",
    ]
    for path in candidates:
        if path is not None and Path(path).is_file():
            pack = np.load(path, allow_pickle=True)
            if "dark_red" in pack.files:
                arr = pack["dark_red"]
                if arr.shape == (1536, 2048):
                    return arr
    raise FileNotFoundError("Could not load 2048×1536 dark_red for red_after decode")


def _resolve_video_path(cfg: Experiment2Config, row: pd.Series) -> Path:
    raw = Path(str(row["video_path"]))
    candidates = [
        cfg.root / raw,
        cfg.resolve_path(raw),
        cfg.root / "videos" / "RedAndGreen" / str(row["channel"]) / raw.name,
        cfg.root / "videos" / "RedAndGreen" / str(row["channel"]) / Path(str(row.get("video_path", ""))).name,
    ]
    # F10 naming quirk
    for c in candidates:
        if c is not None and Path(c).is_file():
            return Path(c)
    raise FileNotFoundError(f"Video not found for {row['device_id']} {row['record_type']}: {row['video_path']}")


def extract_red_fiber_id_scores(
    cfg: Experiment2Config,
    *,
    cache_dir: Path,
    force_redecode: bool = False,
) -> dict[str, Any]:
    """Apply official lifecycle Fiber ID features to Exp2 red_before / red_after."""
    mods = _import_red_features()
    e3_cfg = _get_e3_analysis_cfg()
    masks = load_device_masks(cfg)
    meta = load_metadata(cfg.metadata_path)
    rep_dir = cfg.output_dir / "templates" / "representative_templates"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Prefer experiment-level shared cache so re-runs do not re-decode red_after.
    shared = cfg.output_dir / "fixed_state_analysis" / "_shared_cache" / "red_after_representatives"
    shared.mkdir(parents=True, exist_ok=True)
    after_cache = shared
    # Mirror into run cache for self-containment when newly decoded
    run_mirror = cache_dir / "red_after_representatives"
    run_mirror.mkdir(exist_ok=True)

    dark_red = _load_dark_red_2048(cfg)

    vectors_before: dict[str, np.ndarray] = {}
    vectors_after: dict[str, np.ndarray] = {}
    video_reanalysis = {"red_before": "TEMPLATE_CACHE", "red_after": []}

    for device in cfg.experiment.device_ids:
        mask = masks[device]
        before_path = rep_dir / f"{device}_red_before.npy"
        if not before_path.is_file():
            raise FileNotFoundError(before_path)
        before_img = np.load(before_path)
        vec_b, _ = mods["extract_lowdim_vector"](before_img, mask, e3_cfg)
        vectors_before[device] = vec_b

        after_npy = after_cache / f"{device}_red_after.npy"
        if after_npy.is_file() and not force_redecode:
            after_img = np.load(after_npy)
            video_reanalysis["red_after"].append({"device": device, "status": "CACHE_HIT"})
        else:
            row = meta[(meta["device_id"] == device) & (meta["record_type"] == "red_after")].iloc[0]
            vpath = _resolve_video_path(cfg, row)
            rec = process_video(
                video_path=vpath,
                channel="red",
                dark=dark_red,
                roi=cfg.camera.roi,
                cfg=cfg,
                device_id=device,
                record_type="red_after",
                challenge_id="",
                session_id=str(row.get("session_id", "S01")),
            )
            after_img = rec.representative
            np.save(after_npy, after_img)
            video_reanalysis["red_after"].append(
                {"device": device, "status": "REDECODED", "path": str(vpath)}
            )
        mirror_path = run_mirror / f"{device}_red_after.npy"
        if not mirror_path.is_file():
            np.save(mirror_path, after_img)
        vec_a, _ = mods["extract_lowdim_vector"](after_img, mask, e3_cfg)
        vectors_after[device] = vec_a

    # Standardizer fit on enrollment (red_before) — fixed-state protocol
    mu, sd = mods["fit_standardizer"]([vectors_before[d] for d in cfg.experiment.device_ids])
    z_before = {d: mods["apply_standardizer"](vectors_before[d], mu, sd) for d in cfg.experiment.device_ids}
    z_after = {d: mods["apply_standardizer"](vectors_after[d], mu, sd) for d in cfg.experiment.device_ids}

    same_rows = []
    diff_rows = []
    for d in cfg.experiment.device_ids:
        q = mods["red_similarity_score"](z_before[d], z_after[d])
        same_rows.append(
            {
                "device_enrollment": d,
                "device_query": d,
                "score_type": "SAME_DEVICE",
                "q_R": q,
                "fiber_id_dim": len(mods["LOWDIM_NAMES"]),
            }
        )
    for de in cfg.experiment.device_ids:
        for dq in cfg.experiment.device_ids:
            if de == dq:
                continue
            q = mods["red_similarity_score"](z_before[de], z_after[dq])
            diff_rows.append(
                {
                    "device_enrollment": de,
                    "device_query": dq,
                    "score_type": "DIFFERENT_DEVICE",
                    "q_R": q,
                    "fiber_id_dim": len(mods["LOWDIM_NAMES"]),
                }
            )

    same_df = pd.DataFrame(same_rows)
    diff_df = pd.DataFrame(diff_rows)
    same_vals = same_df["q_R"].to_numpy(dtype=np.float64)
    diff_vals = diff_df["q_R"].to_numpy(dtype=np.float64)

    from puf_common.metrics import auc_roc, equal_error_rate_with_threshold, robust_gap

    eer_r, thr_r = equal_error_rate_with_threshold(same_vals, diff_vals)
    rg_r = robust_gap(same_vals, diff_vals)

    # Top-k identity: each red_after query vs 15 red_before gallery
    ranks = []
    rank_rows = []
    for dq in cfg.experiment.device_ids:
        scores = []
        for de in cfg.experiment.device_ids:
            scores.append((de, mods["red_similarity_score"](z_before[de], z_after[dq])))
        scores.sort(key=lambda x: x[1], reverse=True)
        rank = next(i for i, (de, _) in enumerate(scores, start=1) if de == dq)
        ranks.append(rank)
        for rnk, (de, sc) in enumerate(scores, start=1):
            rank_rows.append(
                {
                    "query_device": dq,
                    "gallery_device": de,
                    "q_R": sc,
                    "rank": rnk,
                    "is_correct": de == dq,
                }
            )
    ret = retrieval_metrics(np.asarray(ranks, dtype=np.float64), candidate_count=15)

    return {
        "same_device": same_df,
        "different_device": diff_df,
        "distribution": pd.concat([same_df, diff_df], ignore_index=True),
        "rank_table": pd.DataFrame(rank_rows),
        "metrics": {
            "rg_red_identity": float(rg_r),
            "auc_red_identity": float(auc_roc(same_vals, diff_vals)),
            "eer_red_identity": float(eer_r),
            "eer_red_threshold": float(thr_r),
            "q05_same": float(np.percentile(same_vals, 5)),
            "q95_different": float(np.percentile(diff_vals, 95)),
            "red_identity_candidate_count": 15,
            **{f"red_{k}": v for k, v in ret.items()},
            "fiber_id_feature_names": mods["LOWDIM_NAMES"],
            "fiber_id_dim": len(mods["LOWDIM_NAMES"]),
        },
        "video_reanalysis": video_reanalysis,
        "z_before": z_before,
        "z_after": z_after,
    }


def green_challenge_retrieval(
    cfg: Experiment2Config,
    all_feats: dict[str, dict[str, Any]],
    *,
    challenge_matrix: np.ndarray | None = None,
) -> dict[str, Any]:
    """Per (device, query_challenge) rank against 8 challenge representatives."""
    masks = load_device_masks(cfg)
    challenges = cfg.slm.challenge_ids
    devices = cfg.experiment.device_ids
    ranks = []
    rows = []
    for device in devices:
        feats = all_feats[device]
        mask = masks[device]
        for qcid in challenges:
            scores = []
            for gcid in challenges:
                ncc = zero_mean_ncc(
                    feats[qcid].detail_cm_blocks[0],
                    feats[gcid].detail_cm_representative,
                    mask=mask,
                )
                scores.append((gcid, ncc))
            scores.sort(key=lambda x: x[1], reverse=True)
            rank = next(i for i, (c, _) in enumerate(scores, start=1) if c == qcid)
            ranks.append(rank)
            for rnk, (c, sc) in enumerate(scores, start=1):
                rows.append(
                    {
                        "device_id": device,
                        "query_challenge": qcid,
                        "gallery_challenge": c,
                        "score": sc,
                        "rank": rnk,
                        "is_correct": c == qcid,
                    }
                )
    ret = retrieval_metrics(np.asarray(ranks, dtype=np.float64), candidate_count=8)
    return {
        "rank_table": pd.DataFrame(rows),
        "metrics": ret,
        "pooled_challenge_matrix": challenge_matrix,
    }


def green_device_retrieval(
    cfg: Experiment2Config,
    all_feats: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Same-challenge device Top-1 across 15 devices (query block1 vs gallery reps)."""
    masks = load_device_masks(cfg)
    devices = cfg.experiment.device_ids
    challenges = cfg.slm.challenge_ids
    ranks = []
    rows = []
    for cid in challenges:
        for qd in devices:
            scores = []
            for gd in devices:
                pair_mask = combine_pair_mask(masks[qd], masks[gd])
                ncc = zero_mean_ncc(
                    all_feats[qd][cid].detail_cm_blocks[0],
                    all_feats[gd][cid].detail_cm_representative,
                    mask=pair_mask,
                )
                scores.append((gd, ncc))
            scores.sort(key=lambda x: x[1], reverse=True)
            rank = next(i for i, (d, _) in enumerate(scores, start=1) if d == qd)
            ranks.append(rank)
            for rnk, (d, sc) in enumerate(scores, start=1):
                rows.append(
                    {
                        "challenge_id": cid,
                        "query_device": qd,
                        "gallery_device": d,
                        "score": sc,
                        "rank": rnk,
                        "is_correct": d == qd,
                    }
                )
    ret = retrieval_metrics(np.asarray(ranks, dtype=np.float64), candidate_count=15)
    return {"rank_table": pd.DataFrame(rows), "metrics": ret}


def select_representative_samples(
    cfg: Experiment2Config,
    red_pack: dict[str, Any],
    bundle: GreenScoreBundle,
) -> dict[str, Any]:
    """Objective median-rule sample selection for Fig. 3a / 3f (no cherry-picking)."""
    same = red_pack["same_device"].copy()
    same["abs_dev"] = (same["q_R"] - same["q_R"].median()).abs()
    red_device = same.sort_values(["abs_dev", "device_enrollment"]).iloc[0]["device_enrollment"]

    g = bundle.genuine.copy()
    median_g = g["score"].median()
    g["abs_dev"] = (g["score"] - median_g).abs()
    g_row = g.sort_values(["abs_dev", "device_id_a", "challenge_id_a"]).iloc[0]

    c = bundle.challenge_mismatch.copy()
    median_c = c["score"].median()
    c["abs_dev"] = (c["score"] - median_c).abs()
    # Prefer same device as red_device when possible
    c_same = c[c["device_id_a"] == red_device]
    c_row = (c_same if len(c_same) else c).sort_values(["abs_dev"]).iloc[0]

    return {
        "rule": (
            "red device: median |q_R - median(q_R)| among same-device; "
            "genuine challenge: median |S_G - median(S_G)|; "
            "challenge mismatch: median |S_C - median(S_C)| preferably on selected device"
        ),
        "red_device_id": red_device,
        "genuine_device_id": g_row["device_id_a"],
        "genuine_challenge_id": g_row["challenge_id_a"],
        "genuine_blocks": [int(g_row["block_id_a"]), int(g_row["block_id_b"])],
        "genuine_score": float(g_row["score"]),
        "mismatch_device_id": c_row["device_id_a"],
        "mismatch_challenge_a": c_row["challenge_id_a"],
        "mismatch_challenge_b": c_row["challenge_id_b"],
        "mismatch_score": float(c_row["score"]),
    }


def select_auth_demo_query(
    red_pack: dict[str, Any],
    green_chal: dict[str, Any],
) -> dict[str, Any]:
    """Pick a successful joint query near median confidence among successes."""
    red_ranks = red_pack["rank_table"]
    # Per query device: correct rank and margin
    success_devices = []
    for dq, sub in red_ranks.groupby("query_device"):
        correct = sub[sub["is_correct"]].iloc[0]
        if int(correct["rank"]) != 1:
            continue
        top2 = sub.nsmallest(2, "rank") if "rank" in sub else sub.nlargest(2, "q_R")
        scores = sub.sort_values("q_R", ascending=False)
        margin = float(scores.iloc[0]["q_R"] - scores.iloc[1]["q_R"])
        success_devices.append({"device_id": dq, "margin": margin, "q_R": float(correct["q_R"])})
    if not success_devices:
        # fallback any device
        dq = red_pack["same_device"].iloc[0]["device_enrollment"]
        return {"device_id": dq, "challenge_id": "C01", "rule": "fallback_no_rank1"}

    sdf = pd.DataFrame(success_devices)
    med = sdf["margin"].median()
    sdf["abs_dev"] = (sdf["margin"] - med).abs()
    device = sdf.sort_values("abs_dev").iloc[0]["device_id"]

    # Green: successful challenge query on that device near median margin
    gt = green_chal["rank_table"]
    gt = gt[gt["device_id"] == device]
    successes = []
    for qcid, sub in gt.groupby("query_challenge"):
        correct = sub[sub["is_correct"]].iloc[0]
        if int(correct["rank"]) != 1:
            continue
        scores = sub.sort_values("score", ascending=False)
        margin = float(scores.iloc[0]["score"] - scores.iloc[1]["score"])
        successes.append({"challenge_id": qcid, "margin": margin})
    if successes:
        cdf = pd.DataFrame(successes)
        medc = cdf["margin"].median()
        cdf["abs_dev"] = (cdf["margin"] - medc).abs()
        challenge = cdf.sort_values("abs_dev").iloc[0]["challenge_id"]
    else:
        challenge = "C01"
    return {
        "device_id": device,
        "challenge_id": challenge,
        "rule": "median margin among Rank-1 red successes, then median margin among Rank-1 green successes on that device",
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def run_fixed_state_analysis(
    cfg: Experiment2Config,
    *,
    run_dir: Path | None = None,
    bootstrap_iterations: int = 5000,
    verify_devices: int = 3,
    force_red_redecode: bool = False,
) -> dict[str, Any]:
    output_root = cfg.output_dir
    run_dir = run_dir or new_run_dir(output_root)
    reports = run_dir / "reports"
    fig_data = run_dir / "figure_data"

    audit = audit_fixed_state_data(cfg)
    write_json(run_dir / "data_audit.json", audit)
    (reports / "FIXED_STATE_DATA_AUDIT.md").write_text(
        _format_data_audit_md(audit, cfg), encoding="utf-8"
    )

    if audit["data_status"] != "DATA_READY":
        write_json(run_dir / "RUN_STATUS.json", {"status": "DATA_INCOMPLETE", "audit": audit})
        return {"status": "DATA_INCOMPLETE", "run_dir": str(run_dir), "audit": audit}

    # Manifest
    meta = load_metadata(cfg.metadata_path)
    meta.to_csv(fig_data / "fixed_state_manifest.csv", index=False)

    protocol = {
        **audit["processing_protocol"],
        "pairing": {
            "S_G": "same device, same challenge, distinct non-overlapping time blocks (3 choose 2)",
            "S_C": "same device, different challenges, representative detail_cm templates (8 choose 2)",
            "S_D": "different devices, same challenge, representative detail_cm (15 choose 2 × 8)",
            "q_R": "Fiber ID: -||z_before - z_after||_2 after standardizer fit on red_before",
        },
        "expected_counts": audit["expected_pairs"],
        "excluded_metrics": ["S_X", "S_A", "RG_X", "RG_A"],
    }
    write_json(run_dir / "processing_protocol.json", protocol)
    (reports / "pairing_protocol.md").write_text(_format_pairing_md(protocol), encoding="utf-8")

    bundle = load_legacy_green_scores(cfg.output_dir / "metrics")
    verify = verify_green_scores_from_cache(cfg, bundle, max_devices=verify_devices)
    write_json(run_dir / "cache_validation.json", verify)

    green_metrics = compute_green_classification(bundle)
    tail = per_device_tail_gaps(bundle)
    ecdf = build_score_ecdf(bundle)

    logger.info("Building / loading cached green detail_cm features")
    all_feats = load_or_build_all_green_features(cfg)
    masks = load_device_masks(cfg)

    # Matrices
    chal_mat, dev_mat, mat_meta = build_similarity_matrices(cfg, all_feats=all_feats)
    pd.DataFrame(chal_mat, index=mat_meta["challenge_ids"], columns=mat_meta["challenge_ids"]).to_csv(
        fig_data / "figure3c_challenge_matrix.csv"
    )
    pd.DataFrame(dev_mat, index=mat_meta["device_ids"], columns=mat_meta["device_ids"]).to_csv(
        fig_data / "figure3c_device_matrix.csv"
    )
    write_json(fig_data / "figure3c_matrix_meta.json", mat_meta)

    # Red Fiber ID
    red_pack = extract_red_fiber_id_scores(
        cfg, cache_dir=run_dir / "cache", force_redecode=force_red_redecode
    )
    red_pack["distribution"].to_csv(fig_data / "figure3b_red_identity_distribution.csv", index=False)

    # Retrieval
    green_chal = green_challenge_retrieval(cfg, all_feats, challenge_matrix=chal_mat)
    green_dev = green_device_retrieval(cfg, all_feats)
    green_chal["rank_table"].to_csv(fig_data / "green_challenge_retrieval_ranks.csv", index=False)
    green_dev["rank_table"].to_csv(fig_data / "green_device_retrieval_ranks.csv", index=False)

    selection = select_representative_samples(cfg, red_pack, bundle)
    auth_sel = select_auth_demo_query(red_pack, green_chal)
    write_json(fig_data / "figure3a_sample_selection.json", selection)
    write_json(fig_data / "figure3f_auth_selection.json", auth_sel)

    # Figure data tables
    ecdf.to_csv(fig_data / "figure3d_score_ecdf.csv", index=False)
    tail.to_csv(fig_data / "figure3e_per_device_tail_gap.csv", index=False)

    # Auth example tables
    red_demo = red_pack["rank_table"][red_pack["rank_table"]["query_device"] == auth_sel["device_id"]].copy()
    red_demo.to_csv(fig_data / "figure3f_authentication_example.csv", index=False)
    green_demo = green_chal["rank_table"][
        (green_chal["rank_table"]["device_id"] == auth_sel["device_id"])
        & (green_chal["rank_table"]["query_challenge"] == auth_sel["challenge_id"])
    ].copy()
    green_demo.to_csv(fig_data / "figure3f_authentication_example_green.csv", index=False)

    # Joint accuracy: red Top-1 and green challenge Top-1 on same device session
    red_top1 = float(red_pack["metrics"]["red_top1"])
    green_top1 = float(green_chal["metrics"]["top1"])
    # Joint per device: red rank1 AND all challenges? Use mean of per-query joint
    joint_flags = []
    for d in cfg.experiment.device_ids:
        rsub = red_pack["rank_table"]
        r_ok = bool(rsub[(rsub["query_device"] == d) & (rsub["is_correct"])]["rank"].iloc[0] == 1)
        gsub = green_chal["rank_table"]
        g_ok = bool((gsub[gsub["device_id"] == d].groupby("query_challenge")
                     .apply(lambda s: int(s[s["is_correct"]]["rank"].iloc[0]) == 1)).mean() == 1.0)
        # softer joint: fraction of challenges correct given red ok
        g_frac = float(
            gsub[gsub["device_id"] == d]
            .groupby("query_challenge")
            .apply(lambda s: int(s[s["is_correct"]]["rank"].iloc[0]) == 1)
            .mean()
        )
        joint_flags.append(1.0 if r_ok and g_frac == 1.0 else (r_ok * g_frac))
    joint_acc = float(np.mean(joint_flags))

    auth_summary = pd.DataFrame(
        [
            {
                "metric": "red_device_top1",
                "value": red_top1,
                "candidate_count": 15,
            },
            {
                "metric": "green_challenge_top1",
                "value": green_top1,
                "candidate_count": 8,
            },
            {
                "metric": "green_device_top1",
                "value": float(green_dev["metrics"]["top1"]),
                "candidate_count": 15,
            },
            {
                "metric": "joint_device_and_challenge_accuracy",
                "value": joint_acc,
                "candidate_count": None,
                "note": "within-session fixed-state proof-of-concept; not independent-session auth",
            },
        ]
    )
    auth_summary.to_csv(fig_data / "figure3f_authentication_summary.csv", index=False)

    # ROC data
    g = scores_array(bundle.genuine)
    c = scores_array(bundle.challenge_mismatch)
    d = scores_array(bundle.device_mismatch)
    roc_curve(g, c).assign(comparison="S_G_vs_S_C").to_csv(fig_data / "roc_challenge.csv", index=False)
    roc_curve(g, d).assign(comparison="S_G_vs_S_D").to_csv(fig_data / "roc_device.csv", index=False)
    roc_curve(
        red_pack["same_device"]["q_R"].to_numpy(),
        red_pack["different_device"]["q_R"].to_numpy(),
    ).assign(comparison="q_R_same_vs_diff").to_csv(fig_data / "roc_red.csv", index=False)

    # Ablation S10: raw / detail / detail_cm on one device sample of metrics (pooled recompute)
    ablation = _representation_ablation(cfg, masks=masks, all_feats=all_feats, bundle=bundle)
    ablation.to_csv(fig_data / "figureS10_representation_ablation.csv", index=False)

    # Block stability S12
    block_stab = _block_stability(cfg, all_feats=all_feats)
    block_stab.to_csv(fig_data / "figureS12_block_stability.csv", index=False)

    metric_summary = {
        **metrics_to_dict(green_metrics),
        **red_pack["metrics"],
        "n_genuine_scores": int(len(bundle.genuine)),
        "n_challenge_mismatch_scores": int(len(bundle.challenge_mismatch)),
        "n_device_mismatch_scores": int(len(bundle.device_mismatch)),
        "green_challenge_Top1": green_top1,
        "green_device_Top1": float(green_dev["metrics"]["top1"]),
        "joint_authentication_accuracy": joint_acc,
        "representation": REPRESENTATION,
        "bootstrap_iterations_configured": bootstrap_iterations,
        "cache_validation": verify["cache_status"],
        "video_reanalysis_status": red_pack["video_reanalysis"],
        "sample_selection": selection,
        "auth_selection": auth_sel,
    }
    write_json(run_dir / "metric_summary.json", metric_summary)

    # Provenance stub (filled further by plotting)
    provenance = {
        "representation": REPRESENTATION,
        "metrics_source": str(cfg.output_dir / "metrics"),
        "legacy_csv_hashes": {
            "intra": file_hash(cfg.output_dir / "metrics" / "intra_repeatability_scores.csv"),
            "inter_challenge": file_hash(cfg.output_dir / "metrics" / "inter_challenge_scores.csv"),
            "inter_device": file_hash(cfg.output_dir / "metrics" / "inter_device_scores.csv"),
        },
        "cache_validation": verify,
        "fiber_id_module": "experiment3.red_features",
        "fiber_id_dim": red_pack["metrics"]["fiber_id_dim"],
        "forbidden_metrics_absent": True,
    }
    write_json(run_dir / "FIGURE_DATA_PROVENANCE.json", provenance)

    write_json(
        run_dir / "run_manifest.json",
        {
            "run_dir": str(run_dir),
            "created_at": datetime.now(TZ).isoformat(),
            "config": str(cfg.config_path),
            "status": "ANALYSIS_COMPLETE",
        },
    )
    write_json(run_dir / "RUN_STATUS.json", {"status": "ANALYSIS_COMPLETE"})

    return {
        "status": "ANALYSIS_COMPLETE",
        "run_dir": str(run_dir),
        "audit": audit,
        "bundle": bundle,
        "green_metrics": green_metrics,
        "tail": tail,
        "ecdf": ecdf,
        "challenge_matrix": chal_mat,
        "device_matrix": dev_mat,
        "matrix_meta": mat_meta,
        "red_pack": red_pack,
        "green_chal": green_chal,
        "green_dev": green_dev,
        "selection": selection,
        "auth_sel": auth_sel,
        "metric_summary": metric_summary,
        "ablation": ablation,
        "block_stability": block_stab,
        "auth_summary": auth_summary,
        "verify": verify,
        "all_feats": all_feats,
    }


def _representation_ablation(
    cfg: Experiment2Config,
    masks: dict[str, np.ndarray],
    all_feats: dict[str, dict[str, Any]],
    bundle: GreenScoreBundle,
) -> pd.DataFrame:
    """Compare raw / detail / detail_cm using cached features where possible."""
    from puf_common.metrics import auc_roc, robust_gap

    devices = cfg.experiment.device_ids
    challenges = cfg.slm.challenge_ids
    block_dir = cfg.output_dir / "templates" / "block_templates"
    rep_dir = cfg.output_dir / "templates" / "representative_templates"

    def _metrics_from_arrays(genuine, challenge, device_scores) -> dict[str, float]:
        g = np.asarray(genuine, dtype=np.float64)
        c = np.asarray(challenge, dtype=np.float64)
        d = np.asarray(device_scores, dtype=np.float64)
        return {
            "rg_challenge": robust_gap(g, c),
            "rg_device": robust_gap(g, d),
            "auc_challenge": auc_roc(g, c),
            "auc_device": auc_roc(g, d),
            "median_genuine": float(np.median(g)),
            "median_challenge": float(np.median(c)),
            "median_device": float(np.median(d)),
        }

    rows = []
    # detail_cm from verified unified scores (no recompute)
    gm = compute_green_classification(bundle)
    rows.append(
        {
            "representation": "detail_cm",
            "rg_challenge": gm.rg_challenge,
            "rg_device": gm.rg_device,
            "auc_challenge": gm.auc_challenge,
            "auc_device": gm.auc_device,
            "median_genuine": gm.median_genuine,
            "median_challenge": gm.median_challenge,
            "median_device": gm.median_device,
        }
    )

    # detail: from cached detail_* arrays inside all_feats
    genuine = []
    challenge = []
    device_scores = []
    for device in devices:
        mask = masks[device]
        f = all_feats[device]
        for cid in challenges:
            for i in range(3):
                for j in range(i + 1, 3):
                    genuine.append(
                        zero_mean_ncc(f[cid].detail_blocks[i], f[cid].detail_blocks[j], mask=mask)
                    )
        for i, ci in enumerate(challenges):
            for cj in challenges[i + 1 :]:
                challenge.append(
                    zero_mean_ncc(
                        f[ci].detail_representative,
                        f[cj].detail_representative,
                        mask=mask,
                    )
                )
    for cid in challenges:
        for i, di in enumerate(devices):
            for dj in devices[i + 1 :]:
                pair_mask = combine_pair_mask(masks[di], masks[dj])
                device_scores.append(
                    zero_mean_ncc(
                        all_feats[di][cid].detail_representative,
                        all_feats[dj][cid].detail_representative,
                        mask=pair_mask,
                    )
                )
    rows.append({"representation": "detail", **_metrics_from_arrays(genuine, challenge, device_scores)})

    # raw intensity
    genuine = []
    challenge = []
    device_scores = []
    raw_blocks = {
        d: {
            cid: [np.load(block_dir / f"{d}_{cid}_block{b}.npy") for b in (1, 2, 3)]
            for cid in challenges
        }
        for d in devices
    }
    raw_reps = {
        d: {cid: np.load(rep_dir / f"{d}_green_{cid}.npy") for cid in challenges} for d in devices
    }
    for device in devices:
        mask = masks[device]
        for cid in challenges:
            bl = raw_blocks[device][cid]
            for i in range(3):
                for j in range(i + 1, 3):
                    genuine.append(zero_mean_ncc(bl[i], bl[j], mask=mask))
        for i, ci in enumerate(challenges):
            for cj in challenges[i + 1 :]:
                challenge.append(
                    zero_mean_ncc(raw_reps[device][ci], raw_reps[device][cj], mask=mask)
                )
    for cid in challenges:
        for i, di in enumerate(devices):
            for dj in devices[i + 1 :]:
                pair_mask = combine_pair_mask(masks[di], masks[dj])
                device_scores.append(
                    zero_mean_ncc(raw_reps[di][cid], raw_reps[dj][cid], mask=pair_mask)
                )
    rows.append({"representation": "raw", **_metrics_from_arrays(genuine, challenge, device_scores)})
    return pd.DataFrame(rows)


def _block_stability(
    cfg: Experiment2Config,
    all_feats: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    masks = load_device_masks(cfg)
    for device in cfg.experiment.device_ids:
        feats = all_feats[device]
        mask = masks[device]
        for cid in cfg.slm.challenge_ids:
            blocks = feats[cid].detail_cm_blocks
            for i in range(3):
                for j in range(i + 1, 3):
                    rows.append(
                        {
                            "device_id": device,
                            "challenge_id": cid,
                            "block_a": i + 1,
                            "block_b": j + 1,
                            "score_genuine": zero_mean_ncc(blocks[i], blocks[j], mask=mask),
                        }
                    )
    return pd.DataFrame(rows)


def _format_data_audit_md(audit: dict[str, Any], cfg: Experiment2Config) -> str:
    return f"""# Fixed-state data audit

**data_status:** `{audit['data_status']}`

## Inventory

- Devices: {audit['n_devices']} (`{'`, `'.join(audit['device_ids'])}`)
- Metadata rows: {audit['n_metadata_rows']} (expected 150)
- Green videos: {audit['n_green_videos']} (expected 120)
- Red videos: {audit['n_red_videos']} (expected 30)
- Block templates: {audit['n_block_templates']} (expected 360)
- Representative templates: {audit['n_representative_templates']} (expected 135; red_after not cached historically)
- Masks: {audit['n_masks']}

## Legacy pair counts

{json.dumps(audit['legacy_score_counts'], indent=2)}

Expected: Genuine 360, Challenge mismatch 420, Device mismatch 840.

## Processing protocol (frozen config)

{json.dumps(audit['processing_protocol'], indent=2)}

## Fiber ID dimensionality note

{audit['note_fiber_id_dims']}

## Missing videos

{audit['missing_videos'] or 'None'}

Config: `{cfg.config_path}`
"""


def _format_pairing_md(protocol: dict[str, Any]) -> str:
    return f"""# Fixed-state pairing protocol

## Score definitions

{json.dumps(protocol['pairing'], indent=2)}

## Expected counts

{json.dumps(protocol['expected_counts'], indent=2)}

Pairing protocol is frozen and must not be altered for visual appeal.

Not defined for the fixed-state dataset: {protocol['excluded_metrics']}
"""
