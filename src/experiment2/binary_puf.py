"""Standard binary PUF metrics of the fixed-state green responses (Fig. 6c-d).

Uses formal representation fullres_detail_cm_fixed_state from shared detail_cm cache.
Bit positions are shared across all devices (common valid mask + frozen grid).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

DEVICES = [f"F{i:02d}" for i in range(1, 16)]
CHALLENGES = [f"C{i:02d}" for i in range(1, 9)]
BLOCKS = (1, 2, 3)

# Frozen encoding knobs — do not retune against metrics.
BINARY_SEED = 20260729
GRID_STEP_PX = 16
CODE_LENGTH_M = 2048
THRESHOLD = 0.0  # global fixed zero threshold on detail_cm
CORR_TARGET_LAG = 1.0 / np.e


@dataclass(frozen=True)
class BinaryEncodingConfig:
    representation: str = "fullres_detail_cm_fixed_state"
    cache_key_pattern: str = "{cid}_cm_{kind}"  # kind = b1|b2|b3|rep
    grid_step_px: int = GRID_STEP_PX
    code_length_M: int = CODE_LENGTH_M
    threshold: float = THRESHOLD
    random_seed: int = BINARY_SEED
    subsample_method: str = "linspace_raster"
    corr_length_estimate_px: float = 5.0
    M_raw: int = 0
    M_effective_estimate: int = 0
    n_common_mask_pixels: int = 0
    bit_definition_status: str = "common_mask_shared_grid"


def _load_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def build_common_mask(output_dir: Path) -> np.ndarray:
    masks_dir = output_dir / "templates" / "masks"
    masks = [_load_mask(masks_dir / f"{d}_valid_mask.png") for d in DEVICES]
    return np.logical_and.reduce(masks)


def estimate_corr_length_px(output_dir: Path, common: np.ndarray, *, n_samples: int = 8) -> float:
    """Enrollment ACF lag where correlation falls below 1/e (median over samples)."""
    cache = output_dir / "fixed_state_analysis" / "_shared_cache" / "green_detail_cm"
    lags: list[float] = []
    ys, xs = np.where(common)
    cy = int(np.median(ys))
    for di, device in enumerate(DEVICES[:n_samples]):
        z = np.load(cache / f"{device}_detail_cm.npz")
        cid = CHALLENGES[di % len(CHALLENGES)]
        img = z[f"{cid}_cm_rep"]
        row = img[cy]
        m = common[cy]
        vals = row[m].astype(np.float64)
        vals = vals - vals.mean()
        n = min(400, vals.size)
        if n < 40:
            continue
        v = vals[:n]
        acf = np.correlate(v, v, mode="full")[n - 1 :]
        if acf[0] <= 0:
            continue
        acf = acf / acf[0]
        hit = np.where(acf < CORR_TARGET_LAG)[0]
        if hit.size:
            lags.append(float(hit[0]))
    return float(np.median(lags)) if lags else 5.0


def build_shared_bit_coords(
    common: np.ndarray,
    *,
    grid_step: int,
    code_length: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return (ys, xs) shared sampling coordinates and (M_raw, M)."""
    ys_m, xs_m = np.where(common)
    y0, y1 = int(ys_m.min()), int(ys_m.max())
    x0, x1 = int(xs_m.min()), int(xs_m.max())
    yy = np.arange(y0, y1 + 1, grid_step)
    xx = np.arange(x0, x1 + 1, grid_step)
    YY, XX = np.meshgrid(yy, xx, indexing="ij")
    sel = common[YY, XX]
    ys = YY[sel].astype(np.int32).ravel()
    xs = XX[sel].astype(np.int32).ravel()
    # Raster order
    order = np.lexsort((xs, ys))
    ys, xs = ys[order], xs[order]
    m_raw = int(ys.size)
    if m_raw > code_length:
        idx = np.linspace(0, m_raw - 1, code_length).astype(np.int64)
        ys, xs = ys[idx], xs[idx]
    m_eff = int(ys.size)
    return ys, xs, m_raw, m_eff


def _extract_bits(img: np.ndarray, ys: np.ndarray, xs: np.ndarray, threshold: float) -> np.ndarray:
    vals = img[ys, xs].astype(np.float64)
    return (vals > threshold).astype(np.uint8)


def encode_all_binary_codes(
    output_dir: Path,
    *,
    ys: np.ndarray,
    xs: np.ndarray,
    threshold: float = THRESHOLD,
) -> dict[str, Any]:
    """Encode block and representative binary codes for all devices/challenges."""
    cache = output_dir / "fixed_state_analysis" / "_shared_cache" / "green_detail_cm"
    blocks: dict[tuple[str, str, int], np.ndarray] = {}
    reps: dict[tuple[str, str], np.ndarray] = {}
    for device in DEVICES:
        z = np.load(cache / f"{device}_detail_cm.npz")
        for cid in CHALLENGES:
            for b in BLOCKS:
                blocks[(device, cid, b)] = _extract_bits(z[f"{cid}_cm_b{b}"], ys, xs, threshold)
            reps[(device, cid)] = _extract_bits(z[f"{cid}_cm_rep"], ys, xs, threshold)
    return {"blocks": blocks, "reps": reps, "M": int(ys.size)}


def normalized_hd(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return float("nan")
    return float(np.mean(a != b))


def compute_hd_tables(codes: dict[str, Any]) -> dict[str, pd.DataFrame]:
    blocks = codes["blocks"]
    reps = codes["reps"]
    rows_g: list[dict[str, Any]] = []
    for device in DEVICES:
        for cid in CHALLENGES:
            for i, b1 in enumerate(BLOCKS):
                for b2 in BLOCKS[i + 1 :]:
                    hd = normalized_hd(blocks[(device, cid, b1)], blocks[(device, cid, b2)])
                    rows_g.append(
                        {
                            "pair_type": "HD_G",
                            "label": "Intra-HD (BER)",
                            "device": device,
                            "challenge": cid,
                            "block_a": b1,
                            "block_b": b2,
                            "hd": hd,
                        }
                    )
    rows_c: list[dict[str, Any]] = []
    for device in DEVICES:
        for i, c1 in enumerate(CHALLENGES):
            for c2 in CHALLENGES[i + 1 :]:
                hd = normalized_hd(reps[(device, c1)], reps[(device, c2)])
                rows_c.append(
                    {
                        "pair_type": "HD_C",
                        "label": "Inter-challenge HD",
                        "device": device,
                        "challenge_a": c1,
                        "challenge_b": c2,
                        "hd": hd,
                    }
                )
    rows_d: list[dict[str, Any]] = []
    for cid in CHALLENGES:
        for i, d1 in enumerate(DEVICES):
            for d2 in DEVICES[i + 1 :]:
                hd = normalized_hd(reps[(d1, cid)], reps[(d2, cid)])
                rows_d.append(
                    {
                        "pair_type": "HD_D",
                        "label": "Inter-device HD",
                        "challenge": cid,
                        "device_a": d1,
                        "device_b": d2,
                        "hd": hd,
                    }
                )
    return {
        "HD_G": pd.DataFrame(rows_g),
        "HD_C": pd.DataFrame(rows_c),
        "HD_D": pd.DataFrame(rows_d),
    }


def _bootstrap_mean(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int = 5000,
    seed: int = BINARY_SEED,
) -> tuple[float, float, float]:
    """Hierarchical-ish bootstrap: resample groups, then pairs within groups."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    point = float(np.mean(values))
    boots = np.empty(n_boot, dtype=np.float64)
    grouped = {g: values[groups == g] for g in uniq}
    for i in range(n_boot):
        chosen = rng.choice(uniq, size=uniq.size, replace=True)
        sample = []
        for g in chosen:
            arr = grouped[g]
            if arr.size == 0:
                continue
            sample.append(rng.choice(arr, size=arr.size, replace=True))
        boots[i] = float(np.mean(np.concatenate(sample))) if sample else point
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def compute_uniformity(codes: dict[str, Any]) -> pd.DataFrame:
    reps = codes["reps"]
    rows = []
    for device in DEVICES:
        for cid in CHALLENGES:
            bits = reps[(device, cid)]
            rows.append(
                {
                    "device": device,
                    "challenge": cid,
                    "uniformity": float(np.mean(bits)),
                    "n_ones": int(bits.sum()),
                    "M": int(bits.size),
                }
            )
    return pd.DataFrame(rows)


def compute_bit_aliasing(codes: dict[str, Any], ys: np.ndarray, xs: np.ndarray) -> pd.DataFrame:
    """Bit-aliasing requires shared bit definition — satisfied by common grid."""
    reps = codes["reps"]
    M = codes["M"]
    rows = []
    for cid in CHALLENGES:
        stack = np.stack([reps[(d, cid)] for d in DEVICES], axis=0)  # (15, M)
        p = stack.mean(axis=0)
        for k in range(M):
            pk = float(p[k])
            # entropy of Bernoulli(pk)
            if 0.0 < pk < 1.0:
                h = float(-(pk * np.log2(pk) + (1 - pk) * np.log2(1 - pk)))
            else:
                h = 0.0
            rows.append(
                {
                    "challenge": cid,
                    "bit_index": k,
                    "y": int(ys[k]),
                    "x": int(xs[k]),
                    "bit_aliasing": pk,
                    "abs_dev_from_half": abs(pk - 0.5),
                    "bit_entropy": h,
                }
            )
    return pd.DataFrame(rows)


def summarize_standard_metrics(
    hd: dict[str, pd.DataFrame],
    uniformity: pd.DataFrame,
    bit_aliasing: pd.DataFrame,
    *,
    n_boot: int = 5000,
) -> dict[str, Any]:
    g = hd["HD_G"]
    c = hd["HD_C"]
    d = hd["HD_D"]

    rel_mean, rel_lo, rel_hi = _bootstrap_mean(1.0 - g["hd"].to_numpy(), g["device"].to_numpy(), n_boot=n_boot)
    uniq_mean, uniq_lo, uniq_hi = _bootstrap_mean(d["hd"].to_numpy(), d["challenge"].to_numpy(), n_boot=n_boot, seed=BINARY_SEED + 1)
    diff_mean, diff_lo, diff_hi = _bootstrap_mean(c["hd"].to_numpy(), c["device"].to_numpy(), n_boot=n_boot, seed=BINARY_SEED + 2)
    uni_mean, uni_lo, uni_hi = _bootstrap_mean(
        uniformity["uniformity"].to_numpy(),
        uniformity["device"].to_numpy(),
        n_boot=n_boot,
        seed=BINARY_SEED + 3,
    )

    ba_vals = bit_aliasing["bit_aliasing"].to_numpy()
    ba_mean = float(np.mean(ba_vals))
    # Bootstrap over challenges for BA mean
    ba_point, ba_lo, ba_hi = _bootstrap_mean(
        bit_aliasing.groupby("challenge")["bit_aliasing"].mean().to_numpy(),
        bit_aliasing.groupby("challenge")["bit_aliasing"].mean().index.to_numpy(),
        n_boot=n_boot,
        seed=BINARY_SEED + 4,
    )
    ent_mean = float(bit_aliasing["bit_entropy"].mean())
    bal_frac = float(np.mean(np.abs(ba_vals - 0.5) < 0.1))

    per_device_rel = (
        g.groupby("device")["hd"].mean().map(lambda x: 1.0 - float(x)).rename("reliability").reset_index()
    )
    per_device_diff = c.groupby("device")["hd"].mean().rename("challenge_diffuseness").reset_index()
    per_device_uni = uniformity.groupby("device")["uniformity"].mean().reset_index()
    per_chal_uniq = d.groupby("challenge")["hd"].mean().rename("uniqueness").reset_index()

    metrics = {
        "Reliability": {
            "estimate": rel_mean,
            "ci95": [rel_lo, rel_hi],
            "ideal": 1.0,
            "median_HD_G": float(g["hd"].median()),
            "mean_HD_G": float(g["hd"].mean()),
            "q95_HD_G": float(g["hd"].quantile(0.95)),
        },
        "Uniqueness": {
            "estimate": uniq_mean,
            "ci95": [uniq_lo, uniq_hi],
            "ideal": 0.5,
            "median_HD_D": float(d["hd"].median()),
            "mean_HD_D": float(d["hd"].mean()),
        },
        "Challenge_diffuseness": {
            "estimate": diff_mean,
            "ci95": [diff_lo, diff_hi],
            "ideal": 0.5,
            "median_HD_C": float(c["hd"].median()),
            "mean_HD_C": float(c["hd"].mean()),
        },
        "Uniformity": {
            "estimate": uni_mean,
            "ci95": [uni_lo, uni_hi],
            "ideal": 0.5,
        },
        "Bit_aliasing": {
            "estimate": ba_point,
            "ci95": [ba_lo, ba_hi],
            "ideal": 0.5,
            "mean_abs_dev_from_half": float(np.mean(np.abs(ba_vals - 0.5))),
            "note": (
                "Computed under shared common-mask grid bit definition across 15 devices; "
                "with only 15 devices BA takes discrete values k/15."
            ),
        },
        "Mean_bit_entropy": {
            "estimate": ent_mean,
            "ideal": 1.0,
        },
        "Effective_balanced_bit_fraction": {
            "estimate": bal_frac,
            "definition": "fraction of bits with |BA-0.5|<0.1",
        },
        "pair_counts": {
            "n_HD_G": int(len(g)),
            "n_HD_C": int(len(c)),
            "n_HD_D": int(len(d)),
            "expected_HD_G": 360,
            "expected_HD_C": 420,
            "expected_HD_D": 840,
        },
    }
    return {
        "metrics": metrics,
        "per_device_reliability": per_device_rel,
        "per_device_challenge_diffuseness": per_device_diff,
        "per_device_uniformity": per_device_uni,
        "per_challenge_uniqueness": per_chal_uniq,
    }


def run_binary_puf_analysis(output_dir: Path, *, n_boot: int = 5000) -> dict[str, Any]:
    output_dir = Path(output_dir)
    common = build_common_mask(output_dir)
    corr = estimate_corr_length_px(output_dir, common)
    # Ensure grid step is at least ~2× correlation length
    step = max(GRID_STEP_PX, int(np.ceil(2.0 * corr)))
    ys, xs, m_raw, m_eff = build_shared_bit_coords(common, grid_step=step, code_length=CODE_LENGTH_M)
    # Effective independent-bit estimate ~ mask area / corr^2
    m_eff_est = int(max(1, round(float(common.sum()) / max(corr, 1.0) ** 2)))

    cfg = BinaryEncodingConfig(
        grid_step_px=step,
        code_length_M=int(ys.size),
        corr_length_estimate_px=float(corr),
        M_raw=m_raw,
        M_effective_estimate=min(m_eff_est, m_raw),
        n_common_mask_pixels=int(common.sum()),
        bit_definition_status="common_mask_shared_grid_ok",
    )
    codes = encode_all_binary_codes(output_dir, ys=ys, xs=xs, threshold=THRESHOLD)
    hd = compute_hd_tables(codes)
    uniformity = compute_uniformity(codes)
    bit_aliasing = compute_bit_aliasing(codes, ys, xs)
    summary = summarize_standard_metrics(hd, uniformity, bit_aliasing, n_boot=n_boot)

    # Manifests
    bit_def = pd.DataFrame(
        {
            "bit_index": np.arange(len(ys), dtype=int),
            "y": ys,
            "x": xs,
            "threshold": THRESHOLD,
            "grid_step_px": step,
        }
    )
    code_manifest_rows = []
    for (device, cid, b), bits in codes["blocks"].items():
        code_manifest_rows.append(
            {
                "device": device,
                "challenge": cid,
                "kind": f"block{b}",
                "n_ones": int(bits.sum()),
                "uniformity": float(bits.mean()),
                "M": int(bits.size),
            }
        )
    for (device, cid), bits in codes["reps"].items():
        code_manifest_rows.append(
            {
                "device": device,
                "challenge": cid,
                "kind": "representative",
                "n_ones": int(bits.sum()),
                "uniformity": float(bits.mean()),
                "M": int(bits.size),
            }
        )

    hd_all = pd.concat(
        [hd["HD_G"][["pair_type", "label", "hd", "device"]].assign(group=hd["HD_G"].get("challenge", "")),
         hd["HD_C"][["pair_type", "label", "hd", "device"]],
         hd["HD_D"].rename(columns={"challenge": "device"})[["pair_type", "label", "hd", "device"]]],
        ignore_index=True,
        sort=False,
    )
    # Cleaner combined HD scores table
    hd_scores = pd.concat(
        [
            hd["HD_G"].assign(score_family="HD_G"),
            hd["HD_C"].assign(score_family="HD_C"),
            hd["HD_D"].assign(score_family="HD_D"),
        ],
        ignore_index=True,
        sort=False,
    )

    return {
        "config": cfg,
        "common_mask": common,
        "bit_coords": (ys, xs),
        "codes": codes,
        "hd": hd,
        "hd_scores": hd_scores,
        "uniformity": uniformity,
        "bit_aliasing": bit_aliasing,
        "summary": summary,
        "bit_definition": bit_def,
        "codes_manifest": pd.DataFrame(code_manifest_rows),
        "hd_all_preview": hd_all,
    }


def write_binary_outputs(result: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    fd = out_dir / "figure_data"
    fd.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    cfg = result["config"]
    cfg_path = out_dir / "binary_encoding_config.json"
    cfg_path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    paths["binary_encoding_config"] = cfg_path

    metrics = result["summary"]["metrics"]
    flat_rows = []
    for name, blob in metrics.items():
        if not isinstance(blob, dict) or "estimate" not in blob:
            continue
        flat_rows.append(
            {
                "metric": name,
                "estimate": blob["estimate"],
                "ci95_lo": blob.get("ci95", [np.nan, np.nan])[0] if "ci95" in blob else np.nan,
                "ci95_hi": blob.get("ci95", [np.nan, np.nan])[1] if "ci95" in blob else np.nan,
                "ideal": blob.get("ideal", np.nan),
            }
        )
    metrics_df = pd.DataFrame(flat_rows)
    metrics_csv = out_dir / "binary_metric_summary.csv"
    metrics_json = out_dir / "binary_metric_summary.json"
    metrics_df.to_csv(metrics_csv, index=False)
    metrics_json.write_text(json.dumps(result["summary"], indent=2, default=str), encoding="utf-8")
    paths["binary_metric_summary_csv"] = metrics_csv
    paths["binary_metric_summary_json"] = metrics_json

    def _save(name: str, df: pd.DataFrame) -> None:
        p = fd / name
        df.to_csv(p, index=False)
        paths[name] = p

    _save("figure3d_binary_hd_scores.csv", result["hd_scores"])
    _save("figure3e_standard_puf_metrics.csv", metrics_df)
    _save("binary_codes_manifest.csv", result["codes_manifest"])
    _save("binary_bit_definition.csv", result["bit_definition"])
    _save("binary_per_device_metrics.csv", result["summary"]["per_device_reliability"].merge(
        result["summary"]["per_device_challenge_diffuseness"], on="device"
    ).merge(result["summary"]["per_device_uniformity"], on="device"))
    _save("binary_per_challenge_metrics.csv", result["summary"]["per_challenge_uniqueness"])

    # Bootstrap intervals table
    boot_rows = []
    for name, blob in metrics.items():
        if isinstance(blob, dict) and "ci95" in blob:
            boot_rows.append(
                {
                    "metric": name,
                    "estimate": blob["estimate"],
                    "ci95_lo": blob["ci95"][0],
                    "ci95_hi": blob["ci95"][1],
                    "ideal": blob.get("ideal", np.nan),
                }
            )
    _save("binary_bootstrap_intervals.csv", pd.DataFrame(boot_rows))
    return paths
