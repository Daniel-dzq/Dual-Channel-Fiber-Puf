"""Challenge-bank-level bootstrap confidence intervals.

Only F01 is available in this pilot, so bootstrap resampling can only
characterize *within-device* challenge-sampling uncertainty. It must never
be reported or interpreted as a device-population confidence interval.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

BOOTSTRAP_SCOPE_NOTE = (
    "This bootstrap resamples challenge banks within a single device (F01) "
    "only. It quantifies challenge-sampling uncertainty for F01 and cannot "
    "substitute for device-to-device (population) uncertainty, which would "
    "require multiple devices."
)


def bootstrap_ci_by_bank(
    values_by_challenge: dict[str, float],
    bank_of_challenge: dict[str, str],
    *,
    n_resamples: int,
    seed: int,
    stat_fn: Callable[[np.ndarray], float] = np.mean,
    ci: float = 0.95,
) -> dict[str, float]:
    banks: dict[str, list[float]] = {}
    for cid, val in values_by_challenge.items():
        if cid not in bank_of_challenge or val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        banks.setdefault(bank_of_challenge[cid], []).append(float(val))
    bank_ids = sorted(banks.keys())
    if not bank_ids:
        return {"point_estimate": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"), "n_resamples": 0, "n_banks": 0}

    rng = np.random.default_rng(seed)
    all_vals = np.concatenate([np.asarray(banks[b]) for b in bank_ids])
    point_estimate = float(stat_fn(all_vals))

    boot_stats = np.empty(n_resamples, dtype=np.float64)
    n_banks = len(bank_ids)
    for i in range(n_resamples):
        sampled_banks = rng.integers(0, n_banks, size=n_banks)
        pooled = np.concatenate([np.asarray(banks[bank_ids[j]]) for j in sampled_banks])
        boot_stats[i] = stat_fn(pooled)

    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(boot_stats, 100 * alpha))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha)))
    return {
        "point_estimate": point_estimate,
        "ci_lower": lo,
        "ci_upper": hi,
        "n_resamples": n_resamples,
        "n_banks": n_banks,
        "scope_note": BOOTSTRAP_SCOPE_NOTE,
    }
