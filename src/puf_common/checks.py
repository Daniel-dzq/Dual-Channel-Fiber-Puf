"""Numerical authority checks shared by the reproduction scripts.

Every ``reproduce_*.py`` script recomputes quantities from the public dataset and
compares them with the values reported in the manuscript. Tolerance policy:

* values read back from frozen tables are compared exactly (``tol=0``);
* deterministic recomputations (quantiles, medians, counts) use ``tol=1e-9``;
* floating-point analyses that involve linear algebra or resampling use the
  tolerance stated next to the check.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from puf_common.public_data import DATA_ROOT_ENV, PublicDataset, resolve_data_root


@dataclass
class Check:
    name: str
    value: float | int | str
    expected: float | int | str
    tolerance: float
    passed: bool


def check(name: str, value, expected, tolerance: float = 1e-9) -> Check:
    if isinstance(expected, (str, list, tuple, bool)) or isinstance(value, (str, list, tuple, bool)):
        ok = value == expected
    else:
        v, e = float(value), float(expected)
        ok = math.isclose(v, e, rel_tol=0.0, abs_tol=tolerance) if math.isfinite(v) else False
    return Check(name, value, expected, tolerance, ok)


def report(checks: list[Check], out_dir: Path, title: str) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checks.json").write_text(json.dumps([asdict(c) for c in checks], indent=2, default=str), encoding="utf-8")
    width = max(len(c.name) for c in checks)
    print(f"\n{title}")
    for c in checks:
        print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name.ljust(width)}  value={c.value!r}  expected={c.expected!r}")
    ok = all(c.passed for c in checks)
    print(f"  -> {sum(c.passed for c in checks)}/{len(checks)} checks passed; outputs in {out_dir}\n")
    return ok


def reproduction_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data-root", default=None, help=f"Zenodo dataset directory (or set ${DATA_ROOT_ENV})")
    p.add_argument("--output-root", default=os.environ.get("PUF_OUTPUT_ROOT", "outputs"), help="Where results are written (default: ./outputs)")
    p.add_argument("--no-figures", action="store_true", help="Skip matplotlib figure rendering")
    return p


def open_dataset(args: argparse.Namespace, figure_slug: str) -> tuple[PublicDataset, Path]:
    ds = PublicDataset(resolve_data_root(args.data_root))
    out = Path(args.output_root).expanduser().resolve() / figure_slug
    out.mkdir(parents=True, exist_ok=True)
    return ds, out
