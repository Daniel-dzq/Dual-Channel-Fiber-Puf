"""Read-only bridge to ml_attack green detail_cm / enrollment / tracks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiment4_security.identity_credential.cache_adapter import make_detail_lookup, vector_dir
from experiment4_security.ml_attack.enrollment import (
    EnrollmentCommon,
    build_all_enrollment_commons,
    enrollment_templates,
    query_vectors,
)
from experiment4_security.ml_attack.metric_schema import REPRESENTATION


def build_device_commons(
    device_id: str,
    states: list[str],
    challenge_ids: list[str],
    *,
    detail_lookup: Callable[[str, str, str], np.ndarray],
) -> dict[str, EnrollmentCommon]:
    return build_all_enrollment_commons(
        states, challenge_ids, detail_lookup=detail_lookup, device_id=device_id
    )


def representation_name() -> str:
    return REPRESENTATION


def green_detail_lookup_for_run(
    run_dir: Path, device_id: str
) -> Callable[[str, str, str], np.ndarray]:
    return make_detail_lookup(vector_dir(run_dir), device_id)


__all__ = [
    "EnrollmentCommon",
    "build_device_commons",
    "enrollment_templates",
    "query_vectors",
    "representation_name",
    "green_detail_lookup_for_run",
]
