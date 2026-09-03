"""Reporting for prepare-only and (later) formal runs — with hard gates."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiment4_security.identity_credential.credential_lifecycle import lifecycle_narrative
from experiment4_security.identity_credential.dataset_layout import layout_checklist_markdown
from experiment4_security.identity_credential.gates import (
    GateError,
    assert_prepare_text_safe,
    assert_not_writing_protected,
)
from experiment4_security.identity_credential import PROTOCOL_NAME
from experiment4_security.identity_credential.red_feature_adapter import assert_feature_contract
from experiment4_security.identity_credential.schemas import EXPECTED_INVENTORY, MODE_PREPARE


def _write(path: Path, text: str, *, project_root: Path, prepare_safe: bool = True) -> None:
    assert_not_writing_protected(path, project_root)
    if prepare_safe:
        assert_prepare_text_safe(text, context=str(path.name))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_json(path: Path, obj: Any, *, project_root: Path, prepare_safe: bool = True) -> None:
    text = json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n"
    _write(path, text, project_root=project_root, prepare_safe=prepare_safe)


def emit_prepare_artifacts(
    *,
    prepared_dir: Path,
    project_root: Path,
    cfg_resolved: dict[str, Any],
    data_audit: dict[str, Any],
    code_readiness: dict[str, Any],
    synthetic_report: dict[str, Any],
) -> list[str]:
    prepared_dir = Path(prepared_dir)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    write_json(prepared_dir / "code_readiness.json", code_readiness, project_root=project_root)
    written.append("code_readiness.json")

    write_json(prepared_dir / "synthetic_test_report.json", synthetic_report, project_root=project_root)
    written.append("synthetic_test_report.json")

    checklist = layout_checklist_markdown()
    _write(prepared_dir / "dataset_layout_checklist.md", checklist, project_root=project_root)
    written.append("dataset_layout_checklist.md")

    status_md = _data_waiting_markdown(data_audit)
    _write(prepared_dir / "DATA_WAITING_STATUS.md", status_md, project_root=project_root)
    written.append("DATA_WAITING_STATUS.md")

    # expected manifest is written by data_validation; ensure note file
    write_json(
        prepared_dir / "prepare_manifest.json",
        {
            "protocol": PROTOCOL_NAME,
            "mode": MODE_PREPARE,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "config_resolved": cfg_resolved,
            "data_status": data_audit.get("data_status"),
            "expected_inventory": EXPECTED_INVENTORY,
            "lifecycle_narrative": lifecycle_narrative(),
            "red_feature_contract": assert_feature_contract(),
            "artifacts": written,
            "forbidden_today": [
                "red AUC/EER/Top-1 numeric outputs",
                "joint identity–credential conclusions",
                "red-conditioned attack conclusions",
                "overwrite frozen F01 green pilot run or lifecycle outputs",
                "figures",
                "formal full run without DATA_READY",
            ],
        },
        project_root=project_root,
    )
    written.append("prepare_manifest.json")
    return written


def _data_waiting_markdown(audit: dict[str, Any]) -> str:
    status = audit.get("data_status", "DATA_WAITING")
    return f"""# DATA WAITING STATUS

**status:** `{status}`

## Expected inventory (frozen)

| Metric | Expected |
|--------|----------:|
| n_devices | {EXPECTED_INVENTORY['n_devices']} |
| n_states_per_device | {EXPECTED_INVENTORY['n_states_per_device']} |
| n_green_videos | {EXPECTED_INVENTORY['n_green_videos']} |
| n_red_videos | {EXPECTED_INVENTORY['n_red_videos']} |
| n_total_videos | {EXPECTED_INVENTORY['n_total_videos']} |
| n_ab_pairs | {EXPECTED_INVENTORY['n_ab_pairs']} |

## Current scan

- videos_root: `{audit.get('videos_root')}`
- n_green_found: {audit.get('n_green_found', 0)}
- n_red_found: {audit.get('n_red_found', 0)}
- n_ab_pairs: {audit.get('n_ab_pairs', 0)}
- n_missing: {audit.get('n_missing', 'n/a')}
- n_state_conflicts: {audit.get('n_state_conflicts', 0)}

## Next steps (tomorrow)

1. Place synchronized F01–F10 data under `videos/identity_credential/` using the frozen layout.
2. Run: `python -m experiment4_security.identity_credential.cli --config configs/formal_reconfiguration.yaml validate`
3. When status is `DATA_READY`, run: `python -m experiment4_security.identity_credential.cli --config configs/formal_reconfiguration.yaml formal`

Formal outputs go to: `outputs/experiment4/security/runs/<timestamp>_formal/`.
F01 green pilot remains at: `outputs/experiment4/security/runs/20260728_213919_f01_green_pilot/`.

## Forbidden

- Do not copy `outputs/experiment4/lifecycle/` red into the new tree.
- Do not invent placeholder videos for a formal run.
- Do not emit red AUC/EER/Top-1 or joint/attack conclusions until the formal run.
"""


def refuse_formal_science_in_prepare() -> None:
    raise GateError("Formal scientific reports are disabled in prepare-only mode")
