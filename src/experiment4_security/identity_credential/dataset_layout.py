"""Frozen on-disk dataset layout for synchronized red+green multi-device captures."""

from __future__ import annotations

from pathlib import Path

from experiment4_security.identity_credential.schemas import (
    CHALLENGE_ID_END,
    CHALLENGE_ID_START,
    DEVICES,
    STATES,
    challenge_ids,
)

# Relative to the repository root
DEFAULT_VIDEOS_REL = Path("videos/identity_credential")
DEFAULT_OUTPUT_REL = Path("outputs/experiment4/security")
PREPARED_DIRNAME = "prepared"
CURRENT_DIRNAME = "current"

LAYOUT_VERSION = "identity_credential_layout_v1"


def device_dir(root: Path, device_id: str) -> Path:
    return Path(root) / device_id


def state_dir(root: Path, device_id: str, state_id: str) -> Path:
    return device_dir(root, device_id) / state_id


def red_dir(root: Path, device_id: str, state_id: str) -> Path:
    return state_dir(root, device_id, state_id) / "red"


def green_round_dir(root: Path, device_id: str, state_id: str, round_id: str) -> Path:
    return state_dir(root, device_id, state_id) / "green" / round_id


def red_filename(phase: str, state_id: str, device_id: str) -> str:
    if phase not in {"before", "after"}:
        raise ValueError(f"red phase must be before|after, got {phase!r}")
    return f"R_{phase}_{state_id}_{device_id}.mp4"


def green_filename(round_id: str, challenge_index: int, challenge_id: str, state_id: str, device_id: str) -> str:
    """challenge_index is 1-based presentation index (may differ from cid number)."""
    cid_num = int(challenge_id.replace("C", ""))
    return f"{round_id}_{challenge_index}_C{cid_num:03d}_{state_id}_{device_id}.mp4"


def expected_red_path(root: Path, device_id: str, state_id: str, phase: str) -> Path:
    return red_dir(root, device_id, state_id) / red_filename(phase, state_id, device_id)


def expected_green_path(
    root: Path,
    device_id: str,
    state_id: str,
    round_id: str,
    challenge_id: str,
    *,
    challenge_index: int | None = None,
) -> Path:
    cid_num = int(challenge_id.replace("C", ""))
    idx = challenge_index if challenge_index is not None else cid_num
    return green_round_dir(root, device_id, state_id, round_id) / green_filename(
        round_id, idx, challenge_id, state_id, device_id
    )


def iter_expected_paths(root: Path) -> list[dict[str, str]]:
    """Return expected relative path records for the full 10-device inventory."""
    root = Path(root)
    rows: list[dict[str, str]] = []
    for device in DEVICES:
        for state in STATES:
            for phase in ("before", "after"):
                p = expected_red_path(root, device, state, phase)
                rows.append(
                    {
                        "channel": "red",
                        "device_id": device,
                        "state_id": state,
                        "round_id": "",
                        "phase": phase,
                        "challenge_id": "",
                        "relpath": str(p.relative_to(root)),
                        "filename": p.name,
                    }
                )
            for round_id in ("A", "B"):
                for i, cid in enumerate(challenge_ids(), start=1):
                    p = expected_green_path(root, device, state, round_id, cid, challenge_index=i)
                    rows.append(
                        {
                            "channel": "green",
                            "device_id": device,
                            "state_id": state,
                            "round_id": round_id,
                            "phase": "",
                            "challenge_id": cid,
                            "relpath": str(p.relative_to(root)),
                            "filename": p.name,
                        }
                    )
    return rows


def layout_checklist_markdown() -> str:
    cids = f"C{CHALLENGE_ID_START:03d}–C{CHALLENGE_ID_END:03d}"
    return f"""# Identity–credential dataset layout checklist

**Layout version:** `{LAYOUT_VERSION}`

## Root

```
videos/identity_credential/
├── F01/ … F10/
│   ├── M0/ … M7/
│   │   ├── red/
│   │   │   ├── R_before_M0_F01.mp4
│   │   │   └── R_after_M0_F01.mp4
│   │   └── green/
│   │       ├── A/   # Round A enrollment  ({cids})
│   │       └── B/   # Round B query       ({cids})
```

## Filename rules (frozen)

| Channel | Pattern | Example |
|---------|---------|---------|
| Red before/after | `R_{{before\\|after}}_{{state}}_{{device}}.mp4` | `R_before_M0_F01.mp4` |
| Green | `{{A\\|B}}_{{index}}_C{{cid:03d}}_{{state}}_{{device}}.mp4` | `A_1_C001_M0_F01.mp4` |

- `device` ∈ {list(DEVICES)}
- `state` ∈ {list(STATES)}
- Folder `state` is authoritative if filename tokens conflict.
- Green Round A/B must form 128 pairs per device×state by `challenge_id`.
- Red standardizer development: F01–F05; held-out: F06–F10 (green tracks still use all devices).

## Expected counts

| Metric | Value |
|--------|------:|
| Devices | 10 |
| States / device | 8 |
| Green videos | 20480 |
| Red videos | 160 |
| Total videos | 20640 |
| A/B pairs | 10240 |

## Forbidden

- Do not place data under `videos/attack _videos/` for this protocol.
- Do not splice `outputs/experiment4/lifecycle/` red into this tree.
- Do not invent placeholder `.mp4` files for formal runs.
"""
