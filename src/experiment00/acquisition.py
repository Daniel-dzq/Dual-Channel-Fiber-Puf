"""Challenge acquisition sequence: fiber_id -> sequence_id -> acquisition_position.

Chronology is NEVER inferred from directory listing, glob order, or file mtimes.
Primary A/B pairing always joins on challenge_id (C01-C08).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

ORDER_SOURCE = "fiber_sequence_mapping"

DEFAULT_SEQUENCE_ASSIGNMENT = {
    "F01": "P1",
    "F02": "P2",
    "F03": "P3",
    "F04": "P4",
    "F05": "P5",
}

DEFAULT_CHALLENGE_SEQUENCES = {
    "P1": ["C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08"],
    "P2": ["C03", "C04", "C05", "C06", "C07", "C08", "C01", "C02"],
    "P3": ["C05", "C06", "C07", "C08", "C01", "C02", "C03", "C04"],
    "P4": ["C07", "C08", "C01", "C02", "C03", "C04", "C05", "C06"],
    "P5": ["C02", "C04", "C06", "C08", "C01", "C03", "C05", "C07"],
}

EXPECTED_CHALLENGES = [f"C{i:02d}" for i in range(1, 9)]


@dataclass(frozen=True)
class AcquisitionMeta:
    challenge_id: str | None
    sequence_id: str | None
    acquisition_position: int | None  # 1-8
    acquisition_position_zero_based: int | None  # 0-7
    order_source: str | None


class AcquisitionConfigError(ValueError):
    pass


def validate_challenge_sequences(
    sequences: Mapping[str, list[str]],
    *,
    expected_challenges: list[str] | None = None,
) -> None:
    expected = list(expected_challenges or EXPECTED_CHALLENGES)
    for sid, seq in sequences.items():
        if len(seq) != 8:
            raise AcquisitionConfigError(f"{sid}: expected 8 challenges, got {len(seq)}")
        if len(set(seq)) != 8:
            raise AcquisitionConfigError(f"{sid}: duplicate challenges in {seq}")
        missing = sorted(set(expected) - set(seq))
        extra = sorted(set(seq) - set(expected))
        if missing or extra:
            raise AcquisitionConfigError(
                f"{sid}: must contain C01-C08 exactly once; missing={missing} extra={extra}"
            )


def sequence_id_for_fiber(
    fiber_id: str,
    assignment: Mapping[str, str] | None = None,
) -> str:
    mapping = assignment or DEFAULT_SEQUENCE_ASSIGNMENT
    fiber = fiber_id.upper()
    if fiber not in mapping:
        raise AcquisitionConfigError(f"No sequence assignment for fiber {fiber}")
    return mapping[fiber]


def acquisition_position_for_challenge(
    sequence_id: str,
    challenge_id: str,
    sequences: Mapping[str, list[str]] | None = None,
) -> int:
    """Return 1-based chronological position of challenge_id within sequence_id."""
    seqs = sequences or DEFAULT_CHALLENGE_SEQUENCES
    if sequence_id not in seqs:
        raise AcquisitionConfigError(f"Unknown sequence_id {sequence_id}")
    seq = seqs[sequence_id]
    ch = challenge_id.upper()
    try:
        return seq.index(ch) + 1
    except ValueError as exc:
        raise AcquisitionConfigError(
            f"challenge_id {ch} not found in {sequence_id}: {seq}"
        ) from exc


def resolve_acquisition_meta(
    *,
    fiber_id: str | None,
    challenge_id: str | None,
    illumination: str | None,
    assignment: Mapping[str, str] | None = None,
    sequences: Mapping[str, list[str]] | None = None,
) -> AcquisitionMeta:
    """Resolve sequence/position from fiber_id + challenge_id. Red -> nulls."""
    if illumination == "red" or challenge_id is None or fiber_id is None:
        return AcquisitionMeta(
            challenge_id=None,
            sequence_id=None,
            acquisition_position=None,
            acquisition_position_zero_based=None,
            order_source=None,
        )
    sid = sequence_id_for_fiber(fiber_id, assignment)
    pos = acquisition_position_for_challenge(sid, challenge_id, sequences)
    return AcquisitionMeta(
        challenge_id=challenge_id.upper() if challenge_id.startswith("C") else challenge_id,
        sequence_id=sid,
        acquisition_position=pos,
        acquisition_position_zero_based=pos - 1,
        order_source=ORDER_SOURCE,
    )


def enrich_row_with_acquisition(
    row: dict,
    *,
    assignment: Mapping[str, str] | None = None,
    sequences: Mapping[str, list[str]] | None = None,
) -> dict:
    """Add acquisition columns to an inventory/QC row dict."""
    illum = row.get("illumination")
    fiber = row.get("fiber_id")
    # Prefer explicit challenge_id; fall back to legacy "challenge"
    ch = row.get("challenge_id")
    if ch is None:
        ch = row.get("challenge")
    meta = resolve_acquisition_meta(
        fiber_id=fiber,
        challenge_id=ch,
        illumination=illum,
        assignment=assignment,
        sequences=sequences,
    )
    out = dict(row)
    out["challenge_id"] = meta.challenge_id if meta.challenge_id is not None else ch
    if illum == "red":
        out["challenge_id"] = None
    out["sequence_id"] = meta.sequence_id
    out["acquisition_position"] = meta.acquisition_position
    out["acquisition_position_zero_based"] = meta.acquisition_position_zero_based
    out["order_source"] = meta.order_source if meta.order_source else (
        ORDER_SOURCE if illum == "green" and ch else None
    )
    if illum == "red":
        out["order_source"] = None
    return out
