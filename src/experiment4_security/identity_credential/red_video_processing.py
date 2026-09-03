"""Red video → ProcessedRecording via common.video_io (read-only reuse)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiment4_security.common.video_io import ProcessedRecording, process_video


def process_red_video(
    path: Path | str,
    *,
    device_id: str,
    state_id: str,
    phase: str,
) -> ProcessedRecording:
    """Decode red capture with the same official channel pipeline as lifecycle."""
    path = Path(path)
    return process_video(
        video_path=path,
        channel="red",
        dark=None,
        device_id=device_id,
        state_id=state_id,
        round_id=f"R_{phase}",
        challenge_id="",
    )


def red_qc_flags(rec: ProcessedRecording) -> dict[str, Any]:
    return {
        "device_id": getattr(rec, "device_id", device_id_fallback(rec)),
        "state_id": getattr(rec, "state_id", ""),
        "severe_saturation": bool(getattr(rec, "severe_saturation", False)),
        "nearly_absent_signal": bool(getattr(rec, "nearly_absent_signal", False)),
        "n_frames_used": int(getattr(rec, "n_frames_used", 0) or 0),
    }


def device_id_fallback(rec: Any) -> str:
    return str(getattr(rec, "device_id", ""))
