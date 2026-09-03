"""Timestamped run registry: local TZ, RUN_STATUS, atomic latest_* updates."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

# Prefer Asia/Singapore; fall back to system local if zoneinfo unavailable.
DEFAULT_TZ_NAME = os.environ.get("EXPERIMENT4_TZ", "Asia/Singapore")


def get_tz(name: str | None = None) -> ZoneInfo:
    return ZoneInfo(name or DEFAULT_TZ_NAME)


def now_local(tz_name: str | None = None) -> datetime:
    return datetime.now(get_tz(tz_name))


def format_run_stamp(dt: datetime | None = None, *, tz_name: str | None = None) -> str:
    dt = dt or now_local(tz_name)
    return dt.strftime("%Y%m%d_%H%M%S")


def new_run_id(kind: str, *, tz_name: str | None = None) -> str:
    """Local-timezone run id: YYYYMMDD_HHMMSS_<kind>."""
    kind = kind.strip().replace(" ", "_")
    return f"{format_run_stamp(tz_name=tz_name)}_{kind}"


def isoformat_local(dt: datetime | None = None, *, tz_name: str | None = None) -> str:
    dt = dt or now_local(tz_name)
    return dt.isoformat(timespec="seconds")


RUN_KINDS = ("prepare", "validate", "f01_green_pilot", "formal")

STATUS_RUNNING = "RUNNING"
STATUS_SUCCESS = "SUCCESS"
STATUS_FAILED = "FAILED"

LATEST_FILES = {
    "prepare": "latest_prepare.json",
    "validate": "latest_validate.json",
    "formal": "latest_formal.json",
    "f01_green_pilot": "latest_green_pilot.json",
}


REQUIRED_RUN_FILES = (
    "run_manifest.json",
    "config_resolved.yaml",
    "data_manifest.csv",
    "data_audit.json",
    "execution.log",
    "metrics_summary.json",
    "code_diff_summary.txt",
    "environment.json",
    "RUN_STATUS.json",
)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True, default=str)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def sha256_file(path: Path | None) -> str | None:
    if path is None or not Path(path).exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_commit(project_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def code_diff_summary(project_root: Path) -> str:
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        head = git_commit(project_root) or "UNKNOWN"
        lines = [f"git_head: {head}", ""]
        if not dirty.strip():
            lines.append("working_tree: clean")
        else:
            lines.append("working_tree: dirty")
            lines.extend(dirty.strip().splitlines()[:200])
        return "\n".join(lines) + "\n"
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        return f"git unavailable: {exc}\n"


def environment_payload() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "executable": sys.executable,
        "tz": DEFAULT_TZ_NAME,
    }


@dataclass
class RunContext:
    output_root: Path
    run_id: str
    run_kind: str
    run_dir: Path
    tz_name: str = DEFAULT_TZ_NAME
    formal_scientific_run: bool = False
    parent_run_id: str | None = None
    cache_namespace: str = "security_v1"
    started_at: str = ""
    project_root: Path | None = None
    log_path: Path | None = None
    _log_fh: Any = field(default=None, repr=False)

    def relative_path(self) -> str:
        return f"runs/{self.run_id}"

    def write_json(self, name: str, payload: dict[str, Any] | list[Any]) -> Path:
        path = self.run_dir / name
        _atomic_write_json(path, payload if isinstance(payload, dict) else {"items": payload})
        return path

    def write_text(self, name: str, text: str) -> Path:
        path = self.run_dir / name
        path.write_text(text)
        return path

    def log(self, msg: str) -> None:
        line = f"{isoformat_local(tz_name=self.tz_name)} {msg}\n"
        if self._log_fh is None and self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(self.log_path, "a", encoding="utf-8")
        if self._log_fh is not None:
            self._log_fh.write(line)
            self._log_fh.flush()

    def close_log(self) -> None:
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None

    def base_manifest(self, **extra: Any) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_kind": self.run_kind,
            "started_at": self.started_at,
            "finished_at": None,
            "timezone": self.tz_name,
            "status": STATUS_RUNNING,
            "git_commit": git_commit(self.project_root) if self.project_root else None,
            "config_hash": extra.pop("config_hash", None),
            "dataset_manifest_hash": extra.pop("dataset_manifest_hash", None),
            "valid_mask_hash": extra.pop("valid_mask_hash", None),
            "parent_run_id": self.parent_run_id,
            "cache_namespace": self.cache_namespace,
            "formal_scientific_run": self.formal_scientific_run,
            "relative_path": self.relative_path(),
            **extra,
        }

    def write_status(self, status: str, **extra: Any) -> None:
        payload = {
            "run_id": self.run_id,
            "run_kind": self.run_kind,
            "status": status,
            "updated_at": isoformat_local(tz_name=self.tz_name),
            "timezone": self.tz_name,
            **extra,
        }
        _atomic_write_json(self.run_dir / "RUN_STATUS.json", payload)

    def finalize_success(self, manifest: dict[str, Any]) -> None:
        finished = isoformat_local(tz_name=self.tz_name)
        manifest = dict(manifest)
        manifest["status"] = STATUS_SUCCESS
        manifest["finished_at"] = finished
        _atomic_write_json(self.run_dir / "run_manifest.json", manifest)
        self.write_status(STATUS_SUCCESS, finished_at=finished)
        self.close_log()
        update_latest_pointers(
            self.output_root,
            run_id=self.run_id,
            run_kind=self.run_kind,
            status=STATUS_SUCCESS,
            relative_path=self.relative_path(),
            tz_name=self.tz_name,
        )

    def finalize_failed(self, manifest: dict[str, Any], error: str) -> None:
        finished = isoformat_local(tz_name=self.tz_name)
        manifest = dict(manifest)
        manifest["status"] = STATUS_FAILED
        manifest["finished_at"] = finished
        manifest["error"] = error
        _atomic_write_json(self.run_dir / "run_manifest.json", manifest)
        self.write_status(STATUS_FAILED, finished_at=finished, error=error)
        self.close_log()
        # Do NOT update latest_* on failure


def reopen_run(
    output_root: Path,
    run_id: str,
    *,
    project_root: Path | None = None,
    parent_run_id: str | None = None,
    formal_scientific_run: bool = False,
    cache_namespace: str = "security_v1",
    tz_name: str | None = None,
) -> RunContext:
    """Re-open an existing run directory (e.g. FAILED formal) and mark RUNNING again."""
    output_root = Path(output_root)
    run_dir = output_root / "runs" / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Cannot reopen missing run: {run_dir}")
    kind = run_id.rsplit("_", 1)[-1]
    # run_id is YYYYMMDD_HHMMSS_<kind> — kind may contain underscores (f01_green_pilot)
    for candidate in RUN_KINDS:
        if run_id.endswith("_" + candidate):
            kind = candidate
            break
    tz = tz_name or DEFAULT_TZ_NAME
    started = now_local(tz)
    ctx = RunContext(
        output_root=output_root,
        run_id=run_id,
        run_kind=kind,
        run_dir=run_dir,
        tz_name=tz,
        formal_scientific_run=formal_scientific_run,
        parent_run_id=parent_run_id,
        cache_namespace=cache_namespace,
        started_at=isoformat_local(started, tz_name=tz),
        project_root=project_root,
        log_path=run_dir / "execution.log",
    )
    ctx.write_status(STATUS_RUNNING)
    ctx.log(f"reopened run {run_id} (resume)")
    return ctx


def create_run(
    output_root: Path,
    kind: str,
    *,
    project_root: Path | None = None,
    parent_run_id: str | None = None,
    formal_scientific_run: bool = False,
    cache_namespace: str = "security_v1",
    tz_name: str | None = None,
) -> RunContext:
    if kind not in RUN_KINDS:
        raise ValueError(f"Unknown run kind {kind!r}; expected one of {RUN_KINDS}")
    tz = tz_name or DEFAULT_TZ_NAME
    started = now_local(tz)
    base = f"{format_run_stamp(started, tz_name=tz)}_{kind}"
    run_id = base
    runs_root = Path(output_root) / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    # Avoid same-second collisions
    n = 2
    while (runs_root / run_id).exists():
        run_id = f"{base}_{n}"
        n += 1
        if n > 99:
            raise RuntimeError(f"Too many colliding run ids for {base}")
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    ctx = RunContext(
        output_root=Path(output_root),
        run_id=run_id,
        run_kind=kind,
        run_dir=run_dir,
        tz_name=tz,
        formal_scientific_run=formal_scientific_run,
        parent_run_id=parent_run_id,
        cache_namespace=cache_namespace,
        started_at=isoformat_local(started, tz_name=tz),
        project_root=project_root,
        log_path=run_dir / "execution.log",
    )
    # Seed required stubs so the directory is always self-describing mid-run
    ctx.write_text("execution.log", "")
    ctx.write_text("code_diff_summary.txt", code_diff_summary(project_root) if project_root else "n/a\n")
    ctx.write_json("environment.json", environment_payload())
    ctx.write_json(
        "metrics_summary.json",
        {
            "formal_scientific_run": formal_scientific_run,
            "metrics_emitted": False,
            "note": "Filled only when this run kind produces metrics.",
        },
    )
    ctx.write_text("data_manifest.csv", "sample_id,channel,device_id,state_id,status\n")
    ctx.write_json("data_audit.json", {"status": "PENDING"})
    ctx.write_text("config_resolved.yaml", "{}\n")
    man = ctx.base_manifest()
    _atomic_write_json(run_dir / "run_manifest.json", man)
    ctx.write_status(STATUS_RUNNING)
    ctx.log(f"created run {run_id}")
    return ctx


def update_latest_pointers(
    output_root: Path,
    *,
    run_id: str,
    run_kind: str,
    status: str,
    relative_path: str,
    tz_name: str | None = None,
) -> None:
    """Atomic update of latest_<kind>.json and latest_run.json — SUCCESS only."""
    if status != STATUS_SUCCESS:
        return
    output_root = Path(output_root)
    updated_at = isoformat_local(tz_name=tz_name)
    payload = {
        "run_id": run_id,
        "run_kind": run_kind,
        "status": status,
        "relative_path": relative_path,
        "updated_at": updated_at,
        "timezone": tz_name or DEFAULT_TZ_NAME,
    }
    kind_file = LATEST_FILES.get(run_kind)
    if kind_file:
        _atomic_write_json(output_root / kind_file, payload)
    _atomic_write_json(output_root / "latest_run.json", payload)


def read_latest(output_root: Path, kind: str) -> dict[str, Any] | None:
    fname = LATEST_FILES.get(kind)
    if not fname:
        return None
    path = Path(output_root) / fname
    if not path.exists():
        return None
    return json.loads(path.read_text())


def assert_validate_ready_for_formal(output_root: Path, *, strict: bool = True) -> dict[str, Any]:
    """Load latest SUCCESS validate run and assert DATA_READY inventory."""
    from experiment4_security.identity_credential.schemas import (
        N_AB_PAIRS_EXPECTED,
        N_GREEN_VIDEOS_EXPECTED,
        N_RED_VIDEOS_EXPECTED,
        STATUS_DATA_READY,
    )

    latest = read_latest(output_root, "validate")
    if latest is None:
        raise RuntimeError("No latest_validate.json — run validate --strict first")
    if latest.get("status") != STATUS_SUCCESS:
        raise RuntimeError(f"latest validate status is {latest.get('status')!r}, need SUCCESS")
    run_dir = Path(output_root) / latest["relative_path"]
    audit_path = run_dir / "data_audit.json"
    if not audit_path.exists():
        raise RuntimeError(f"Missing data_audit.json in {run_dir}")
    audit = json.loads(audit_path.read_text())
    if audit.get("data_status") != STATUS_DATA_READY:
        raise RuntimeError(f"validate data_status={audit.get('data_status')!r}, need DATA_READY")
    if strict:
        from experiment4_security.identity_credential.schemas import N_DEVICES

        checks = {
            "n_devices": audit.get("expected_inventory", {}).get("n_devices") == N_DEVICES
            and len(audit.get("devices_present", [])) >= N_DEVICES,
            "n_green_videos": audit.get("n_green_found") == N_GREEN_VIDEOS_EXPECTED,
            "n_red_videos": audit.get("n_red_found") == N_RED_VIDEOS_EXPECTED,
            "n_ab_pairs": audit.get("n_ab_pairs") == N_AB_PAIRS_EXPECTED,
            "n_missing_required": audit.get("n_missing", 1) == 0,
            "n_conflicts": audit.get("n_state_conflicts", 1) == 0,
        }
        if not all(checks.values()):
            raise RuntimeError(f"strict validate inventory failed: {checks}")
    status_path = run_dir / "RUN_STATUS.json"
    if status_path.exists():
        st = json.loads(status_path.read_text())
        if st.get("status") != STATUS_SUCCESS:
            raise RuntimeError("validate RUN_STATUS is not SUCCESS")
    return {"latest": latest, "audit": audit, "run_dir": str(run_dir)}
