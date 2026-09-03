"""Experiment 4 multi-device identity–credential formal pipeline.

Scientific story (frozen):
  Persistent physical identity (red)
  + State-bound registered credential (green)
  + Credential revocation and re-enrollment
  + Enrollment-database disclosure attacks (exact replay is digital replay, not a physical clone)
  + Red-to-green side-information leakage audit

Hard constraints (today / always):
  - Do not modify lifecycle/ source or outputs
  - Do not overwrite frozen F01 green pilot run under security/runs/
  - Do not fabricate red results or splice old lifecycle red into new captures
  - prepare-only mode must not emit formal red/joint/attack conclusions or metrics
  - Formal full run requires DATA_READY after dataset validation
  - Outputs use timestamped security/runs/<YYYYMMDD_HHMMSS>_<kind>/
"""

from __future__ import annotations

__all__ = [
    "PACKAGE_NAME",
    "PROTOCOL_NAME",
]

PACKAGE_NAME = "identity_credential"
PROTOCOL_NAME = "identity_credential_decoupling_v1"
