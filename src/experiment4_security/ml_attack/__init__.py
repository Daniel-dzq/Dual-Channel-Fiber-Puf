"""Registered-database green-credential security analysis for Experiment 4.

Protocol:
- Fixed challenge bank C001–C128
- Round A = enrollment / registered database
- Round B = independent query / authentication (paired by challenge_id)
- Tracks: A database auth, B template revocation, C response reconstruction, D frozen-source transfer
- Representation: fullres detail_cm with state-specific Round-A enrollment common

Hard constraints:
- Never write into outputs/experiment4/lifecycle/
- Never recompute tau_R / tau_G / k_of_8
- absolute_asr remains null (score-space / protocol mismatch)
- No figures in this pilot
- No unseen-challenge generalization / sequential / LOSO unknown-state tracks
"""

from __future__ import annotations
