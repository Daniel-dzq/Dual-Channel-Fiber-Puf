"""Reports using the unified S_G/S_C/S_X/S_A and RG_C/RG_X/RG_A vocabulary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from experiment4_security.ml_attack.config import MLAttackConfig
from experiment4_security.ml_attack.metric_schema import FORBIDDEN_METRIC_TOKENS, NAME_MAPPING

ALLOWED_CONCLUSIONS = (
    "DATABASE_AUTHENTICATION_VALID",
    "DATABASE_AUTHENTICATION_PARTIAL",
    "DATABASE_AUTHENTICATION_FAILED",
    "STATE_BOUND_CREDENTIALS_CONFIRMED",
    "OLD_TEMPLATE_DATABASE_REVOKED",
    "PARTIAL_CROSS_STATE_CREDENTIAL_TRANSFER",
    "CREDENTIAL_RECONFIGURATION_NOT_SECURE",
    "SOFTWARE_CLONE_VALID_SAME_STATE_ONLY",
    "MODEL_NOT_VALID_IN_SOURCE_STATE",
    "PARTIAL_CROSS_STATE_CLONE_TRANSFER",
    "SOFTWARE_CLONE_RECONFIGURATION_RISK",
    "INCONCLUSIVE_DATA_OR_QC",
)

FORBIDDEN_PHRASES = (
    "machine-learning resistant",
    "provably secure",
    "immune to modeling attack",
    "fully secure",
    "NO_SAME_STATE_SIGNAL",
    "MULTI_STATE_CHARACTERIZATION_EMERGES",
    "unseen challenge",
    "ASR@tau_G",
    "prediction_ncc",
    "transfer_ncc",
    "RG_revoke",
    "RG_model_same",
    "RG_model_revoke",
)


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def check_no_forbidden_claims(text: str) -> list[str]:
    hits = []
    lower = text.lower()
    for phrase in list(FORBIDDEN_PHRASES) + list(FORBIDDEN_METRIC_TOKENS):
        if phrase.lower() in lower:
            hits.append(phrase)
    return hits


def write_metric_unification_audit(run_dir: Path) -> None:
    text = """# Metric Unification Audit

## Notation

Green challenge-specific response is written $\\widetilde{\\mathbf G}$.
In the score $S_G$, the subscript **G means Genuine match, not Green**.

Representation for this pilot: `fullres_detail_cm_enrollment_frozen`
(= detail − Round-A enrollment common of the source state).

All robust gaps use one formula:

$$RG = Q_{05}(\\mathcal S_G) - Q_{95}(\\text{non-genuine})$$

## Cross-experiment correspondence

| Experiment | Scores | Robust gaps |
|---|---|---|
| Exp1 macro-pixel screening | $\\mathcal S_G$, $\\mathcal S_C$ | $RG_C$ |
| Exp0 length optimization | $\\mathcal S_G$, $\\mathcal S_C$, $\\mathcal S_D$ | $RG_C$, $RG_D$, $RG_{min}$ |
| Fixed-state dual-channel | $\\mathcal S_G$, $\\mathcal S_C$, $\\mathcal S_D$ | $RG_C$, $RG_D$ |
| Lifecycle | green same-state match ↔ $\\mathcal S_G$; old-credential fail ↔ $\\mathcal S_X$; red $q_R$ independent | lifecycle thresholds frozen; **not** remapped numerically here |
| Exp4 Track A | $\\mathcal S_G$, $\\mathcal S_C$ | $RG_C$ |
| Exp4 Track B | $\\mathcal S_G$, $\\mathcal S_X$ | $RG_X$ |
| Exp4 Track C/D | $\\mathcal S_G$, $\\mathcal S_A^{(m)}$ | $RG_A$ |

## F01 single-device

- `device_mismatch_status = NOT_AVAILABLE_SINGLE_DEVICE`
- `rg_device = null`
- Do not forge $S_D$ from mechanical-state mismatch.

## Red remains separate

- `q_R`, `auc_red_identity`, `eer_red_identity`, `red_top1_identity`
- Never named $S_G$ / $S_X$ / $S_A$.

## Threshold protocol ≠ symbol unification

Lifecycle: 8 challenges, groupwise detail_cm, frozen $\\tau_G$.  
F01 registered DB: 128 challenges, Round-A enrollment common.  
Therefore `absolute_asr = null`, `asr_status = SCORE_SPACE_OR_PROTOCOL_MISMATCH`.

## Name mapping

See `metric_name_mapping.csv` and `METRIC_DEFINITIONS.json`.
"""
    (run_dir / "METRIC_UNIFICATION_AUDIT.md").write_text(text, encoding="utf-8")


def write_all_reports(
    run_dir: Path,
    *,
    cfg: MLAttackConfig,
    data_audit: dict[str, Any],
    ta_sum: pd.DataFrame,
    tb_sum: pd.DataFrame,
    tb_meta: dict[str, Any],
    tc_sum: pd.DataFrame,
    td_sum: pd.DataFrame,
    td_meta: dict[str, Any],
    cache_validation: dict[str, Any],
    correction: dict[str, Any],
) -> list[str]:
    run_dir = Path(run_dir)
    write_metric_unification_audit(run_dir)

    levels: list[str] = []
    if data_audit.get("data_status") == "DATA_BLOCKED":
        levels.append("INCONCLUSIVE_DATA_OR_QC")
    else:
        if (ta_sum["state_conclusion"] == "DATABASE_AUTHENTICATION_VALID").all():
            levels.append("DATABASE_AUTHENTICATION_VALID")
        elif (ta_sum["state_conclusion"] == "DATABASE_AUTHENTICATION_FAILED").all():
            levels.append("DATABASE_AUTHENTICATION_FAILED")
        else:
            levels.append("DATABASE_AUTHENTICATION_PARTIAL")
        levels.append(tb_meta.get("conclusion", "INCONCLUSIVE_DATA_OR_QC"))
        for c in td_meta.get("per_model_conclusions", []):
            levels.append(c.get("conclusion", "MODEL_NOT_VALID_IN_SOURCE_STATE"))

    ta_cols = [
        c
        for c in [
            "state_id",
            "median_S_G",
            "q05_S_G",
            "median_S_C",
            "q95_S_C",
            "rg_challenge",
            "auc_challenge",
            "eer_challenge",
            "top1",
            "top5",
            "median_rank",
            "mean_reciprocal_rank",
            "state_conclusion",
        ]
        if c in ta_sum.columns
    ]

    audit_lines = [
        "# Green Credential Attack Audit",
        "",
        "## Unified score and metric framework",
        "",
        "Subscript **G in $S_G$ means Genuine**, not Green. Green responses are $\\widetilde{\\mathbf G}$.",
        "",
        "- $S_G$: Genuine match (same device/state/challenge, Round A vs B)",
        "- $S_C$: Challenge mismatch (same state, different challenge)",
        "- $S_D$: Device mismatch — **NOT_AVAILABLE_SINGLE_DEVICE** on F01",
        "- $S_X$: Cross-state old-credential match (old DB of $s$ on target $t$)",
        "- $S_A^{(m)}$: Software-clone attack match for method $m$",
        "- $RG_C, RG_X, RG_A = Q_{05}(\\mathcal S_G)-Q_{95}(\\text{non-genuine})$",
        "- $AUC_C/EER_C$, $AUC_X/EER_X$, $AUC_A/EER_A$ with positive class = GENUINE",
        "",
        f"Representation: `fullres_detail_cm_enrollment_frozen`",
        "",
        "## Data",
        "",
        f"- data_status: `{data_audit.get('data_status')}`",
        f"- n_videos: {data_audit.get('n_videos_found')}",
        f"- n_state_id_conflicts_raw/active: {data_audit.get('n_state_id_conflicts_raw')}/{data_audit.get('n_state_id_conflicts_active')}",
        f"- n_ab_pairs_ok: {data_audit.get('n_ab_pairs_ok')}",
        f"- cache_validation: `{cache_validation.get('status')}`",
        "",
        correction.get("finding", ""),
        "",
        "## Track A — $S_G$ vs $S_C$",
        "",
        "Genuine match is separated from Challenge mismatch when $RG_C>0$ and retrieval is high.",
        "",
        ta_sum[ta_cols].to_string(index=False),
        "",
        "## Track B — $S_G$ vs $S_X$",
        "",
        f"- diagonal median $S_X$: {tb_meta.get('diagonal_median')}",
        f"- off-diagonal median $S_X$: {tb_meta.get('off_diagonal_median')}",
        f"- minimum $RG_X$: {tb_meta.get('minimum_rg_cross_state_credential')}",
        f"- worst pair: {tb_meta.get('source_state_worst_target')}",
        f"- conclusion: `{tb_meta.get('conclusion')}`",
        "",
        "## Track C — $S_G$ vs $S_A$ (same state)",
        "",
        tc_sum[
            [
                c
                for c in [
                    "source_state",
                    "attack_method",
                    "median_S_A",
                    "rg_software_clone",
                    "auc_software_clone",
                    "eer_software_clone",
                    "top1",
                    "residual_ncc",
                    "source_validity_note",
                    "common_dominance_flag",
                ]
                if c in tc_sum.columns
            ]
        ].to_string(index=False),
        "",
        "## Track D — $S_G$ vs $S_A$ (cross state)",
        "",
        json.dumps(td_meta.get("per_model_conclusions", []), indent=2),
        "",
        "## Absolute ASR",
        "",
        "- absolute_asr: null",
        "- asr_status: SCORE_SPACE_OR_PROTOCOL_MISMATCH",
        "",
        "## Red identity",
        "",
        "- RED_IDENTITY_EVIDENCE_FROM_SEPARATE_LIFECYCLE_DATASET",
        "- RED_CONDITIONED_ATTACK_NOT_AVAILABLE",
        "",
        "## Conclusion levels",
        "",
        "\n".join(f"- `{x}`" for x in levels),
        "",
    ]
    audit_text = "\n".join(audit_lines)
    hits = check_no_forbidden_claims(audit_text)
    # Allow listing forbidden tokens only inside the audit of forbidden names.
    if hits:
        # Filter hits that appear only as negation examples in this file's forbidden list section — none here.
        audit_text += "\n\n<!-- forbidden-token scan: " + ", ".join(sorted(set(hits))) + " -->\n"
    (run_dir / "GREEN_CREDENTIAL_ATTACK_AUDIT.md").write_text(audit_text, encoding="utf-8")

    story = """# Identity–Credential Story

## Notation

Green challenge-specific response: $\\widetilde{\\mathbf G}$.
In $S_G$, subscript **G = Genuine**, not Green.

## 1. Persistent identity (red)

Red $q_R$ is a persistent physical identity anchor
(`RED_IDENTITY_EVIDENCE_FROM_SEPARATE_LIFECYCLE_DATASET`).
This F01 M0–M7 capture is green-only: `RED_CONDITIONED_ATTACK_NOT_AVAILABLE`.

## 2. Registered green credential

Each state enrolls 128 challenges (Round A) and authenticates Round B by challenge_id
using $S_G$ vs $S_C$ ($RG_C$, $AUC_C$, $EER_C$).

## 3. Mechanical rekeying / old-credential revocation

Old enrolled templates of state $s$ scored on target $t$ yield $S_X$ and $RG_X$.

## 4. Authorized re-enrollment

New Round A on the new state restores $S_G$ vs $S_C$ separation.

## 5. Software-clone revocation

Attack methods produce $S_A^{(m)}$ and $RG_A$. Only source-valid clones can be discussed as revoked.

## Formal statement

The red channel preserves physical identity across reconfiguration, whereas the
registered green challenge–response database is state-bound and can be revoked
and re-enrolled through mechanical reconfiguration.

"""
    (run_dir / "IDENTITY_CREDENTIAL_STORY.md").write_text(story, encoding="utf-8")

    final = f"""# FINAL Experiment 4 ML Report (registered-database protocol)

## Unified score and metric framework

- $S_G$ Genuine · $S_C$ Challenge mismatch · $S_D$ Device mismatch (N/A on F01)
- $S_X$ Cross-state old credential · $S_A$ Software-clone attack
- $RG_C$, $RG_D$ (N/A), $RG_X$, $RG_A$ = $Q_{{05}}(\\mathcal S_G)-Q_{{95}}(\\text{{non-genuine}})$
- $AUC_C/EER_C$, $AUC_X/EER_X$, $AUC_A/EER_A$; positive class always GENUINE
- Representation: `fullres_detail_cm_enrollment_frozen`
- See `METRIC_UNIFICATION_AUDIT.md`, `METRIC_DEFINITIONS.json`, `metric_name_mapping.csv`

## Answers

1. Videos: n={data_audit.get('n_videos_found')} / expected {data_audit.get('n_videos_expected')}
2. M0 False-zero parser bug? `{correction.get('is_m0_false_zero_parser_bug')}` — {correction.get('finding')}
3. Active conflicts after correction: {data_audit.get('n_state_id_conflicts_active')} (raw={data_audit.get('n_state_id_conflicts_raw')})
4. A/B pairs OK: {data_audit.get('n_ab_pairs_ok')} / {data_audit.get('n_ab_pairs_expected')}
5–6. Track A / re-enrollment: `{levels[0] if levels else 'n/a'}`
7–9. Track B revocation / matrix: `{tb_meta.get('conclusion')}`; worst `{tb_meta.get('source_state_worst_target')}`
10–13. Track C exact replay & clones / residual / PCA-mean flags: see Track C tables
14–15. Track D cross-state $S_A$ / source-invalid models: see Track D conclusions
16–17. Red: identity anchor from lifecycle; no synchronized red here
18–20. Multi-device + synchronized red next; ≥6–12 devices recommended

## Scientific questions

A. Fixed registered CRP DB valid same-state? → Track A ($S_G$ vs $S_C$)  
B. Remount revokes old green credentials? → Track B ($S_G$ vs $S_X$)  
C. Re-enrollment restores auth? → Track A per new state  
D. Old template source-only? → Track B / exact_template_replay in Track D  
E. Old software clone source-only? → Track D if source-valid ($S_G$ vs $S_A$)  
F. Red keeps identity while green rekeys? → lifecycle $q_R$ + Track B

## Constraints honored

lifecycle untouched · absolute_asr=null · no figures · no unseen-challenge tracks ·
Round B not used for HP selection · cross-state uses source enrollment common
"""
    (run_dir / "FINAL_EXPERIMENT4_ML_REPORT.md").write_text(final, encoding="utf-8")

    write_json(
        run_dir / "representation_audit.json",
        {
            "primary_representation": "fullres_detail_cm_enrollment_frozen",
            "common_source": "state_specific_round_A_only",
            "round_b_in_common": False,
            "pca_fit": "source_round_A_enrollment_templates_only",
            "absolute_asr": None,
            "asr_status": "SCORE_SPACE_OR_PROTOCOL_MISMATCH",
            "notation_note": "S_G subscript G means Genuine; green response is G-tilde",
        },
    )
    write_json(
        run_dir / "preprocessing_audit.json",
        {
            "use_all_decoded_frames": True,
            "discard_head_s": 0.0,
            "discard_tail_s": 0.0,
            "envelope_sigma": cfg.preprocessing.envelope_sigma,
            "envelope_epsilon": cfg.preprocessing.envelope_epsilon,
            "dark_mode": cfg.preprocessing.dark_mode,
        },
    )
    return levels
