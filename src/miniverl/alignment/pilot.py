"""Conservative, inspectable rules for the bounded alignment pilot."""

from __future__ import annotations

from miniverl.alignment.schema import PilotEvidence, PilotRecommendation, PilotResult

__all__ = ["recommend_alignment_method"]


def recommend_alignment_method(evidence: PilotEvidence) -> PilotResult:
    """Return a recommendation without treating a tiny pilot as certainty."""
    reasons: list[str] = []
    if evidence.sample_size < 24 or (
        evidence.uncertainty_half_width is not None and evidence.uncertainty_half_width > 0.25
    ):
        recommendation = PilotRecommendation.INSUFFICIENT_EVIDENCE
        reasons.append("fewer than 24 observations or uncertainty wider than 0.25")
    elif evidence.teacher_policy_competence < 0.70:
        recommendation = PilotRecommendation.CONTINUED_SFT
        reasons.append("teacher policy competence is below the preregistered 0.70 floor")
    elif evidence.preference_win_gap >= 0.10 and evidence.hard_soft_gap < 0.03:
        recommendation = PilotRecommendation.DPO
        reasons.append("paired preference signal is material while soft-target evidence is weak")
    elif abs(evidence.fresh_state_gap) < 0.03 and evidence.hard_soft_gap >= 0.03:
        recommendation = PilotRecommendation.OFFLINE_DISTILLATION
        reasons.append("fresh-state gain is below 0.03 while soft distributions retain signal")
    elif (
        evidence.fresh_state_gap >= 0.03
        and evidence.hard_soft_gap >= 0.03
        and evidence.policy_sensitive_token_fraction <= 0.35
        and evidence.verifier_precision >= 0.80
    ):
        recommendation = PilotRecommendation.VERIFIER_GATED_OPD
        reasons.append(
            "fresh soft supervision matters and a precise gate covers a sparse policy region"
        )
    elif evidence.fresh_state_gap >= 0.03 and evidence.hard_soft_gap >= 0.03:
        recommendation = PilotRecommendation.STANDARD_OPD
        reasons.append("matched diagnostics support both fresh states and soft distributions")
    else:
        recommendation = PilotRecommendation.INSUFFICIENT_EVIDENCE
        reasons.append("diagnostic gaps do not clear a preregistered method threshold")
    return PilotResult(
        recommendation=recommendation,
        evidence=evidence,
        reasons=reasons,
        cost_assumptions={
            "estimated_vram_gib": evidence.estimated_vram_gib,
            "estimated_time_seconds": evidence.estimated_time_seconds,
            "teacher_query_fraction_assumption": evidence.policy_sensitive_token_fraction,
        },
        uncertainty_note=(
            "A pilot is a bounded diagnostic, not a final comparison. Freeze a matched, "
            "multi-seed test before making an alignment claim."
        ),
    )
