"""
guardrails/human_guardrail.py

GUARDRAIL 8: Human Guardrail (Human-in-the-Loop)
Safely escalates the decision to a human telecom operations engineer or supervisor
when AI agents cannot reach a safe, high-confidence, or validated consensus.
"""

from typing import Dict, Any, List
from datetime import datetime


def assemble_human_review_dossier(
    state: Dict[str, Any],
    escalation_reason: str,
    violations: List[str]
) -> Dict[str, Any]:
    """
    Assemble a structured human review dossier for manual intervention.

    Parameters
    ----------
    state : Dict[str, Any]
        Current LangGraph state containing all multi-agent outputs.
    escalation_reason : str
        Primary trigger reason for human escalation.
    violations : List[str]
        List of all guardrail violations leading to escalation.

    Returns
    -------
    Dict[str, Any]
        Structured human review package.
    """
    complaint_id = state.get("complaint_id", "UNKNOWN")
    customer_id = state.get("customer_id", "UNKNOWN")
    complaint_text = state.get("complaint_text") or state.get("complaint", "")
    scope = state.get("scope", "individual")
    days_unresolved = state.get("days_unresolved", 1)
    duration_hours = state.get("duration_hours", 0.0)

    # Collect agent recommendations
    diagnosis = state.get("diagnosis", "Not completed")
    root_cause = state.get("root_cause") or state.get("diagnosis_root_cause", "UNKNOWN")
    policy_status = state.get("policy_status", "UNKNOWN")
    risk_level = state.get("risk_level", "UNKNOWN")

    proposed_dec = state.get("revised_decision") or state.get("proposed_decision", "UNRESOLVED")
    proposed_prio = state.get("revised_priority") or state.get("priority", "HIGH")
    proposed_act = state.get("revised_action") or state.get("proposed_action", "Manual Triage Required")

    critic_feedback = state.get("critic_reason", "Critic rejected or failed to validate proposal.")

    # Recommended operator checklist
    checklist = [
        "1. Verify customer circuit and physical network telemetry in OSS/BSS.",
        "2. Review Critic rejection reasons and resolve policy/risk discrepancies.",
        "3. Authorize technician dispatch or tier-2 escalation in ticketing portal.",
        "4. Contact customer directly with SLA expectation and incident reference."
    ]

    if days_unresolved >= 7:
        checklist.insert(0, "🚨 CRITICAL SLA BREACH (>=7 Days): Contact Executive Customer Care Lead immediately.")

    dossier = {
        "dossier_id": f"HUMAN-REV-{complaint_id}-{int(datetime.now().timestamp())}",
        "escalation_timestamp": datetime.now().isoformat(),
        "escalation_reason": escalation_reason,
        "complaint_summary": {
            "complaint_id": complaint_id,
            "customer_id": customer_id,
            "text": complaint_text,
            "scope": scope,
            "days_unresolved": days_unresolved,
            "duration_hours": duration_hours,
        },
        "agent_signals": {
            "diagnosis": diagnosis,
            "root_cause": root_cause,
            "policy_status": policy_status,
            "risk_level": risk_level,
            "last_proposed_decision": proposed_dec,
            "last_proposed_priority": proposed_prio,
            "last_proposed_action": proposed_act,
            "critic_feedback": critic_feedback,
        },
        "guardrail_violations": violations,
        "human_operator_checklist": checklist,
        "suggested_human_action": f"Manual operational review and technician dispatch authorization for ticket {customer_id}."
    }

    return dossier
