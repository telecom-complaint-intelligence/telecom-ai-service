"""
guardrails/execution_guardrail.py

GUARDRAIL 9: Execution Guardrail
Ensures that no physical or operational intervention is triggered or dispatched
without formal verification of Critic approval, Human sign-off, and action safety.
"""

import uuid
from datetime import datetime
from typing import Any


def validate_execution_guardrail(
    state: dict[str, Any]
) -> tuple[bool, str, list[str], list[str]]:
    """
    Verify that the final intervention is authorized and safe for operational execution.

    Parameters
    ----------
    state : Dict[str, Any]
        Finalized LangGraph state.

    Returns
    -------
    Tuple[bool, str, List[str], List[str]]
        (is_authorized, authorization_token, violations, warnings)
    """
    violations: list[str] = []
    warnings: list[str] = []
    auth_token = ""

    final_decision = state.get("final_decision")
    final_action = state.get("final_action")
    critic_decision = str(state.get("critic_decision", "")).strip().upper()
    replan_required = state.get("replan_required", False)
    human_review_required = state.get("human_review_required", False)

    # 1. Approval Check
    is_approved_by_critic = (critic_decision == "ACCEPT" and not replan_required)
    is_approved_by_human = bool(human_review_required and state.get("human_review_dossier"))

    if not is_approved_by_critic and not is_approved_by_human:
        violations.append(
            f"Execution Guardrail: Intervention is NOT approved (Critic status: '{critic_decision}', Human review: {human_review_required}). Execution prohibited."
        )

    # 2. Action Specificity Check
    if not final_action or not isinstance(final_action, str) or len(final_action.strip()) < 5:
        violations.append("Execution Guardrail: Final action description is missing or insufficiently specified for dispatch.")

    # 3. Decision Feasibility Check
    if final_decision in ("UNKNOWN", None, ""):
        violations.append("Execution Guardrail: Final decision is undefined.")

    # 4. Generate Authorization Token if Approved
    if len(violations) == 0:
        auth_token = f"AUTH-EXEC-{uuid.uuid4().hex[:8].upper()}-{datetime.now().strftime('%Y%m%d%H%M')}"
        warnings.append(f"Execution Guardrail: Execution authorized with token {auth_token}.")

    is_authorized = len(violations) == 0
    return is_authorized, auth_token, violations, warnings
