"""
guardrails/policy_guardrail.py

GUARDRAIL 4: Policy Guardrail
Validates that proposed interventions adhere strictly to telecom operational policy,
compliance regulations, and authorization boundaries.
"""

import re
from typing import Any

# Prohibited phrases in proposed actions without human supervisor authorization
PROHIBITED_ACTION_PATTERNS = [
    (r"\b(terminate|cancel)\s+(account|service|contract)\b", "Direct account/contract termination requires manual supervisor sign-off."),
    (r"\b(refund|credit)\s+(\$?[0-9]{3,}|greater than 100|exceeding 100)\b", "Automatic refunds exceeding $100 require financial authorization."),
    (r"\b(disconnect|shut down|blackout)\s+(area|substation|district)\b", "Area-wide power or substation disconnections cannot be automated."),
    (r"\b(ignore|dismiss|close\s+ticket\s+without\s+action)\b", "Closing severe or unresolved tickets without action violates care policy."),
]

# Mandatory policies for specific triggers
MANDATORY_POLICY_RULES = [
    {
        "condition": lambda s: s.get("policy_status") == "ELEVATED" or s.get("risk_level") in ("HIGH", "CRITICAL"),
        "rule": lambda a: not any(w in a.lower() for w in ["no action", "ignore", "standard self-service", "close ticket"]),
        "violation": "Policy Guardrail: Elevated/High-risk complaints cannot be resolved with passive self-service or no action."
    },
    {
        "condition": lambda s: bool(s.get("days_unresolved", 0) >= 7),
        "rule": lambda a: any(w in a.lower() for w in ["escalat", "dispatch", "retention", "senior", "executive", "tier-2", "field", "priority"]),
        "violation": "Policy Guardrail: 7-Day overdue SLA breach requires active technical or executive escalation in the proposed action."
    }
]


def validate_policy_guardrail(
    state: dict[str, Any],
    proposed_action: str
) -> tuple[bool, list[str], list[str]]:
    """
    Validate proposed action against company policy and compliance rules.

    Parameters
    ----------
    state : Dict[str, Any]
        Current state containing complaint and agent outputs.
    proposed_action : str
        Proposed action string from Planner or Replan agent.

    Returns
    -------
    Tuple[bool, List[str], List[str]]
        (is_valid, violations, warnings)
    """
    violations: list[str] = []
    warnings: list[str] = []

    if not proposed_action or not isinstance(proposed_action, str) or not proposed_action.strip():
        violations.append("Policy Guardrail: Proposed action is empty or undefined.")
        return False, violations, warnings

    action_lower = proposed_action.lower()

    # 1. Check for Prohibited Action Patterns
    for pattern, reason in PROHIBITED_ACTION_PATTERNS:
        if re.search(pattern, action_lower, re.IGNORECASE):
            violations.append(f"Policy Guardrail: Prohibited Action Detected - {reason}")

    # 2. Check Mandatory Policy Rules
    for check in MANDATORY_POLICY_RULES:
        if check["condition"](state) and not check["rule"](proposed_action):
            violations.append(check["violation"])

    # 3. Policy Status Consistency
    policy_status = state.get("policy_status", "UNKNOWN")
    if policy_status == "ELEVATED" and state.get("proposed_decision") == "NORMAL":
        warnings.append("Policy Guardrail: Decision is 'NORMAL' despite ELEVATED policy status; verify justification.")

    is_valid = len(violations) == 0
    return is_valid, violations, warnings
