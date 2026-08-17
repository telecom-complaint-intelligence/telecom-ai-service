"""
guardrails/risk_guardrail.py

GUARDRAIL 6: Risk Guardrail
Ensures high-risk, catastrophic, area-wide, or SLA-breaching complaints are NEVER
downplayed, treated as normal, or assigned inadequate priority.
"""

from typing import Any


def validate_risk_guardrail(
    state: dict[str, Any],
    proposed_decision: str,
    proposed_priority: str
) -> tuple[bool, list[str], list[str]]:
    """
    Validate that priority and decision properly reflect underlying risk metrics.

    Parameters
    ----------
    state : Dict[str, Any]
        Current state containing risk level, duration, scope, and affected subscribers.
    proposed_decision : str
        Decision ('NORMAL', 'ESCALATE', 'CRITICAL').
    proposed_priority : str
        Priority ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL').

    Returns
    -------
    Tuple[bool, List[str], List[str]]
        (is_valid, violations, warnings)
    """
    violations: list[str] = []
    warnings: list[str] = []

    dec = str(proposed_decision or "").strip().upper()
    prio = str(proposed_priority or "").strip().upper()

    risk_level = str(state.get("risk_level", "UNKNOWN")).strip().upper()
    days_unresolved = int(state.get("days_unresolved") or 1)
    affected_subscribers = int(state.get("affected_subscribers") or 1)
    scope = str(state.get("scope", "individual")).strip().lower()

    # Rule 1: 7-Day Unresolved SLA Breach must be CRITICAL
    if days_unresolved >= 7 and prio != "CRITICAL" and dec != "CRITICAL":
        violations.append(
            f"Risk Guardrail: Issue unresolved for {days_unresolved} days (>= 7-day threshold) must have CRITICAL priority (received '{prio}')."
        )

    # Rule 2: High Risk Level cannot be LOW priority or NORMAL decision
    if risk_level == "HIGH":
        if prio in ("LOW", "MEDIUM") and dec == "NORMAL":
            violations.append(
                f"Risk Guardrail: Complaint assessed with HIGH risk cannot be handled with {prio} priority and NORMAL decision."
            )
        elif prio in ("LOW", "MEDIUM"):
            warnings.append(
                f"Risk Guardrail: Priority is {prio} despite HIGH risk level; escalated attention recommended."
            )

    # Rule 3: Major Area Disruption (> 500 affected subscribers) must be HIGH or CRITICAL
    if (scope in ("area", "district", "neighborhood") or affected_subscribers >= 500) and prio in ("LOW", "MEDIUM"):
        violations.append(
            f"Risk Guardrail: Major area disruption impacting {affected_subscribers:,} consumers requires at least HIGH or CRITICAL priority (received '{prio}')."
        )

    # Rule 4: Critical Risk Level must have CRITICAL priority
    if risk_level == "CRITICAL" and prio != "CRITICAL":
        violations.append(
            f"Risk Guardrail: System risk level is CRITICAL; priority cannot be downgraded to '{prio}'."
        )

    is_valid = len(violations) == 0
    return is_valid, violations, warnings
