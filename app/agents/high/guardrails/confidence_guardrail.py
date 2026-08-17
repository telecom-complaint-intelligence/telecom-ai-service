"""
guardrails/confidence_guardrail.py

GUARDRAIL 3: Confidence Guardrail
Rejects or flags low-confidence decisions from agents to prevent unreliable decisions.
"""

from typing import Dict, Any, Tuple, List, Optional
from app.agents.high.config import CONFIDENCE_THRESHOLD, CRITICAL_CONFIDENCE_THRESHOLD


def validate_confidence_guardrail(
    agent_name: str,
    confidence: Optional[float],
    priority: Optional[str] = None
) -> Tuple[bool, List[str], List[str]]:
    """
    Validate an agent's confidence score against system safety thresholds.

    Parameters
    ----------
    agent_name : str
        Name of the agent (e.g., 'diagnosis', 'policy', 'risk', 'planner', 'critic').
    confidence : Optional[float]
        Confidence value between 0.0 and 1.0.
    priority : Optional[str]
        Proposed priority (e.g. 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW').

    Returns
    -------
    Tuple[bool, List[str], List[str]]
        (is_valid, violations, warnings)
    """
    violations: List[str] = []
    warnings: List[str] = []

    if confidence is None:
        warnings.append(f"Confidence Guardrail [{agent_name}]: Confidence score not provided; assuming minimum baseline.")
        confidence = 0.5

    try:
        conf_val = float(confidence)
    except (ValueError, TypeError):
        violations.append(f"Confidence Guardrail [{agent_name}]: Invalid confidence value '{confidence}'.")
        return False, violations, warnings

    # General threshold check
    if conf_val < CONFIDENCE_THRESHOLD:
        warnings.append(
            f"Confidence Guardrail [{agent_name}]: Low confidence score {conf_val:.2f} (below threshold {CONFIDENCE_THRESHOLD:.2f})."
        )
        if conf_val < 0.40:
            violations.append(
                f"Confidence Guardrail [{agent_name}]: Unacceptable confidence score {conf_val:.2f} (< 0.40); decision cannot be automated safely."
            )

    # Stricter threshold for CRITICAL priority actions
    if priority and str(priority).strip().upper() == "CRITICAL":
        if conf_val < CRITICAL_CONFIDENCE_THRESHOLD:
            warnings.append(
                f"Confidence Guardrail [{agent_name}]: CRITICAL priority decision has confidence {conf_val:.2f}, below critical threshold {CRITICAL_CONFIDENCE_THRESHOLD:.2f}."
            )

    is_valid = len(violations) == 0
    return is_valid, violations, warnings


def check_multi_agent_confidence(state: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """
    Evaluate overall confidence across parallel agents (Diagnosis, Policy, Risk).

    Parameters
    ----------
    state : Dict[str, Any]
        Current LangGraph state.

    Returns
    -------
    Tuple[bool, List[str], List[str]]
        (is_valid, violations, warnings)
    """
    violations: List[str] = []
    warnings: List[str] = []

    diag_conf = state.get("diagnosis_confidence")
    pol_conf = state.get("policy_confidence")
    risk_conf = state.get("risk_confidence")

    valid_d, viol_d, warn_d = validate_confidence_guardrail("diagnosis", diag_conf)
    valid_p, viol_p, warn_p = validate_confidence_guardrail("policy", pol_conf)
    valid_r, viol_r, warn_r = validate_confidence_guardrail("risk", risk_conf)

    violations.extend(viol_d + viol_p + viol_r)
    warnings.extend(warn_d + warn_p + warn_r)

    is_valid = len(violations) == 0
    return is_valid, violations, warnings
