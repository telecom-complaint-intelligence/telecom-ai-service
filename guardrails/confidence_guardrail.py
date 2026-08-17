"""
guardrails/confidence_guardrail.py

GUARDRAIL 3: Confidence Guardrail
Rejects or flags low-confidence decisions from agents to prevent unreliable decisions.
"""

from typing import Any

try:
    from app.agents.high.config import CONFIDENCE_THRESHOLD, CRITICAL_CONFIDENCE_THRESHOLD
except ImportError:
    try:
        from config import CONFIDENCE_THRESHOLD, CRITICAL_CONFIDENCE_THRESHOLD
    except ImportError:
        CONFIDENCE_THRESHOLD = 0.60
        CRITICAL_CONFIDENCE_THRESHOLD = 0.75


def validate_confidence_guardrail(
    agent_name: str,
    confidence: float | None,
    priority: str | None = None,
) -> tuple[bool, list[str], list[str]]:
    """
    Validate an agent's confidence score against system safety thresholds.

    Parameters
    ----------
    agent_name : str
        Name of the agent (e.g., 'diagnosis', 'policy', 'risk', 'planner', 'critic').
    confidence : float | None
        Confidence value between 0.0 and 1.0.
    priority : str | None
        Proposed priority (e.g. 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW').

    Returns
    -------
    tuple[bool, list[str], list[str]]
        (is_valid, violations, warnings)
    """
    violations: list[str] = []
    warnings: list[str] = []

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

    # Safety Hard Limit: extremely low confidence fails validation
    if conf_val < 0.40:
        violations.append(
            f"Confidence Guardrail [{agent_name}]: Unacceptable confidence score {conf_val:.2f} (< 0.40 safety limit). Decision unsafe for automation."
        )

    # Stricter threshold for CRITICAL priority actions
    if priority and str(priority).strip().upper() == "CRITICAL" and conf_val < CRITICAL_CONFIDENCE_THRESHOLD:
        warnings.append(
            f"Confidence Guardrail [{agent_name}]: CRITICAL priority decision has confidence {conf_val:.2f}, below critical threshold {CRITICAL_CONFIDENCE_THRESHOLD:.2f}."
        )

    is_valid = len(violations) == 0
    return is_valid, violations, warnings


def check_multi_agent_confidence(state: dict[str, Any]) -> tuple[bool, list[str], list[str]]:
    """
    Evaluate overall confidence across parallel agents (Diagnosis, Policy, Risk).

    Parameters
    ----------
    state : dict[str, Any]
        Aggregated graph state.

    Returns
    -------
    tuple[bool, list[str], list[str]]
        (is_valid, violations, warnings)
    """
    violations: list[str] = []
    warnings: list[str] = []

    diag_conf = state.get("diagnosis_confidence")
    pol_conf = state.get("policy_confidence")
    risk_conf = state.get("risk_confidence")

    for name, score in [("Diagnosis", diag_conf), ("Policy", pol_conf), ("Risk", risk_conf)]:
        _, v, w = validate_confidence_guardrail(name, score)
        violations.extend(v)
        warnings.extend(w)

    is_valid = len(violations) == 0
    return is_valid, violations, warnings
