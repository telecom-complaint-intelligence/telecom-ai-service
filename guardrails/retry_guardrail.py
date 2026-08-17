"""
guardrails/retry_guardrail.py

GUARDRAIL 7: Retry Guardrail
Prevents infinite Critic <-> Replan cycling by enforcing strict retry quotas and
initiating human escalation when automated iterations are exhausted.
"""

from typing import Any

try:
    from app.agents.high.config import MAX_REPLANS
except ImportError:
    try:
        from config import MAX_REPLANS
    except ImportError:
        MAX_REPLANS = 1


def validate_retry_guardrail(
    state: dict[str, Any],
    max_retries: int = MAX_REPLANS,
) -> tuple[bool, bool, list[str], list[str]]:
    """
    Evaluate whether a replan cycle is allowed or if retries have been exhausted.

    Parameters
    ----------
    state : dict[str, Any]
        Current state with 'retry_count' and 'max_retries'.
    max_retries : int
        Maximum permitted replan iterations.

    Returns
    -------
    tuple[bool, bool, list[str], list[str]]
        (can_retry, should_escalate_human, violations, warnings)
    """
    violations: list[str] = []
    warnings: list[str] = []


    retry_count = int(state.get("retry_count", 0))
    configured_max = int(state.get("max_retries", max_retries))

    if retry_count < configured_max:
        can_retry = True
        should_escalate_human = False
        warnings.append(
            f"Retry Guardrail: Replan permitted (Attempt {retry_count + 1} of {configured_max})."
        )
    else:
        can_retry = False
        should_escalate_human = True
        violations.append(
            f"Retry Guardrail: Maximum retry limit reached ({retry_count}/{configured_max} replan cycles executed). Replan locked; escalating to Human Review."
        )

    return can_retry, should_escalate_human, violations, warnings
