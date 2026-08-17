"""
guardrails/input_guardrail.py

GUARDRAIL 1: Input Guardrail
Validates complaint payload and historical data integrity before multi-agent processing.
"""

from typing import Dict, Any, Tuple, List
from app.agents.high.config import MIN_COMPLAINT_TEXT_LENGTH, MAX_COMPLAINT_TEXT_LENGTH


def validate_input_guardrail(state: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """
    Validate input complaint data and historical metrics.

    Parameters
    ----------
    state : Dict[str, Any]
        Incoming complaint state.

    Returns
    -------
    Tuple[bool, List[str], List[str]]
        (is_valid, violations, warnings)
    """
    violations: List[str] = []
    warnings: List[str] = []

    # 1. Mandatory Identity Fields
    complaint_id = state.get("complaint_id")
    if not complaint_id or str(complaint_id).strip() in ("", "UNKNOWN", "None"):
        violations.append("Input Guardrail: Missing or empty 'complaint_id'.")

    customer_id = state.get("customer_id")
    if not customer_id or str(customer_id).strip() in ("", "UNKNOWN", "None"):
        warnings.append("Input Guardrail: 'customer_id' is missing or unassigned; defaulting to ticket reference.")

    # 2. Complaint Text Content & Length
    complaint_text = state.get("complaint_text") or state.get("complaint")
    if not complaint_text or not isinstance(complaint_text, str) or not complaint_text.strip():
        violations.append("Input Guardrail: Complaint text is missing or empty.")
    else:
        clean_text = complaint_text.strip()
        if len(clean_text) < MIN_COMPLAINT_TEXT_LENGTH:
            violations.append(
                f"Input Guardrail: Complaint text too short ({len(clean_text)} chars, minimum {MIN_COMPLAINT_TEXT_LENGTH})."
            )
        if len(clean_text) > MAX_COMPLAINT_TEXT_LENGTH:
            warnings.append(
                f"Input Guardrail: Complaint text is unusually long ({len(clean_text)} chars); truncating to {MAX_COMPLAINT_TEXT_LENGTH}."
            )

    # 3. Numeric Metric Boundaries
    duration_hours = state.get("duration_hours")
    if duration_hours is not None:
        try:
            dur_val = float(duration_hours)
            if dur_val < 0:
                violations.append("Input Guardrail: 'duration_hours' cannot be negative.")
            elif dur_val > 8760:  # > 1 year
                warnings.append("Input Guardrail: 'duration_hours' exceeds 1 year (8,760h); verify timestamp accuracy.")
        except (ValueError, TypeError):
            violations.append("Input Guardrail: 'duration_hours' must be a numeric value.")

    days_unresolved = state.get("days_unresolved")
    if days_unresolved is not None:
        try:
            days_val = int(days_unresolved)
            if days_val < 0:
                violations.append("Input Guardrail: 'days_unresolved' cannot be negative.")
        except (ValueError, TypeError):
            violations.append("Input Guardrail: 'days_unresolved' must be an integer.")

    affected_subscribers = state.get("affected_subscribers")
    if affected_subscribers is not None:
        try:
            aff_val = int(affected_subscribers)
            if aff_val < 1:
                warnings.append("Input Guardrail: 'affected_subscribers' is less than 1; normalized to 1.")
        except (ValueError, TypeError):
            violations.append("Input Guardrail: 'affected_subscribers' must be an integer.")

    sentiment_score = state.get("sentiment_score")
    if sentiment_score is not None:
        try:
            sent_val = float(sentiment_score)
            if not (0.0 <= sent_val <= 1.0):
                warnings.append(
                    f"Input Guardrail: 'sentiment_score' ({sent_val}) is outside [0.0, 1.0]; clamping value."
                )
        except (ValueError, TypeError):
            violations.append("Input Guardrail: 'sentiment_score' must be a float between 0.0 and 1.0.")

    # 4. Domain & Problem Type Sanity
    domain = state.get("domain")
    if not domain or str(domain).strip() in ("", "UNKNOWN"):
        warnings.append("Input Guardrail: 'domain' is unspecified; will be inferred by agents.")

    is_valid = len(violations) == 0
    return is_valid, violations, warnings
