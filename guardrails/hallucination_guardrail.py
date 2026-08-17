"""
guardrails/hallucination_guardrail.py

GUARDRAIL 5: Hallucination Guardrail
Detects and blocks unsupported technical assumptions, fabricated hardware telemetry,
or fictitious company policies not grounded in input evidence.
"""

import re
from typing import Any

# Suspicious specific hardware/telemetry phrases when telemetry logs are UNKNOWN
FABRICATED_TELEMETRY_PATTERNS = [
    (r"\b(port\s+[0-9]+\s+flapping|bounced\s+port\s+[0-9]+)\b", "Asserted specific port flapping telemetry without network switch logs."),
    (r"\b(olt\s+card\s+slot\s+[0-9]+|pon\s+port\s+[0-9]+)\b", "Asserted specific OLT chassis slot fault without optical telemetry."),
    (r"\b(firmware\s+v[0-9\.]+\s+corrupt)\b", "Asserted specific firmware version corruption without firmware diagnostic payload."),
    (r"\b(capacitor\s+blown|laser\s+diode\s+burnout)\b", "Asserted component-level silicon/hardware destruction without physical teardown evidence."),
    (r"\b(according to company policy section\s+[0-9\.]+)\b", "Asserted specific non-existent formal policy clause numbers."),
]


def validate_hallucination_guardrail(
    state: dict[str, Any],
    agent_name: str,
    text_to_check: str
) -> tuple[bool, list[str], list[str]]:
    """
    Scan agent output text for evidence hallucinations and unsupported claims.

    Parameters
    ----------
    state : Dict[str, Any]
        Current state containing original complaint text and evidence.
    agent_name : str
        Name of agent being evaluated.
    text_to_check : str
        Reasoning, diagnosis, or action text produced by the agent.

    Returns
    -------
    Tuple[bool, List[str], List[str]]
        (is_valid, violations, warnings)
    """
    violations: list[str] = []
    warnings: list[str] = []

    if not text_to_check or not isinstance(text_to_check, str):
        return True, violations, warnings

    text_lower = text_to_check.lower()

    # 1. Root Cause Hallucination Check
    root_cause = str(state.get("root_cause") or state.get("diagnosis_root_cause") or "UNKNOWN").strip().upper()
    if root_cause == "UNKNOWN" and any(term in text_lower for term in ["confirmed hardware defect", "defective optical laser", "corrupt bootloader"]):
        violations.append(
            f"Hallucination Guardrail [{agent_name}]: Claimed confirmed technical failure when root_cause is officially UNKNOWN."
        )

    # 2. Fabricated Telemetry / Hardware Check
    for pattern, explanation in FABRICATED_TELEMETRY_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            violations.append(f"Hallucination Guardrail [{agent_name}]: {explanation}")

    # 3. Grounding Verification with Complaint Context
    complaint_text = (state.get("complaint_text") or state.get("complaint") or "").lower()

    # If customer only complained about billing, but agent asserts fiber optic cable cut
    if "bill" in complaint_text and not any(k in complaint_text for k in ["cable", "cut", "fiber", "line", "down"]) and any(k in text_lower for k in ["severed fiber cable", "underground cable cut", "optical line break"]):
        violations.append(
            f"Hallucination Guardrail [{agent_name}]: Substituted fiber cut disaster scenario into a pure billing issue."
        )

    is_valid = len(violations) == 0
    return is_valid, violations, warnings
