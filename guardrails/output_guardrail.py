"""
guardrails/output_guardrail.py

GUARDRAIL 2: Output Guardrail
Validates that all agent outputs adhere strictly to expected JSON schemas,
data types, and domain-allowed enums.
"""

from typing import Any

AGENT_SCHEMAS = {
    "diagnosis": {
        "required_fields": ["diagnosis", "root_cause", "confidence"],
        "types": {"diagnosis": str, "root_cause": str, "confidence": (int, float)},
    },
    "policy": {
        "required_fields": ["policy_status", "policy_reason", "policy_confidence"],
        "types": {"policy_status": str, "policy_reason": str, "policy_confidence": (int, float)},
        "allowed_enums": {
            "policy_status": ["NORMAL", "ELEVATED", "CRITICAL", "UNKNOWN"]
        },
    },
    "risk": {
        "required_fields": ["risk_level", "risk_reason", "risk_confidence"],
        "types": {"risk_level": str, "risk_reason": str, "risk_confidence": (int, float)},
        "allowed_enums": {
            "risk_level": ["LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"]
        },
    },
    "planner": {
        "required_fields": ["proposed_decision", "priority", "proposed_action", "reason", "planner_confidence"],
        "types": {
            "proposed_decision": str,
            "priority": str,
            "proposed_action": str,
            "reason": str,
            "planner_confidence": (int, float),
        },
        "allowed_enums": {
            "proposed_decision": ["NORMAL", "ESCALATE", "CRITICAL"],
            "priority": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        },
    },
    "critic": {
        "required_fields": ["critic_decision", "critic_reason", "critic_confidence", "replan_required"],
        "types": {
            "critic_decision": str,
            "critic_reason": str,
            "critic_confidence": (int, float),
            "replan_required": bool,
        },
        "allowed_enums": {
            "critic_decision": ["ACCEPT", "REJECT"]
        },
    },
    "replan": {
        "required_fields": ["revised_decision", "revised_priority", "revised_action", "revised_reason", "replan_confidence"],
        "types": {
            "revised_decision": str,
            "revised_priority": str,
            "revised_action": str,
            "revised_reason": str,
            "replan_confidence": (int, float),
        },
        "allowed_enums": {
            "revised_decision": ["NORMAL", "ESCALATE", "CRITICAL"],
            "revised_priority": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        },
    },
}


def validate_agent_output_schema(
    agent_name: str,
    output_dict: dict[str, Any]
) -> tuple[bool, list[str], list[str]]:
    """
    Validate agent output dictionary against its defined schema.

    Parameters
    ----------
    agent_name : str
        One of 'diagnosis', 'policy', 'risk', 'planner', 'critic', 'replan'.
    output_dict : Dict[str, Any]
        Output dictionary produced by the agent.

    Returns
    -------
    Tuple[bool, List[str], List[str]]
        (is_valid, violations, warnings)
    """
    violations: list[str] = []
    warnings: list[str] = []

    schema = AGENT_SCHEMAS.get(agent_name)
    if not schema:
        warnings.append(f"Output Guardrail: No strict schema registered for agent '{agent_name}'.")
        return True, violations, warnings

    if not isinstance(output_dict, dict):
        violations.append(f"Output Guardrail [{agent_name}]: Output is not a dictionary / JSON object.")
        return False, violations, warnings

    # 1. Check Required Fields
    for field in schema["required_fields"]:
        # Check alternative alias for planner reason
        if agent_name == "planner" and field == "reason" and ("planner_reason" in output_dict or "reason" in output_dict):
            continue
        if field not in output_dict:
            violations.append(f"Output Guardrail [{agent_name}]: Missing required field '{field}'.")

    # 2. Check Data Types
    for field, expected_type in schema["types"].items():
        val = output_dict.get(field)
        if val is None and agent_name == "planner" and field == "reason":
            val = output_dict.get("planner_reason")
        if val is not None and not isinstance(val, expected_type):
            violations.append(
                f"Output Guardrail [{agent_name}]: Field '{field}' has invalid type {type(val).__name__} (expected {expected_type})."
            )

    # 3. Check Allowed Enums
    allowed_enums = schema.get("allowed_enums", {})
    for field, allowed_values in allowed_enums.items():
        val = output_dict.get(field)
        if val is not None:
            val_str = str(val).strip().upper()
            if val_str not in [v.upper() for v in allowed_values]:
                violations.append(
                    f"Output Guardrail [{agent_name}]: Field '{field}' has invalid value '{val}' (allowed: {allowed_values})."
                )

    # 4. Check Confidence Range if present
    confidence_field = next((f for f in output_dict if "confidence" in f.lower()), None)
    if confidence_field and output_dict.get(confidence_field) is not None:
        try:
            conf_val = float(output_dict[confidence_field])
            if not (0.0 <= conf_val <= 1.0):
                violations.append(
                    f"Output Guardrail [{agent_name}]: Confidence '{conf_val}' outside range [0.0, 1.0]."
                )
        except (ValueError, TypeError):
            violations.append(f"Output Guardrail [{agent_name}]: Confidence field '{confidence_field}' is not a valid number.")

    is_valid = len(violations) == 0
    return is_valid, violations, warnings
