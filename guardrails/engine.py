"""
guardrails/engine.py

UNIFIED GUARDRAILS ENGINE
Central orchestrator for executing, logging, and auditing all 9 Guardrails across the LangGraph workflow.
"""

from datetime import datetime
from typing import Any

from guardrails.confidence_guardrail import (
    check_multi_agent_confidence,
    validate_confidence_guardrail,
)
from guardrails.execution_guardrail import validate_execution_guardrail
from guardrails.hallucination_guardrail import validate_hallucination_guardrail
from guardrails.input_guardrail import validate_input_guardrail
from guardrails.output_guardrail import validate_agent_output_schema
from guardrails.policy_guardrail import validate_policy_guardrail
from guardrails.risk_guardrail import validate_risk_guardrail


class GuardrailsEngine:
    """
    Unified manager for executing, evaluating, and auditing guardrails.
    """

    @staticmethod
    def run_input_guardrail(state: dict[str, Any]) -> dict[str, Any]:
        """Run Guardrail 1 (Input Guardrail)."""
        valid, violations, warnings = validate_input_guardrail(state)
        
        status_entry = {
            "guardrail_name": "1. Input Guardrail",
            "passed": valid,
            "violations": violations,
            "warnings": warnings,
            "timestamp": datetime.now().isoformat()
        }

        current_status = dict(state.get("guardrail_status") or {})
        current_status["input_guardrail"] = status_entry

        all_violations = list(state.get("guardrail_violations") or []) + violations
        all_warnings = list(state.get("guardrail_warnings") or []) + warnings

        return {
            "guardrail_status": current_status,
            "guardrail_violations": all_violations,
            "guardrail_warnings": all_warnings,
            "input_valid": valid
        }

    @staticmethod
    def run_multi_agent_guardrail(state: dict[str, Any]) -> dict[str, Any]:
        """Run Guardrails 2 & 3 (Output & Confidence Guardrails on Diagnosis, Policy, Risk)."""
        violations: list[str] = []
        warnings: list[str] = []

        # 1. Output Schema Validation
        for agent in ["diagnosis", "policy", "risk"]:
            v, viol, warn = validate_agent_output_schema(agent, state)
            violations.extend(viol)
            warnings.extend(warn)

        # 2. Confidence Check
        _, c_viol, c_warn = check_multi_agent_confidence(state)
        violations.extend(c_viol)
        warnings.extend(c_warn)

        passed = len(violations) == 0

        status_entry = {
            "guardrail_name": "2 & 3. Output & Confidence Guardrails",
            "passed": passed,
            "violations": violations,
            "warnings": warnings,
            "timestamp": datetime.now().isoformat()
        }

        current_status = dict(state.get("guardrail_status") or {})
        current_status["multi_agent_guardrail"] = status_entry

        all_violations = list(state.get("guardrail_violations") or []) + violations
        all_warnings = list(state.get("guardrail_warnings") or []) + warnings

        return {
            "guardrail_status": current_status,
            "guardrail_violations": all_violations,
            "guardrail_warnings": all_warnings,
        }

    @staticmethod
    def run_planner_guardrails(state: dict[str, Any], planner_output: dict[str, Any]) -> dict[str, Any]:
        """Run Guardrails 2, 3, 4, 5, 6 on Planner proposal."""
        violations: list[str] = []
        warnings: list[str] = []

        # Output schema
        _, v_viol, v_warn = validate_agent_output_schema("planner", planner_output)
        violations.extend(v_viol)
        warnings.extend(v_warn)

        # Confidence
        prio = planner_output.get("priority")
        conf = planner_output.get("planner_confidence")
        _, c_viol, c_warn = validate_confidence_guardrail("planner", conf, prio)
        violations.extend(c_viol)
        warnings.extend(c_warn)

        # Policy
        action = planner_output.get("proposed_action", "")
        _, p_viol, p_warn = validate_policy_guardrail(state, action)
        violations.extend(p_viol)
        warnings.extend(p_warn)

        # Hallucination
        reason = planner_output.get("reason") or planner_output.get("planner_reason", "")
        _, h_viol, h_warn = validate_hallucination_guardrail(state, "planner", f"{action} {reason}")
        violations.extend(h_viol)
        warnings.extend(h_warn)

        # Risk
        dec = planner_output.get("proposed_decision", "")
        _, r_viol, r_warn = validate_risk_guardrail(state, dec, prio)
        violations.extend(r_viol)
        warnings.extend(r_warn)

        passed = len(violations) == 0

        status_entry = {
            "guardrail_name": "Planner Constraints Guardrail (Policy + Risk + Anti-Hallucination)",
            "passed": passed,
            "violations": violations,
            "warnings": warnings,
            "timestamp": datetime.now().isoformat()
        }

        current_status = dict(state.get("guardrail_status") or {})
        current_status["planner_guardrail"] = status_entry

        all_violations = list(state.get("guardrail_violations") or []) + violations
        all_warnings = list(state.get("guardrail_warnings") or []) + warnings

        return {
            "guardrail_status": current_status,
            "guardrail_violations": all_violations,
            "guardrail_warnings": all_warnings,
        }

    @staticmethod
    def run_execution_guardrail(state: dict[str, Any]) -> dict[str, Any]:
        """Run Guardrail 9 (Execution Guardrail)."""
        is_auth, auth_token, violations, warnings = validate_execution_guardrail(state)

        status_entry = {
            "guardrail_name": "6. Execution Authorization Guardrail",
            "passed": is_auth,
            "authorization_token": auth_token if is_auth else None,
            "violations": violations,
            "warnings": warnings,
            "timestamp": datetime.now().isoformat()
        }

        current_status = dict(state.get("guardrail_status") or {})
        current_status["execution_guardrail"] = status_entry

        all_violations = list(state.get("guardrail_violations") or []) + violations
        all_warnings = list(state.get("guardrail_warnings") or []) + warnings

        return {
            "execution_authorized": is_auth,
            "authorization_token": auth_token,
            "execution_blocked_reason": violations[0] if violations else None,
            "guardrail_status": current_status,
            "guardrail_violations": all_violations,
            "guardrail_warnings": all_warnings,
        }
