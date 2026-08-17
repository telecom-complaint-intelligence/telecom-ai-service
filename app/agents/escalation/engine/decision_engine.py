import json
import re
from typing import Any

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict


class EscalationState(TypedDict):
    complaint: str
    current_severity: str
    customer_feedback: str
    solution_agent_output: Any
    category: str | None
    technical_information: dict[str, Any] | None
    complexity: str | None
    complexity_score: float | None
    weighted_negativity_score: float | None
    age_in_days: int | None
    category_complaint_count: int | None
    relevance_evaluation: str
    resolution_status: str
    llm_reasoning: str
    next_severity: str
    escalated: bool
    reasoning: str
def format_solution_output(sol: Any) -> str:
    if isinstance(sol, dict):
        return "\n".join(f"- {k}: {v}" for k, v in sol.items())
    return str(sol)
def parse_json_from_llm(output_text: str) -> dict[str, Any]:
    clean_text = output_text.strip()
    if clean_text.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", clean_text, re.DOTALL)
        if match:
            clean_text = match.group(1).strip()
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Failed to parse LLM output as JSON. Output was: {output_text}") from None
def preprocess_input(state: EscalationState) -> dict[str, Any]:
    severity = state.get("current_severity", "").strip().upper()
    if severity not in {"LOW", "MEDIUM"}:
        raise ValueError("Input severity must be LOW or MEDIUM")
    return {"current_severity": severity}
def llm_interpretation(state: EscalationState) -> dict[str, Any]:
    from app.agents.escalation.llm.llm_service import MockChatLLM, get_llm
    from app.agents.escalation.prompts.escalation_prompt import get_escalation_prompt
    llm = get_llm()
    prompt_template = get_escalation_prompt()
    formatted_solution = format_solution_output(state["solution_agent_output"])
    messages = prompt_template.format_messages(
        complaint=state["complaint"],
        current_severity=state["current_severity"],
        solution_agent_output=formatted_solution,
        customer_feedback=state["customer_feedback"]
    )
    try:
        response = llm.invoke(messages)
    except Exception as e:
        print(f"⚠️ Escalation LLM failed (likely API quota), falling back to mock: {e}")
        response = MockChatLLM().invoke(messages)
    response_content = str(response.content) if hasattr(response, "content") else str(response)
    parsed = parse_json_from_llm(response_content)
    return {
        "relevance_evaluation": parsed.get("relevance_evaluation", ""),
        "resolution_status": parsed.get("resolution_status", "unknown").strip().lower(),
        "llm_reasoning": parsed.get("reasoning", "")
    }
def deterministic_policy(state: EscalationState) -> dict[str, Any]:
    resolution = state.get("resolution_status", "unknown").lower()
    relevance = state.get("relevance_evaluation", "").lower()
    current_severity = state.get("current_severity", "LOW").upper()
    llm_reasoning = state.get("llm_reasoning", "")
    tech_info = state.get("technical_information") or {}
    scope = str(tech_info.get("scope", "")).strip().lower()
    occurrence = str(tech_info.get("occurrence_pattern", "")).strip().lower()
    complexity = str(state.get("complexity", "")).strip().upper()
    comp_score = state.get("complexity_score") or 0.0
    negativity = state.get("weighted_negativity_score") or 0.0
    age_in_days = state.get("age_in_days") or 0
    category_complaint_count = state.get("category_complaint_count") or 0
    
    escalated = False
    next_severity = current_severity
    policy_reasoning = ""
    
    if current_severity == "LOW":
        # Check HIGH triggers first for direct escalation
        if scope == "multiple_customers":
            escalated = True
            next_severity = "HIGH"
            policy_reasoning = "Direct escalation: System outage scope detected (affects multiple customers)."
        elif occurrence == "recurring":
            escalated = True
            next_severity = "HIGH"
            policy_reasoning = "Direct escalation: Recurring failure pattern detected."
        elif complexity == "CRITICAL" or comp_score >= 70.0:
            escalated = True
            next_severity = "HIGH"
            policy_reasoning = f"Direct escalation: Critical complexity score ({comp_score}) detected."
        elif negativity >= 5.0:
            escalated = True
            next_severity = "HIGH"
            policy_reasoning = f"Direct escalation: Highly frustrated customer sentiment (negativity: {negativity}) detected."
        # Check MEDIUM triggers next
        elif resolution == "unresolved" or "irrelevant" in relevance:
            escalated = True
            next_severity = "MEDIUM"
            policy_reasoning = "Escalated to MEDIUM: Complaint is unresolved or the solution was irrelevant."
        elif age_in_days >= 2:
            escalated = True
            next_severity = "MEDIUM"
            policy_reasoning = f"Escalated to MEDIUM: SLA breach threshold met (age: {age_in_days} days)."
        elif category_complaint_count >= 3:
            escalated = True
            next_severity = "MEDIUM"
            policy_reasoning = f"Escalated to MEDIUM: Widespread category cluster pattern detected (count: {category_complaint_count})."
        elif negativity >= 2.0:
            escalated = True
            next_severity = "MEDIUM"
            policy_reasoning = f"Escalated to MEDIUM: Moderately frustrated customer sentiment (negativity: {negativity}) detected."
        else:
            policy_reasoning = "No escalation criteria met. Complaint remains at LOW."
            
    elif current_severity == "MEDIUM":
        # Check HIGH triggers
        if resolution == "unresolved" or "irrelevant" in relevance:
            escalated = True
            next_severity = "HIGH"
            policy_reasoning = "Escalated to HIGH: Complaint remains unresolved or the solution was irrelevant."
        elif scope == "multiple_customers":
            escalated = True
            next_severity = "HIGH"
            policy_reasoning = "Escalated to HIGH: System outage scope detected (affects multiple customers)."
        elif occurrence == "recurring":
            escalated = True
            next_severity = "HIGH"
            policy_reasoning = "Escalated to HIGH: Recurring failure pattern detected."
        elif complexity == "CRITICAL" or comp_score >= 70.0:
            escalated = True
            next_severity = "HIGH"
            policy_reasoning = f"Escalated to HIGH: Critical complexity score ({comp_score}) detected."
        elif negativity >= 5.0:
            escalated = True
            next_severity = "HIGH"
            policy_reasoning = f"Escalated to HIGH: Highly frustrated customer sentiment (negativity: {negativity}) detected."
        elif age_in_days >= 3:
            escalated = True
            next_severity = "HIGH"
            policy_reasoning = f"Escalated to HIGH: SLA breach threshold met (age: {age_in_days} days)."
        elif category_complaint_count >= 10:
            escalated = True
            next_severity = "HIGH"
            policy_reasoning = f"Escalated to HIGH: Widespread category cluster pattern detected (count: {category_complaint_count})."
        else:
            policy_reasoning = "No escalation criteria met. Complaint remains at MEDIUM."
            
    final_reasoning = f"[Policy Decision: {next_severity}] escalated={escalated}. {policy_reasoning} LLM Insights: {llm_reasoning}"
    return {
        "escalated": escalated,
        "next_severity": next_severity,
        "reasoning": final_reasoning
    }
def guardrail_validation(state: EscalationState) -> dict[str, Any]:
    current_severity = state.get("current_severity", "").upper()
    next_severity = state.get("next_severity", "").upper()
    allowed_severities = {"LOW", "MEDIUM", "HIGH"}
    if current_severity not in {"LOW", "MEDIUM"}:
        raise ValueError(f"Guardrail violation: Invalid input severity '{current_severity}'")
    if next_severity not in allowed_severities:
        raise ValueError(f"Guardrail violation: Invalid output severity '{next_severity}'")
    if current_severity == "MEDIUM" and next_severity == "LOW":
        raise ValueError("Guardrail violation: MEDIUM severity cannot transition to LOW")
    return {}
workflow = StateGraph(EscalationState)
workflow.add_node("preprocess_input", preprocess_input)
workflow.add_node("llm_interpretation", llm_interpretation)
workflow.add_node("deterministic_policy", deterministic_policy)
workflow.add_node("guardrail_validation", guardrail_validation)
workflow.set_entry_point("preprocess_input")
workflow.add_edge("preprocess_input", "llm_interpretation")
workflow.add_edge("llm_interpretation", "deterministic_policy")
workflow.add_edge("deterministic_policy", "guardrail_validation")
workflow.add_edge("guardrail_validation", END)
decision_engine_graph = workflow.compile()
