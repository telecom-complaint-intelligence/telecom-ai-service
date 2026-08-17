"""
graph.py

Enterprise Multi-Agent LangGraph Workflow with 9 Guardrails for Telecom Complaint Intervention:

Workflow Architecture:

                  START
                    │
                    ▼
          ┌─────────────────────┐
          │ 🛡️ INPUT GUARDRAIL   │ (Guardrail 1: Validate input & data)
          └──────────┬──────────┘
                     │
          ┌──────────┼──────────┐
          │          │          │
          ▼          ▼          ▼
     Diagnosis    Policy      Risk
          │          │          │
          └──────────┼──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ 🛡️ AGENT GUARDRAIL  │ (Guardrails 2 & 3: Output Schema & Confidence)
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │   PLANNER AGENT     │ (Guardrails 3, 4, 6: Policy, Risk, Constraints)
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │   CRITIC AGENT      │ (Guardrail 4: Safety, Policy & Anti-Hallucination)
          └──────────┬──────────┘
                     │
              Is it valid?
               ┌─────┴─────┐
              YES          NO
               │            │
               │            ▼
               │      ┌───────────┐
               │      │  REPLAN   │ (Guardrail 5: Retry limit)
               │      └─────┬─────┘
               │            │ (loops back to Critic)
               │            ▼
               │         CRITIC
               │            │
               │       ┌────┴────┐
               │      YES        NO
               │       │          │
               │       │          ▼
               │       │   ┌──────────────┐
               │       │   │ HUMAN REVIEW │ (Guardrail 8: Human-in-the-loop)
               │       │   └──────┬───────┘
               │       │          │
               └───────┼──────────┘
                       │
                       ▼
          ┌─────────────────────────┐
          │      FINALIZE NODE      │
          └────────────┬────────────┘
                       │
                       ▼
          ┌─────────────────────────┐
          │ 🛡️ EXECUTION GUARDRAIL   │ (Guardrail 9: Execution Authorization)
          └────────────┬────────────┘
                       │
                 Authorized?
                  ┌────┴────┐
                 YES        NO
                  │          │
                  ▼          ▼
            ┌───────────┐   END
            │ EXECUTION │
            └─────┬─────┘
                  │
                  ▼
                 END
"""

import sys

# Configure UTF-8 encoding for Windows standard streams
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from langgraph.graph import StateGraph, START, END

from app.agents.high.state import ComplaintState
from app.agents.high.config import MAX_REPLANS, ENABLE_EXECUTION

from app.agents.high.agents.diagnosis import diagnosis_agent
from app.agents.high.agents.policy import policy_agent
from app.agents.high.agents.risk import risk_agent
from app.agents.high.agents.planner import planner_agent
from app.agents.high.agents.critic import critic_agent
from app.agents.high.agents.replan import replan_agent

from app.agents.high.guardrails.engine import GuardrailsEngine
from app.agents.high.guardrails.retry_guardrail import validate_retry_guardrail
from app.agents.high.guardrails.human_guardrail import assemble_human_review_dossier
from app.agents.high.tools.execution import execute_telecom_intervention


# ============================================================
# 1. INPUT GUARDRAIL NODE (GUARDRAIL 1)
# ============================================================

def input_guardrail_node(state: ComplaintState):
    print()
    print("=" * 70)
    print("GRAPH NODE: 🛡️ GUARDRAIL 1 - INPUT & DATA VALIDATION")
    print("=" * 70)

    result = GuardrailsEngine.run_input_guardrail(state)
    
    if result.get("input_valid"):
        print("✅ Input Guardrail: Complaint payload & historical metrics verified.")
    else:
        print("⚠️ Input Guardrail: Violations detected:")
        for v in result.get("guardrail_violations", []):
            print(f"   - {v}")

    return {
        "guardrail_status": result["guardrail_status"],
        "guardrail_violations": result["guardrail_violations"],
        "guardrail_warnings": result["guardrail_warnings"],
        "current_node": "input_guardrail",
    }


# ============================================================
# 2. PARALLEL ANALYSIS NODES
# ============================================================

def diagnosis_node(state: ComplaintState):
    print()
    print("=" * 70)
    print("GRAPH NODE: DIAGNOSIS AGENT")
    print("=" * 70)
    return diagnosis_agent(state)


def policy_node(state: ComplaintState):
    print()
    print("=" * 70)
    print("GRAPH NODE: POLICY AGENT")
    print("=" * 70)
    return policy_agent(state)


def risk_node(state: ComplaintState):
    print()
    print("=" * 70)
    print("GRAPH NODE: RISK AGENT")
    print("=" * 70)
    return risk_agent(state)


# ============================================================
# 3. MULTI-AGENT OUTPUT & CONFIDENCE GUARDRAILS (GUARDRAILS 2 & 3)
# ============================================================

def multi_agent_guardrail_node(state: ComplaintState):
    print()
    print("=" * 70)
    print("GRAPH NODE: 🛡️ GUARDRAILS 2 & 3 - AGENT OUTPUT & CONFIDENCE VERIFICATION")
    print("=" * 70)

    result = GuardrailsEngine.run_multi_agent_guardrail(state)
    status_entry = result["guardrail_status"].get("multi_agent_guardrail", {})

    if status_entry.get("passed"):
        print("✅ Multi-Agent Guardrail: Schemas and confidence scores validated.")
    else:
        print("⚠️ Multi-Agent Guardrail: Warnings/Violations noted:")
        for w in status_entry.get("warnings", []):
            print(f"   ⚡ {w}")

    return {
        "guardrail_status": result["guardrail_status"],
        "guardrail_violations": result["guardrail_violations"],
        "guardrail_warnings": result["guardrail_warnings"],
        "current_node": "multi_agent_guardrail",
    }


# ============================================================
# 4. PLANNER NODE (GUARDRAIL 3 & CONSTRAINTS)
# ============================================================

def planner_node(state: ComplaintState):
    print()
    print("=" * 70)
    print("GRAPH NODE: INTERVENTION PLANNER AGENT")
    print("=" * 70)

    planner_output = planner_agent(state)
    guardrail_res = GuardrailsEngine.run_planner_guardrails(state, planner_output)

    planner_output["guardrail_status"] = guardrail_res["guardrail_status"]
    planner_output["guardrail_violations"] = guardrail_res["guardrail_violations"]
    planner_output["guardrail_warnings"] = guardrail_res["guardrail_warnings"]
    planner_output["current_node"] = "planner"

    return planner_output


# ============================================================
# 5. CRITIC NODE (GUARDRAIL 4)
# ============================================================

def critic_node(state: ComplaintState):
    print()
    print("=" * 70)
    print("GRAPH NODE: CRITIC AGENT (SAFETY & COMPLIANCE QA)")
    print("=" * 70)

    critic_output = critic_agent(state)
    critic_output["current_node"] = "critic"

    return critic_output


# ============================================================
# 6. REPLAN NODE (GUARDRAIL 5: RETRY LIMIT)
# ============================================================

def replan_node(state: ComplaintState):
    print()
    print("=" * 70)
    print("GRAPH NODE: REPLAN AGENT (GUARDRAIL 5: RETRY CONTROLLER)")
    print("=" * 70)

    result = replan_agent(state)

    # Increment retry count
    current_retries = int(state.get("retry_count", 0)) + 1
    result["retry_count"] = current_retries
    result["replan_required"] = False
    result["current_node"] = "replan"

    print(f"→ Replan cycle {current_retries} completed. Routing back to Critic for re-validation.")

    return result


# ============================================================
# 7. HUMAN REVIEW NODE (GUARDRAIL 8: HUMAN-IN-THE-LOOP)
# ============================================================

def human_review_node(state: ComplaintState):
    print()
    print("=" * 70)
    print("GRAPH NODE: 🧑‍💼 GUARDRAIL 8 - HUMAN ESCALATION & MANUAL REVIEW")
    print("=" * 70)

    escalation_reason = (
        state.get("critic_reason") or
        "Automated AI agents exhausted maximum replan cycles without consensus."
    )
    violations = state.get("guardrail_violations") or []

    dossier = assemble_human_review_dossier(
        state=state,
        escalation_reason=escalation_reason,
        violations=violations
    )

    print("🚨 Case escalated to Human Operations Engineer:")
    print(f"   Dossier ID : {dossier['dossier_id']}")
    print(f"   Reason     : {escalation_reason}")
    print("   Action     : Manual operator review scheduled.")

    return {
        "human_review_required": True,
        "human_review_reason": escalation_reason,
        "human_review_dossier": dossier,
        "proposed_decision": "ESCALATE",
        "priority": "HIGH" if str(state.get("priority", "")).upper() != "CRITICAL" else "CRITICAL",
        "proposed_action": f"Human Operations Review: {dossier['suggested_human_action']}",
        "planner_reason": f"Escalated to human supervisor after {state.get('retry_count', 0)} replan attempts.",
        "current_node": "human_review",
    }


# ============================================================
# 8. FINALIZE NODE
# ============================================================

def finalize_node(state: ComplaintState):
    print()
    print("=" * 70)
    print("GRAPH NODE: FINALIZE RESOLUTION")
    print("=" * 70)

    # Determine final decision fields
    if state.get("human_review_required"):
        final_dec = "ESCALATE"
        final_prio = state.get("priority", "HIGH")
        final_act = state.get("proposed_action", "Manual human triage in progress.")
        final_reas = state.get("human_review_reason", "Escalated for human sign-off.")
        final_conf = 0.99
    elif state.get("revised_decision"):
        final_dec = state.get("revised_decision")
        final_prio = state.get("revised_priority")
        final_act = state.get("revised_action")
        final_reas = state.get("revised_reason")
        final_conf = state.get("replan_confidence")
    else:
        final_dec = state.get("proposed_decision")
        final_prio = state.get("priority")
        final_act = state.get("proposed_action")
        final_reas = state.get("planner_reason")
        final_conf = state.get("planner_confidence")

    # Evaluate whether case was moved to CRITICAL
    moved_to_crit = (
        str(final_prio).strip().upper() == "CRITICAL" or
        str(final_dec).strip().upper() == "CRITICAL"
    )

    return {
        "final_decision": final_dec,
        "final_priority": final_prio,
        "final_action": final_act,
        "final_reason": final_reas,
        "final_confidence": final_conf,
        "moved_to_critical": moved_to_crit,
        "days_unresolved": state.get("days_unresolved", 1),
        "future_impact_days": state.get("future_impact_days", 5),
        "impact_level": state.get("impact_level", "CRITICAL" if moved_to_crit else "HIGH"),
        "impact_reason": state.get("impact_reason", ""),
        "critical_triggers": state.get("critical_triggers", []),
        "affected_subscribers": state.get("affected_subscribers", 1),
        "company_recommendations": state.get("company_recommendations", []),
        "current_node": "finalize",
    }


# ============================================================
# 9. EXECUTION GUARDRAIL NODE (GUARDRAIL 9)
# ============================================================

def execution_guardrail_node(state: ComplaintState):
    print()
    print("=" * 70)
    print("GRAPH NODE: 🛡️ GUARDRAIL 9 - EXECUTION AUTHORIZATION GATEKEEPER")
    print("=" * 70)

    result = GuardrailsEngine.run_execution_guardrail(state)

    if result.get("execution_authorized"):
        print(f"✅ Execution Guardrail: Authorized (Token: {result.get('authorization_token')})")
    else:
        print(f"❌ Execution Guardrail: Blocked - {result.get('execution_blocked_reason')}")

    return {
        "execution_authorized": result["execution_authorized"],
        "authorization_token": result["authorization_token"],
        "execution_blocked_reason": result["execution_blocked_reason"],
        "guardrail_status": result["guardrail_status"],
        "guardrail_violations": result["guardrail_violations"],
        "guardrail_warnings": result["guardrail_warnings"],
        "current_node": "execution_guardrail",
    }


# ============================================================
# 10. EXECUTION NODE (OPERATIONAL DISPATCH)
# ============================================================

def execution_node(state: ComplaintState):
    print()
    print("=" * 70)
    print("GRAPH NODE: ⚡ INTERVENTION EXECUTION")
    print("=" * 70)

    execution_receipt = execute_telecom_intervention(state)

    return {
        "execution_result": execution_receipt,
        "execution_timestamp": execution_receipt.get("timestamp"),
        "current_node": "execution",
    }


# ============================================================
# 11. ROUTING LOGIC
# ============================================================

def route_after_critic(state: ComplaintState) -> str:
    """
    Deterministic router based on structured Critic output & Retry Guardrail.
    """
    raw_decision = state.get("critic_decision")
    decision_str = str(raw_decision).strip().upper() if raw_decision is not None else ""

    raw_replan = state.get("replan_required")
    if isinstance(raw_replan, bool):
        replan_needed = raw_replan
    elif isinstance(raw_replan, str):
        replan_needed = raw_replan.strip().lower() in ("true", "1", "yes")
    else:
        replan_needed = False

    retry_count = int(state.get("retry_count", 0))
    max_retries = int(state.get("max_retries", MAX_REPLANS))

    can_retry, should_escalate, _, _ = validate_retry_guardrail(state, max_retries)

    print()
    print("=" * 70)
    print("CRITIC ROUTER (GUARDRAILS 4 & 5 EVALUATION)")
    print("=" * 70)
    print(f"Critic decision : {decision_str}")
    print(f"Replan required : {replan_needed}")
    print(f"Retry count     : {retry_count} / {max_retries}")

    # Case 1: Critic accepted AND replan is not required -> APPROVED
    if decision_str == "ACCEPT" and not replan_needed:
        print("→ Decision APPROVED by Critic")
        print("→ Routing: CRITIC → FINALIZE")
        return "finalize"

    # Case 2: Critic rejected OR replan required
    if decision_str == "REJECT" or replan_needed:
        if can_retry:
            print("→ Critic REJECTED proposal; Replan permitted under Retry Guardrail.")
            print("→ Routing: CRITIC → REPLAN")
            return "replan"
        else:
            print("→ Maximum replan retries exceeded.")
            print("→ Routing: CRITIC → HUMAN REVIEW (Escalation Guardrail)")
            return "human_review"

    # Default fallback
    print("→ Routing fallback to FINALIZE")
    return "finalize"


def route_after_execution_guardrail(state: ComplaintState) -> str:
    """
    Route to operational execution node if authorized and enabled, otherwise finish.
    """
    is_authorized = state.get("execution_authorized", False)
    
    if is_authorized and ENABLE_EXECUTION:
        print("→ Execution Guardrail PASSED: Routing to EXECUTION node")
        return "execution"
    else:
        print("→ Execution Guardrail BLOCKED or disabled: Routing to END")
        return "end"


# ============================================================
# 12. BUILD LANGGRAPH WORKFLOW
# ============================================================

def build_graph():
    print()
    print("=" * 70)
    print("BUILDING TELECOM COMPLAINT LANGGRAPH WITH 9 GUARDRAILS")
    print("=" * 70)

    workflow = StateGraph(ComplaintState)

    # Add Nodes
    workflow.add_node("input_guardrail", input_guardrail_node)
    workflow.add_node("diagnosis", diagnosis_node)
    workflow.add_node("policy", policy_node)
    workflow.add_node("risk", risk_node)
    workflow.add_node("multi_agent_guardrail", multi_agent_guardrail_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("replan", replan_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("finalize", finalize_node)
    workflow.add_node("execution_guardrail", execution_guardrail_node)
    workflow.add_node("execution", execution_node)

    # 1. START -> Input Guardrail
    workflow.add_edge(START, "input_guardrail")

    # 2. Input Guardrail -> Parallel Agents
    workflow.add_edge("input_guardrail", "diagnosis")
    workflow.add_edge("input_guardrail", "policy")
    workflow.add_edge("input_guardrail", "risk")

    # 3. Parallel Agents -> Multi-Agent Guardrail
    workflow.add_edge("diagnosis", "multi_agent_guardrail")
    workflow.add_edge("policy", "multi_agent_guardrail")
    workflow.add_edge("risk", "multi_agent_guardrail")

    # 4. Multi-Agent Guardrail -> Planner
    workflow.add_edge("multi_agent_guardrail", "planner")

    # 5. Planner -> Critic
    workflow.add_edge("planner", "critic")

    # 6. Critic -> Conditional Router (Accept -> Finalize, Reject -> Replan / Human Review)
    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "finalize": "finalize",
            "replan": "replan",
            "human_review": "human_review",
        }
    )

    # 7. Replan -> Loops back to Critic!
    workflow.add_edge("replan", "critic")

    # 8. Human Review -> Finalize
    workflow.add_edge("human_review", "finalize")

    # 9. Finalize -> Execution Guardrail
    workflow.add_edge("finalize", "execution_guardrail")

    # 10. Execution Guardrail -> Execution or END
    workflow.add_conditional_edges(
        "execution_guardrail",
        route_after_execution_guardrail,
        {
            "execution": "execution",
            "end": END
        }
    )

    # 11. Execution -> END
    workflow.add_edge("execution", END)

    graph = workflow.compile()
    print("✅ LangGraph compiled successfully with 9 Enterprise Guardrails.")

    return graph


graph = build_graph()