"""
graph.py

Main LangGraph workflow for the Telecom Complaint
Intervention Agent.

Architecture:

                    ┌── Diagnosis ──┐
                    │               │
Complaint ──────────┼── Policy ─────┼──> Planner ──> Critic
                    │               │                    │
                    └── Risk ───────┘                    │
                                                   ┌─────┴─────┐
                                                   │           │
                                                ACCEPT       REJECT
                                                   │           │
                                                  END        Replan
                                                               │
                                                              END
"""

import contextlib
import sys

# Configure UTF-8 encoding for Windows standard streams
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        with contextlib.suppress(Exception):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        with contextlib.suppress(Exception):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from langgraph.graph import END, START, StateGraph

from app.agents.high.agents.critic import critic_agent
from app.agents.high.agents.diagnosis import diagnosis_agent
from app.agents.high.agents.planner import planner_agent
from app.agents.high.agents.policy import policy_agent
from app.agents.high.agents.replan import replan_agent
from app.agents.high.agents.risk import risk_agent
from app.agents.high.state import ComplaintState
from guardrails.engine import GuardrailsEngine

# ============================================================
# 1. DIAGNOSIS NODE
# ============================================================

def diagnosis_node(state: ComplaintState):

    print()
    print("=" * 70)
    print("GRAPH NODE: DIAGNOSIS (Guardrail 1: Input Validation)")
    print("=" * 70)

    # Execute Guardrail 1: Input Guardrail
    input_guard_res = GuardrailsEngine.run_input_guardrail(state)

    result = diagnosis_agent(state)
    result.update(input_guard_res)

    return result


# ============================================================
# 2. POLICY NODE
# ============================================================

def policy_node(state: ComplaintState):

    print()
    print("=" * 70)
    print("GRAPH NODE: POLICY")
    print("=" * 70)

    result = policy_agent(state)

    return result


# ============================================================
# 3. RISK NODE
# ============================================================

def risk_node(state: ComplaintState):

    print()
    print("=" * 70)
    print("GRAPH NODE: RISK")
    print("=" * 70)

    result = risk_agent(state)

    return result


# ============================================================
# 4. PLANNER NODE
# ============================================================

def planner_node(state: ComplaintState):

    print()
    print("=" * 70)
    print("GRAPH NODE: PLANNER (Guardrails 2-6: Output, Confidence & Constraints)")
    print("=" * 70)

    # Run Guardrails 2 & 3: Output & Confidence check on Diagnosis, Policy, Risk
    multi_guard_res = GuardrailsEngine.run_multi_agent_guardrail(state)

    result = planner_agent(state)

    # Run Guardrails 4, 5, 6: Policy, Hallucination, and Risk Constraints on Planner
    planner_guard_res = GuardrailsEngine.run_planner_guardrails(state, result)

    result.update(multi_guard_res)
    result["guardrail_status"] = {
        **multi_guard_res.get("guardrail_status", {}),
        **planner_guard_res.get("guardrail_status", {}),
    }
    result["guardrail_violations"] = (
        multi_guard_res.get("guardrail_violations", []) +
        planner_guard_res.get("guardrail_violations", [])
    )
    result["guardrail_warnings"] = (
        multi_guard_res.get("guardrail_warnings", []) +
        planner_guard_res.get("guardrail_warnings", [])
    )

    return result


# ============================================================
# 5. CRITIC NODE
# ============================================================

def critic_node(state: ComplaintState):

    print()
    print("=" * 70)
    print("GRAPH NODE: CRITIC")
    print("=" * 70)

    result = critic_agent(state)

    return result


# ============================================================
# 6. REPLAN NODE
# ============================================================

def replan_node(state: ComplaintState):

    print()
    print("=" * 70)
    print("GRAPH NODE: REPLAN")
    print("=" * 70)

    result = replan_agent(state)

    # Ensure retry count is updated and replan flag is cleared
    result["retry_count"] = state.get("retry_count", 0) + 1
    result["replan_required"] = False
    result["current_node"] = "replan"

    return result


# ============================================================
# 7. FINALIZE NODE
# ============================================================

def finalize_node(state: ComplaintState):

    print()
    print("=" * 70)
    print("GRAPH NODE: FINALIZE (Guardrail 9: Execution Authorization)")
    print("=" * 70)

    # Determine final fields based on whether Replan was executed
    if state.get("revised_decision"):
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

    # Execute Guardrail 9: Execution Authorization
    exec_state = {
        **state,
        "final_decision": final_dec,
        "final_action": final_act,
    }
    exec_res = GuardrailsEngine.run_execution_guardrail(exec_state)

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
        "execution_token": exec_res.get("authorization_token"),
        "guardrail_status": {
            **state.get("guardrail_status", {}),
            **exec_res.get("guardrail_status", {}),
        },
        "guardrail_violations": exec_res.get("guardrail_violations", []),
        "guardrail_warnings": exec_res.get("guardrail_warnings", []),
        "current_node": "finalize",
    }



# ============================================================
# 8. CRITIC ROUTER
# ============================================================

def route_after_critic(state: ComplaintState) -> str:
    """
    Deterministic Python router strictly based on structured Critic output.
    Does NOT use an LLM for routing.

    Routing rules:
    - If critic_decision == "ACCEPT" and not replan_required -> "finalize"
    - If critic_decision == "REJECT" or replan_required is True -> "replan" (if retry_count < max_retries)
    - Otherwise -> "finalize"
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
    max_retries = int(state.get("max_retries", 1))

    print()
    print("=" * 70)
    print("CRITIC ROUTER")
    print("=" * 70)
    print(f"Critic decision : {decision_str}")
    print(f"Replan required : {replan_needed}")
    print(f"Retry count     : {retry_count} / {max_retries}")

    # Case 1: Critic accepted AND replan is not required
    if decision_str == "ACCEPT" and not replan_needed:

        print("→ Decision ACCEPTED by Critic without replan")
        print("→ Routing directly: CRITIC → FINALIZE (REPLAN will NOT execute)")

        return "finalize"

    # Case 2: Critic rejected OR replan required
    if decision_str == "REJECT" or replan_needed:

        if retry_count < max_retries:

            print("→ Critic REJECTED decision (replan required)")
            print("→ Routing: CRITIC → REPLAN")

            return "replan"

        print("→ Maximum replans reached")
        print("→ Routing: CRITIC → FINALIZE")

        return "finalize"

    # Case 3: Default fallback
    print("→ Defaulting to FINALIZE")

    return "finalize"


# ============================================================
# 9. BUILD GRAPH
# ============================================================

def build_graph():

    print()
    print("=" * 70)
    print("BUILDING TELECOM COMPLAINT LANGGRAPH")
    print("=" * 70)


    # --------------------------------------------------------
    # Create graph
    # --------------------------------------------------------

    workflow = StateGraph(ComplaintState)


    # ========================================================
    # ADD NODES
    # ========================================================

    workflow.add_node("diagnosis", diagnosis_node)

    workflow.add_node("policy", policy_node)

    workflow.add_node("risk", risk_node)

    workflow.add_node("planner", planner_node)

    workflow.add_node("critic", critic_node)

    workflow.add_node("replan", replan_node)

    workflow.add_node("finalize", finalize_node)


    # ========================================================
    # PARALLEL START
    # ========================================================

    workflow.add_edge(START, "diagnosis")

    workflow.add_edge(START, "policy")

    workflow.add_edge(START, "risk")


    # ========================================================
    # WAIT FOR ALL THREE PARALLEL AGENTS
    # ========================================================

    workflow.add_edge("diagnosis", "planner")

    workflow.add_edge("policy", "planner")

    workflow.add_edge("risk", "planner")


    # ========================================================
    # PLANNER → CRITIC
    # ========================================================

    workflow.add_edge("planner", "critic")


    # ========================================================
    # CRITIC → ACCEPT / REPLAN / FINALIZE
    # ========================================================

    workflow.add_conditional_edges(

        "critic",

        route_after_critic,

        {
            "finalize": "finalize",
            "replan": "replan",
        }

    )


    # ========================================================
    # REPLAN → FINALIZE
    # ========================================================

    workflow.add_edge("replan", "finalize")


    # ========================================================
    # FINALIZE → END
    # ========================================================

    workflow.add_edge("finalize", END)


    # ========================================================
    # COMPILE
    # ========================================================

    graph = workflow.compile()


    print()
    print("✅ LangGraph compiled successfully.")

    return graph


# ============================================================
# 10. CREATE GRAPH
# ============================================================

graph = build_graph()