"""
agents/planner.py

Planner Agent for the Telecom Complaint Agentic AI system.

The Planner receives the outputs of:

    Diagnosis Agent
    Policy Agent
    Risk Agent

and creates a proposed intervention decision.

IMPORTANT:

The Planner does NOT make the final trusted decision.

The proposed decision MUST be reviewed by the Critic.

Architecture:

    Diagnosis ──┐
                │
    Policy ─────┼──► Planner ──► Critic
                │
    Risk ───────┘
"""


import json

from agents.high.llm.qwen import ask_qwen
from agents.high.retrieval.historical import get_recommendations_for_issue, assess_complaint_multidimensional_impact


# ============================================================
# PLANNER AGENT
# ============================================================

def planner_agent(state: dict) -> dict:
    """
    Combine Diagnosis, Policy and Risk outputs with historical resolution
    insights and impact timeline rules.

    Parameters
    ----------
    state : dict
        Current LangGraph state.

    Returns
    -------
    dict
        Proposed intervention decision.
    """

    print()
    print("=" * 60)
    print("NODE: PLANNER AGENT")
    print("=" * 60)


    # ========================================================
    # 0. GET COMPLAINT CONTEXT & MULTIDIMENSIONAL IMPACT
    # ========================================================

    complaint_id = state.get("complaint_id", "UNKNOWN")
    complaint_text = state.get("complaint_text", state.get("complaint", "UNKNOWN"))
    domain = state.get("domain", "UNKNOWN")
    problem_type = state.get("problem_type", "UNKNOWN")
    issue_signature = state.get("issue_signature", "UNKNOWN")
    severity = state.get("severity", "UNKNOWN")
    scope = state.get("scope", "individual")

    # Multidimensional Impact & Timeline Assessment
    impact_assessment = assess_complaint_multidimensional_impact(state)
    impact_level = impact_assessment["impact_level"]
    impact_reason = impact_assessment["impact_summary"]
    critical_triggers = impact_assessment["critical_triggers"]
    affected_subscribers = impact_assessment["affected_subscribers"]
    days_unresolved = impact_assessment["days_unresolved"]
    duration_hours = impact_assessment["duration_hours"]

    # Fetch historical recommendations and future impact estimate
    hist_match = get_recommendations_for_issue(
        query_text=complaint_text,
        domain=domain,
        problem_type=problem_type
    )
    future_impact_days = state.get("future_impact_days") or hist_match.get("future_impact_days", 5)
    company_recommendations = hist_match.get("company_recommendations", [])


    # ========================================================
    # 1. GET DIAGNOSIS RESULT
    # ========================================================

    diagnosis = state.get(
        "diagnosis",
        "UNKNOWN"
    )

    root_cause = state.get(
        "root_cause",
        state.get("diagnosis_root_cause", "UNKNOWN")
    )

    diagnosis_confidence = state.get(
        "diagnosis_confidence",
        0.0
    )


    # ========================================================
    # 2. GET POLICY RESULT
    # ========================================================

    policy_status = state.get(
        "policy_status",
        "UNKNOWN"
    )

    policy_reason = state.get(
        "policy_reason",
        "UNKNOWN"
    )

    policy_confidence = state.get(
        "policy_confidence",
        0.0
    )


    # ========================================================
    # 3. GET RISK RESULT
    # ========================================================

    risk_level = state.get(
        "risk_level",
        "UNKNOWN"
    )

    risk_reason = state.get(
        "risk_reason",
        "UNKNOWN"
    )

    risk_confidence = state.get(
        "risk_confidence",
        0.0
    )


    # ========================================================
    # 4. SYSTEM PROMPT
    # ========================================================

    system_prompt = """
You are the Planner Agent in a telecom customer complaint decision-making system.

Your job is to combine three independent analyses:
1. Diagnosis Agent
2. Policy Agent
3. Risk Agent

You must create a PROPOSED intervention decision.

IMPORTANT:
The Planner is NOT the final decision maker.
A separate Critic Agent will review your proposal.

RULES ON PRIORITY AND CRITICAL EVALUATION:

1. CRITICAL Priority Criteria (Move to CRITICAL):
   - Larger area problems with physical infrastructure damage (such as damaged network towers, severed main cables, or power station failures) that require immediate HUMAN ON-SITE FIELD INTERVENTION / technician dispatch to restore area service.
   - 7-DAY UNSOLVED RULE: If the complaint has been ongoing / unresolved for 7 or more days (days_unresolved >= 7 or duration >= 168 hours), it MUST be escalated to CRITICAL priority (Move to Critical = YES) as it has breached the maximum 7-day unresolved SLA threshold.
   - Active life-safety / 911 emergency service disruptions.
   - Catastrophic regional blackout or multi-region infrastructure failure.
   - When policy_status is "CRITICAL" or risk_level is "CRITICAL".

2. HIGH Priority Criteria:
   - Severe connection issues, localized degraded performance, or elevated policy complaints that do NOT involve physical destruction or emergency area-wide human field deployment, and have NOT exceeded 7 days of unresolved status.

3. Operational Actions:
   - For CRITICAL area issues, specify the urgent human on-site field intervention (e.g. emergency dispatch of field engineering crew with replacement equipment).
   - For 7-day overdue issues, specify senior engineering dispatch and executive retention intervention.
   - Do NOT invent specific technical root causes (root cause is UNKNOWN).

Allowed proposed_decision values:
NORMAL
ESCALATE
CRITICAL

Allowed priority values:
LOW
MEDIUM
HIGH
CRITICAL

Return ONLY one JSON object.

Required format:
{
    "proposed_decision": "CRITICAL",
    "priority": "CRITICAL",
    "proposed_action": "Urgent Human Field Intervention: Dispatch emergency field engineering crew to inspect and repair damaged network tower infrastructure and restore area service",
    "reason": "Larger area outage caused by physical network tower damage requiring immediate on-site human field intervention.",
    "planner_confidence": 0.98
}
"""


    # ========================================================
    # 5. USER PROMPT
    # ========================================================

    user_prompt = f"""
Create a proposed intervention decision using the following complaint context, timeline metrics, and agent outputs.

========================
COMPLAINT CONTEXT & TIMELINE
========================

Complaint ID:
{complaint_id}

Complaint Text:
{complaint_text}

Domain:
{domain}

Problem Type:
{problem_type}

Issue Signature:
{issue_signature}

Severity:
{severity}

Scope:
{scope}

Days Unresolved / Open:
{days_unresolved}

Duration (Hours):
{duration_hours}

Estimated Future Impact (Days):
{future_impact_days}

========================
DIAGNOSIS AGENT
========================

Diagnosis:
{diagnosis}

Root Cause:
{root_cause}

Diagnosis Confidence:
{diagnosis_confidence}


========================
POLICY AGENT
========================

Policy Status:
{policy_status}

Policy Reason:
{policy_reason}

Policy Confidence:
{policy_confidence}


========================
RISK AGENT
========================

Risk Level:
{risk_level}

Risk Reason:
{risk_reason}

Risk Confidence:
{risk_confidence}


========================
IMPORTANT
========================

The diagnosis says what the complaint is.

The policy result indicates whether elevated policy attention
may be required.

The risk result indicates the overall complaint risk.

Combine these signals conservatively.

The root cause is UNKNOWN, so do not invent a technical root
cause.

The proposed decision will later be reviewed by the Critic.

Return ONLY the required JSON object.
"""


    # ========================================================
    # 6. CALL QWEN
    # ========================================================

    response = ask_qwen(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=700,
        temperature=0.0,
        structured=True
    )


    # ========================================================
    # 7. PARSE JSON
    # ========================================================

    try:

        planner_result = json.loads(response)

    except json.JSONDecodeError as e:

        print()
        print("❌ Planner returned invalid JSON.")

        print("Qwen response:")
        print(response)

        raise ValueError(
            "Planner Agent returned invalid JSON."
        ) from e


    # ========================================================
    # 8. REQUIRED FIELDS
    # ========================================================

    required_fields = [
        "proposed_decision",
        "priority",
        "proposed_action",
        "reason",
        "planner_confidence"
    ]

    for field in required_fields:

        if field not in planner_result:

            raise ValueError(
                f"Planner response is missing "
                f"required field: {field}"
            )


    # ========================================================
    # 9. VALIDATE DECISION
    # ========================================================

    allowed_decisions = {
        "NORMAL",
        "ESCALATE",
        "CRITICAL"
    }

    if planner_result["proposed_decision"] not in allowed_decisions:

        raise ValueError(
            "Invalid proposed decision: "
            f"{planner_result['proposed_decision']}"
        )


    # ========================================================
    # 10. DISPLAY RESULT
    # ========================================================

    print()
    print("Planner result:")

    print(
        json.dumps(
            planner_result,
            indent=2
        )
    )


    # ========================================================
    # 11. RETURN STATE UPDATE
    # ========================================================

    return {
        "proposed_decision": planner_result["proposed_decision"],
        "priority": planner_result["priority"],
        "proposed_action": planner_result["proposed_action"],
        "planner_reason": planner_result["reason"],
        "planner_confidence": planner_result["planner_confidence"],
        "days_unresolved": days_unresolved,
        "future_impact_days": future_impact_days,
        "impact_level": impact_level,
        "impact_reason": impact_reason,
        "critical_triggers": critical_triggers,
        "affected_subscribers": affected_subscribers,
        "company_recommendations": company_recommendations
    }