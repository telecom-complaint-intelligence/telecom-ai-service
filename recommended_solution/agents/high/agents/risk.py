"""
agents/risk.py

Risk Agent for the Telecom Complaint Agentic AI system.

Responsibilities:
    - Evaluate the risk level of the complaint.
    - Use only the supplied complaint evidence.
    - Consider severity, recurrence, previous complaints,
      sentiment, and historical context.
    - Do NOT make the final intervention decision.

Architecture:

    Diagnosis Agent ─┐
    Policy Agent ────┼──► Planner
    Risk Agent ──────┘
"""


import json

from agents.high.llm.qwen import ask_qwen


# ============================================================
# RISK AGENT
# ============================================================

def risk_agent(state: dict) -> dict:
    """
    Analyze the complaint risk.

    Parameters
    ----------
    state : dict
        Current LangGraph state.

    Returns
    -------
    dict
        Risk information to add to LangGraph state.
    """

    print()
    print("=" * 60)
    print("NODE: RISK AGENT")
    print("=" * 60)


    # ========================================================
    # 1. READ COMPLAINT DATA
    # ========================================================

    complaint_id = state.get(
        "complaint_id",
        "UNKNOWN"
    )

    severity = state.get(
        "severity",
        "UNKNOWN"
    )

    historical_occurrences = state.get(
        "historical_occurrences",
        0
    )

    occurrences_last_30_days = state.get(
        "occurrences_last_30_days",
        0
    )

    customer_previous_complaints = state.get(
        "customer_previous_complaints",
        0
    )

    sentiment_score = state.get(
        "sentiment_score",
        0.0
    )

    issue_signature = state.get(
        "issue_signature",
        "UNKNOWN"
    )


    # ========================================================
    # 2. RISK AGENT SYSTEM PROMPT
    # ========================================================

    system_prompt = """
You are the Risk Agent in a telecom customer complaint
decision-making system.

Your responsibility is to evaluate the risk level of a
customer complaint.

Use ONLY the evidence provided.

Consider:

1. Complaint severity.
2. Number of historical occurrences.
3. Number of occurrences during the last 30 days.
4. Number of previous customer complaints.
5. Negative sentiment score.
6. Recurrence and persistence of the issue.

Rules:

- Do not invent technical information.
- Do not invent customer information.
- Do not invent company policies.
- Do not make the final intervention decision.
- Do not create an intervention plan.
- Do not decide whether the case is CRITICAL.
- CRITICAL decision will be made later by the Planner/Critic
  workflow.

Risk levels:

LOW
MEDIUM
HIGH
UNKNOWN

Return ONLY one JSON object.

Required format:

{
    "risk_level": "HIGH",
    "risk_reason": "short evidence-based reason",
    "risk_confidence": 0.0
}

Do not provide reasoning.
Do not provide analysis.
Do not provide markdown.
"""


    # ========================================================
    # 3. USER PROMPT
    # ========================================================

    user_prompt = f"""
Evaluate the risk of this telecom complaint.

Complaint ID:
{complaint_id}

Issue Signature:
{issue_signature}

Severity:
{severity}

Historical Occurrences:
{historical_occurrences}

Occurrences in Last 30 Days:
{occurrences_last_30_days}

Customer Previous Complaints:
{customer_previous_complaints}

Negative Sentiment Score:
{sentiment_score}

Use only these values as evidence.

Return ONLY the required JSON object.
"""


    # ========================================================
    # 4. CALL QWEN
    # ========================================================

    response = ask_qwen(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=700,
        temperature=0.0,
        structured=True
    )


    # ========================================================
    # 5. PARSE RESPONSE
    # ========================================================

    try:

        risk_result = json.loads(response)

    except json.JSONDecodeError as e:

        print()
        print("❌ Risk Agent returned invalid JSON.")

        print("Qwen response:")
        print(response)

        raise ValueError(
            "Risk Agent returned invalid JSON."
        ) from e


    # ========================================================
    # 6. CHECK REQUIRED FIELDS
    # ========================================================

    required_fields = [
        "risk_level",
        "risk_reason",
        "risk_confidence"
    ]

    for field in required_fields:

        if field not in risk_result:

            raise ValueError(
                f"Risk Agent response is missing "
                f"required field: {field}"
            )


    # ========================================================
    # 7. VALIDATE RISK LEVEL
    # ========================================================

    allowed_levels = {
        "LOW",
        "MEDIUM",
        "HIGH",
        "UNKNOWN"
    }

    if risk_result["risk_level"] not in allowed_levels:

        raise ValueError(
            "Invalid risk level returned by Qwen: "
            f"{risk_result['risk_level']}"
        )


    # ========================================================
    # 8. DISPLAY RESULT
    # ========================================================

    print()
    print("Risk result:")

    print(
        json.dumps(
            risk_result,
            indent=2
        )
    )


    # ========================================================
    # 9. RETURN LANGGRAPH STATE UPDATE
    # ========================================================

    return {

        "risk_level":
            risk_result["risk_level"],

        "risk_reason":
            risk_result["risk_reason"],

        "risk_confidence":
            risk_result["risk_confidence"]
    }