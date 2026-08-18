"""
agents/policy.py

Policy Agent for the Telecom Complaint Agentic AI system.

Responsibilities:
    1. Receive complaint information.
    2. Analyze the complaint against policy-related rules.
    3. Determine whether the complaint requires special handling.
    4. Return structured policy information.

Important:
    The Policy Agent does NOT make the final critical decision.
    The Planner/Critic workflow will make the final intervention
    decision later.
"""

import json

from app.agents.high.llm.qwen import ask_qwen

# ============================================================
# POLICY AGENT
# ============================================================

def policy_agent(state: dict) -> dict:
    """
    Execute the Policy Agent.

    Parameters
    ----------
    state : dict
        Current complaint state.

    Returns
    -------
    dict
        Policy analysis that will be added to LangGraph state.
    """

    print()
    print("=" * 60)
    print("NODE: POLICY AGENT")
    print("=" * 60)


    # ========================================================
    # 1. READ COMPLAINT INFORMATION
    # ========================================================

    complaint_id = state.get(
        "complaint_id",
        "UNKNOWN"
    )

    domain = state.get(
        "domain",
        "UNKNOWN"
    )

    problem_type = state.get(
        "problem_type",
        "UNKNOWN"
    )

    issue_signature = state.get(
        "issue_signature",
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


    # ========================================================
    # 2. POLICY SYSTEM PROMPT
    # ========================================================
    #
    # The Policy Agent evaluates the complaint from a
    # policy perspective.
    #
    # It does NOT invent company policies.
    #
    # Since we currently have no actual telecom policy
    # document connected to the system, the model must clearly
    # distinguish evidence from assumptions.
    # ========================================================

    system_prompt = """
You are the Policy Agent in a telecom customer complaint
decision-making system.

Your responsibility is to analyze the supplied complaint
from a policy perspective.

Use ONLY the evidence provided.

Rules:

1. Do not invent company policies.

2. Do not claim that a specific company policy exists unless
   it is explicitly provided in the input.

3. Do not invent technical logs or operational information.

4. Identify whether the complaint appears to require
   elevated policy attention based on the supplied evidence.

5. Recurring complaints, high severity, and repeated customer
   complaints may indicate that the case deserves elevated
   attention.

6. Do not make the final CRITICAL decision.

7. Do not create an intervention plan.

8. Return exactly ONE JSON object.

9. Do not return reasoning.

10. Do not return explanations.

Required format:

{
    "policy_status": "ELEVATED",
    "policy_reason": "short reason",
    "policy_confidence": 0.0
}

Possible policy_status values:

NORMAL
ELEVATED
UNKNOWN

Use UNKNOWN if the available evidence is insufficient.
"""


    # ========================================================
    # 3. USER PROMPT
    # ========================================================

    user_prompt = f"""
Analyze the following telecom complaint from a policy
perspective.

Complaint ID:
{complaint_id}

Domain:
{domain}

Problem Type:
{problem_type}

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

Sentiment Score:
{sentiment_score}

There is currently no external company policy document
provided.

Therefore, do not invent a specific company policy.

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
    # 5. PARSE JSON
    # ========================================================

    try:

        policy_result = json.loads(response)

    except json.JSONDecodeError as e:

        print()
        print("❌ Policy Agent returned invalid JSON.")
        print("Qwen response:")
        print(response)

        raise ValueError(
            "Policy Agent returned invalid JSON."
        ) from e


    # ========================================================
    # 6. VALIDATE REQUIRED FIELDS
    # ========================================================

    required_fields = [
        "policy_status",
        "policy_reason",
        "policy_confidence"
    ]

    for field in required_fields:

        if field not in policy_result:

            raise ValueError(
                f"Policy Agent response is missing "
                f"required field: {field}"
            )


    # ========================================================
    # 7. VALIDATE POLICY STATUS
    # ========================================================

    allowed_statuses = {
        "NORMAL",
        "ELEVATED",
        "UNKNOWN"
    }

    if policy_result["policy_status"] not in allowed_statuses:

        raise ValueError(
            "Invalid policy_status returned by Qwen: "
            f"{policy_result['policy_status']}"
        )


    # ========================================================
    # 8. DISPLAY RESULT
    # ========================================================

    print()
    print("Policy result:")

    print(
        json.dumps(
            policy_result,
            indent=2
        )
    )


    # ========================================================
    # 9. RETURN LANGGRAPH STATE UPDATE
    # ========================================================

    return {

        "policy_status":
            policy_result["policy_status"],

        "policy_reason":
            policy_result["policy_reason"],

        "policy_confidence":
            policy_result["policy_confidence"]
    }