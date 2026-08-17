"""
agents/diagnosis.py

Diagnosis Agent for the Telecom Complaint Agentic AI system.

Responsibilities:
    1. Receive complaint information.
    2. Analyze the complaint using Qwen3.5-4B.
    3. Identify the complaint diagnosis.
    4. Identify the root cause only when supported by evidence.
    5. Return structured diagnosis information.

Important:
    This agent does NOT decide whether the complaint is critical.
    That decision will be handled later by the overall workflow.

    This agent also does NOT create an intervention plan.
"""


import json

from app.agents.high.llm.qwen import ask_qwen


# ============================================================
# 1. DIAGNOSIS AGENT
# ============================================================

def diagnosis_agent(state: dict) -> dict:
    """
    Execute the Diagnosis Agent.

    Parameters
    ----------
    state : dict
        Current LangGraph state containing complaint information.

    Returns
    -------
    dict
        Diagnosis information that can be added to LangGraph state.
    """

    print()
    print("=" * 60)
    print("NODE: DIAGNOSIS AGENT")
    print("=" * 60)


    # ========================================================
    # 2. GET COMPLAINT INFORMATION
    # ========================================================

    complaint_id = state.get("complaint_id", "UNKNOWN")
    domain = state.get("domain", "UNKNOWN")
    problem_type = state.get("problem_type", "UNKNOWN")
    issue_signature = state.get(
        "issue_signature",
        "UNKNOWN"
    )

    severity = state.get("severity", "UNKNOWN")

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

    last_occurrence = state.get(
        "last_occurrence",
        "UNKNOWN"
    )

    similar_complaints = state.get(
        "similar_complaints",
        []
    )

    sentiment_score = state.get(
        "sentiment_score",
        0.0
    )


    # ========================================================
    # 3. SYSTEM PROMPT
    # ========================================================
    #
    # This defines the role and behavior of the Diagnosis Agent.
    #
    # Notice that we explicitly prevent the model from inventing
    # technical information.
    #
    # Example:
    #
    # We don't have router logs.
    #
    # Therefore Qwen should NOT say:
    #
    #     "Router hardware failure"
    #
    # unless the supplied evidence supports it.
    # ========================================================

    system_prompt = """
You are the Diagnosis Agent for a telecom customer
complaint decision-making system.

Your responsibility is to identify the most likely
complaint diagnosis using ONLY the evidence provided.

Rules:

1. Do not invent technical information.

2. Do not invent router, modem, OLT, cable,
   network-log, or infrastructure information.

3. If the technical root cause cannot be determined
   from the supplied evidence, use:

   "UNKNOWN"

4. The diagnosis should describe the complaint problem,
   not invent a technical root cause.

5. Recurring complaints can be reflected in the diagnosis.

6. Return exactly ONE JSON object.

7. Do not return markdown.

8. Do not return explanations.

9. Do not return reasoning.

10. Do not return thinking process.

IMPORTANT:

The confidence value represents confidence in the
DIAGNOSIS classification.

It does NOT represent confidence in the technical
root cause.

Therefore:

A high diagnosis confidence is allowed even when:

"root_cause": "UNKNOWN"

Required JSON format:

{
    "diagnosis": "short diagnosis",
    "root_cause": "UNKNOWN",
    "confidence": 0.0
}
"""


    # ========================================================
    # 4. USER PROMPT
    # ========================================================
    #
    # This contains the actual complaint evidence.
    #
    # The system prompt defines WHAT the agent does.
    #
    # The user prompt provides the DATA it needs to analyze.
    # ========================================================

    user_prompt = f"""
Analyze the following telecom complaint.

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

Last Occurrence:
{last_occurrence}

Similar Complaints:
{similar_complaints}

Sentiment Score:
{sentiment_score}

Return ONLY the required JSON object.
"""


    # ========================================================
    # 5. CALL QWEN
    # ========================================================

    response = ask_qwen(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=700,
        temperature=0.0,
        structured=True
    )


    # ========================================================
    # 6. PARSE QWEN RESPONSE
    # ========================================================

    try:

        diagnosis_result = json.loads(response)

    except json.JSONDecodeError as e:

        print()
        print("❌ Diagnosis Agent returned invalid JSON.")
        print("Qwen response:")
        print(response)

        raise ValueError(
            "Diagnosis Agent returned invalid JSON."
        ) from e


    # ========================================================
    # 7. VALIDATE REQUIRED FIELDS
    # ========================================================

    required_fields = [
        "diagnosis",
        "root_cause",
        "confidence"
    ]

    for field in required_fields:

        if field not in diagnosis_result:

            raise ValueError(
                f"Diagnosis Agent response is missing "
                f"required field: {field}"
            )


    # ========================================================
    # 8. DISPLAY RESULT
    # ========================================================

    print()
    print("Diagnosis result:")
    print(
        json.dumps(
            diagnosis_result,
            indent=2
        )
    )


    # ========================================================
    # 9. RETURN LANGGRAPH STATE UPDATE
    # ========================================================
    #
    # We don't replace the entire state.
    #
    # We return only the new information produced by this node.
    #
    # LangGraph will merge this information into the state.
    # ========================================================

    return {

        "diagnosis": diagnosis_result["diagnosis"],

        "root_cause": diagnosis_result["root_cause"],

        "diagnosis_root_cause":
            diagnosis_result["root_cause"],

        "diagnosis_confidence":
            diagnosis_result["confidence"]
    }