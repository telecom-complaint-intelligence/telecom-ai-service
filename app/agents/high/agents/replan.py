"""
agents/replan.py

Replan Agent for the LangGraph telecom complaint system.

Flow:

Planner
   ↓
Critic
   ↓
if REJECT
   ↓
Replan
   ↓
Revised Decision

The Replan Agent uses Qwen3.5-4B through llm.qwen.ask_qwen().
"""

import json
import re


# ============================================================
# 1. QWEN CLIENT
# ============================================================

from app.agents.high.llm.qwen import ask_qwen


# ============================================================
# 2. REPLAN SYSTEM PROMPT
# ============================================================

REPLAN_SYSTEM_PROMPT = """
You are the Replan Agent in a telecom customer complaint decision-making system.

Your job is to revise the Planner decision after the Critic rejects the original plan.

IMPORTANT RULES:

1. Address Critic feedback directly:
   - If Critic rejected due to priority over-escalation (e.g. CRITICAL when risk was HIGH and issue has NOT exceeded 7 days):
     Correct revised_priority to "HIGH".
   - If issue has been unresolved for 7+ days (days_unresolved >= 7 or duration >= 168 hours):
     Keep revised_priority = "CRITICAL" and revised_decision = "CRITICAL" to resolve the 7-day overdue SLA breach.
2. Keep revised_decision = "ESCALATE" when risk is HIGH and policy is ELEVATED.
3. Formulate a specific, concrete, operational action (e.g. dispatching field engineering to inspect damaged infrastructure and restore service).
4. Do NOT invent unsupported technical root causes (root cause is UNKNOWN).
5. Do NOT invent company policies or customer information.
6. Return ONLY one JSON object.

Required JSON structure:
{
  "revised_decision": "ESCALATE",
  "revised_priority": "HIGH",
  "revised_action": "Escalate to tier-2 network engineering and dispatch a field team to inspect the damaged tower and restore area connectivity.",
  "revised_reason": "The complaint represents a major area outage with physical network damage and high operational risk, requiring technical escalation with HIGH priority.",
  "replan_confidence": 0.95
}
"""


# ============================================================
# 3. JSON EXTRACTION
# ============================================================

def extract_json_from_text(text: str):
    """
    Extract a JSON object from normal Qwen output or reasoning.

    Qwen may return:

        content = ""

    while putting the final JSON somewhere inside:

        reasoning = "Thinking Process ... { ... }"

    This function attempts to recover that JSON.
    """

    if not text:
        return None

    text = text.strip()

    # --------------------------------------------------------
    # Attempt 1: Entire text is JSON
    # --------------------------------------------------------

    try:
        return json.loads(text)
    except Exception:
        pass

    # --------------------------------------------------------
    # Attempt 2: Remove markdown code fences
    # --------------------------------------------------------

    cleaned = re.sub(
        r"```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    cleaned = re.sub(
        r"```\s*",
        "",
        cleaned
    )

    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # --------------------------------------------------------
    # Attempt 3: Find JSON object using brace matching
    # --------------------------------------------------------

    candidates = []

    start_positions = [
        match.start()
        for match in re.finditer(r"\{", text)
    ]

    for start in start_positions:

        depth = 0
        in_string = False
        escape = False

        for i in range(start, len(text)):

            char = text[i]

            # Handle escaped characters inside strings
            if escape:
                escape = False
                continue

            if char == "\\" and in_string:
                escape = True
                continue

            # Handle JSON string boundaries
            if char == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == "{":
                depth += 1

            elif char == "}":
                depth -= 1

                if depth == 0:

                    candidate = text[start:i + 1]

                    try:
                        parsed = json.loads(candidate)

                        if isinstance(parsed, dict):
                            candidates.append(parsed)

                    except Exception:
                        pass

                    break

    # --------------------------------------------------------
    # Prefer Replan JSON
    # --------------------------------------------------------

    for candidate in reversed(candidates):

        if (
            "revised_decision" in candidate
            and
            "revised_priority" in candidate
            and
            "revised_action" in candidate
        ):
            return candidate

    # --------------------------------------------------------
    # Return last valid object
    # --------------------------------------------------------

    if candidates:
        return candidates[-1]

    return None


# ============================================================
# 4. VALIDATE REPLAN RESULT
# ============================================================

def validate_replan_result(result):
    """
    Validate the JSON returned by Qwen.
    """

    if not isinstance(result, dict):
        return False

    required_fields = [
        "revised_decision",
        "revised_priority",
        "revised_action",
        "revised_reason",
        "replan_confidence"
    ]

    for field in required_fields:

        if field not in result:
            return False

    # Check required string fields

    string_fields = [
        "revised_decision",
        "revised_priority",
        "revised_action",
        "revised_reason"
    ]

    for field in string_fields:

        if not isinstance(result[field], str):
            return False

        if not result[field].strip():
            return False

    # Check confidence

    try:

        confidence = float(
            result["replan_confidence"]
        )

        if confidence < 0 or confidence > 1:
            return False

    except Exception:
        return False

    return True


# ============================================================
# 5. BUILD USER PROMPT
# ============================================================

def build_replan_prompt(state):
    """
    Build the Replan Agent input from LangGraph state.

    The function supports the field names used by the previous
    agents in your project.
    """

    # --------------------------------------------------------
    # Planner information
    # --------------------------------------------------------

    planner_decision = state.get(
        "proposed_decision",
        state.get("planner_decision", "UNKNOWN")
    )

    planner_priority = state.get(
        "priority",
        state.get("planner_priority", "UNKNOWN")
    )

    planner_action = state.get(
        "proposed_action",
        state.get("planner_action", "UNKNOWN")
    )

    planner_reason = state.get(
        "planner_reason",
        state.get("reason", "UNKNOWN")
    )

    # --------------------------------------------------------
    # Critic information
    # --------------------------------------------------------

    critic_decision = state.get(
        "critic_decision",
        "UNKNOWN"
    )

    critic_reason = state.get(
        "critic_reason",
        state.get("critic_feedback", "UNKNOWN")
    )

    # --------------------------------------------------------
    # Risk information
    # --------------------------------------------------------

    risk_level = state.get(
        "risk_level",
        "UNKNOWN"
    )

    risk_reason = state.get(
        "risk_reason",
        "UNKNOWN"
    )

    # --------------------------------------------------------
    # Policy information
    # --------------------------------------------------------

    policy_status = state.get(
        "policy_status",
        "UNKNOWN"
    )

    policy_reason = state.get(
        "policy_reason",
        "UNKNOWN"
    )

    # --------------------------------------------------------
    # Complaint information
    # --------------------------------------------------------

    occurrences_last_30_days = state.get(
        "occurrences_last_30_days",
        "UNKNOWN"
    )

    previous_complaints = state.get(
        "customer_previous_complaints",
        state.get(
            "previous_complaints",
            "UNKNOWN"
        )
    )

    complaint_id = state.get(
        "complaint_id",
        "UNKNOWN"
    )

    severity = state.get(
        "severity",
        "UNKNOWN"
    )

    problem_type = state.get(
        "problem_type",
        "UNKNOWN"
    )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
Replan the rejected decision using ONLY the information below.

COMPLAINT
Complaint ID: {complaint_id}
Problem type: {problem_type}
Severity: {severity}

PLANNER
Decision: {planner_decision}
Priority: {planner_priority}
Action: {planner_action}
Reason: {planner_reason}

CRITIC
Decision: {critic_decision}
Feedback: {critic_reason}

RISK
Risk level: {risk_level}
Risk reason: {risk_reason}

POLICY
Policy status: {policy_status}
Policy reason: {policy_reason}

HISTORY
Occurrences in last 30 days: {occurrences_last_30_days}
Previous customer complaints: {previous_complaints}

TASK

Revise the Planner decision so that it directly addresses
the Critic feedback.

The revised action must be clearer than:

"{planner_action}"

Do not invent teams, technical causes, logs, timelines,
or unsupported information.

Return ONLY this JSON structure:

{{
  "revised_decision": "...",
  "revised_priority": "...",
  "revised_action": "...",
  "revised_reason": "...",
  "replan_confidence": 0.0
}}
"""

    return prompt


# ============================================================
# 6. REPLAN AGENT
# ============================================================

def replan_agent(state):
    """
    Replan Agent.

    Called when the Critic rejects the Planner decision.
    """

    print()
    print("=" * 60)
    print("NODE: REPLAN AGENT")
    print("=" * 60)

    # --------------------------------------------------------
    # Display current information
    # --------------------------------------------------------

    planner_decision = state.get(
        "proposed_decision",
        state.get("planner_decision", "UNKNOWN")
    )

    planner_priority = state.get(
        "priority",
        state.get("planner_priority", "UNKNOWN")
    )

    planner_action = state.get(
        "proposed_action",
        state.get("planner_action", "UNKNOWN")
    )

    critic_decision = state.get(
        "critic_decision",
        "UNKNOWN"
    )

    critic_reason = state.get(
        "critic_reason",
        state.get("critic_feedback", "UNKNOWN")
    )

    risk_level = state.get(
        "risk_level",
        "UNKNOWN"
    )

    policy_status = state.get(
        "policy_status",
        "UNKNOWN"
    )

    print()
    print("Original planner decision:", planner_decision)
    print("Original priority:", planner_priority)
    print("Original action:", planner_action)

    print()
    print("Critic decision:", critic_decision)
    print("Critic feedback:", critic_reason)

    print()
    print("Risk level:", risk_level)
    print("Policy status:", policy_status)

    # --------------------------------------------------------
    # Build prompt
    # --------------------------------------------------------

    user_prompt = build_replan_prompt(state)

    # --------------------------------------------------------
    # Call Qwen
    # --------------------------------------------------------

    try:

        response = ask_qwen(
            system_prompt=REPLAN_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=700,
            temperature=0.1,
            structured=True
        )

    except Exception as e:

        print()
        print("❌ Qwen Replan error:")
        print(str(e))

        raise

    # --------------------------------------------------------
    # Parse Qwen response
    # --------------------------------------------------------

    print()
    print("Parsing Replan response...")

    result = extract_json_from_text(response)

    if result is None:

        print()
        print("❌ Could not extract Replan JSON.")

        print()
        print("Raw Qwen response:")
        print(response)

        raise RuntimeError(
            "Replan Agent did not return valid JSON."
        )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not validate_replan_result(result):

        print()
        print("❌ Invalid Replan JSON:")

        print(
            json.dumps(
                result,
                indent=2
            )
        )

        raise RuntimeError(
            "Replan Agent returned invalid JSON structure."
        )

    # --------------------------------------------------------
    # Normalize confidence
    # --------------------------------------------------------

    result["replan_confidence"] = float(
        result["replan_confidence"]
    )

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print()
    print("Replan result:")

    print(
        json.dumps(
            result,
            indent=2
        )
    )

    print()
    print("✅ REPLAN AGENT COMPLETED")

    # --------------------------------------------------------
    # Return LangGraph state update
    # --------------------------------------------------------

    return {
        "revised_decision": result["revised_decision"],
        "revised_priority": result["revised_priority"],
        "revised_action": result["revised_action"],
        "revised_reason": result["revised_reason"],
        "replan_confidence": result["replan_confidence"],

        # Useful for the next Critic iteration
        "proposed_decision": result["revised_decision"],
        "priority": result["revised_priority"],
        "proposed_action": result["revised_action"],
        "planner_reason": result["revised_reason"],

        # Mark that replanning happened
        "replan_required": False
    }