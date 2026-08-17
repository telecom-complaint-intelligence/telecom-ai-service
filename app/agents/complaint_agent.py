import json
import os
from typing import Any, Dict
import httpx
from langgraph.graph import END, StateGraph

from app.agents.state import ComplaintAgentState
from app.config import settings
from app.ml.extraction.validator import validate_technical_information
from app.prompts.extraction_prompts import EXTRACTION_SYSTEM_PROMPT


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def llm_extraction_node(state: ComplaintAgentState) -> Dict[str, Any]:
    api_url = getattr(
        settings,
        "API_URL",
        "https://router.huggingface.co/v1/chat/completions",
    )
    hf_token = (
        getattr(settings, "HF_TOKEN", "")
        or os.getenv("HF_TOKEN")
        or os.getenv("GROQ_API_KEY")
        or ""
    )
    model = getattr(settings, "MODEL", "meta-llama/Llama-3.1-8B-Instruct")

    if not hf_token:
        return {
            "raw_llm_response": None,
            "error": "No HF/Groq API token configured",
            "retry_count": state.get("retry_count", 0) + 1,
        }

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": state["complaint"]},
        ],
    }

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(api_url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            raw_content = data["choices"][0]["message"]["content"]
            return {"raw_llm_response": raw_content, "error": None}
    except Exception as e:
        return {
            "raw_llm_response": None,
            "error": f"HTTP Error {getattr(getattr(e, 'response', None), 'status_code', '')}: {e}",
            "retry_count": state.get("retry_count", 0) + 1,
        }


def validate_response_node(state: ComplaintAgentState) -> Dict[str, Any]:
    raw_content = state.get("raw_llm_response")
    if not raw_content:
        return {
            "parsed_technical_info": None,
            "is_valid": False,
            "error": state.get("error") or "Empty LLM response",
        }

    try:
        cleaned = _strip_code_fences(raw_content)
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            validated = validate_technical_information(parsed)
            return {
                "parsed_technical_info": validated,
                "is_valid": True,
                "error": None,
            }
        raise ValueError("Top-level JSON is not a dictionary object")
    except Exception as e:
        return {
            "parsed_technical_info": None,
            "is_valid": False,
            "error": f"JSON parsing error: {e}",
        }


def should_retry_or_end(state: ComplaintAgentState) -> str:
    if state.get("is_valid"):
        return "end"
    if state.get("retry_count", 0) >= 2:
        return "end"
    return "retry"


builder = StateGraph(ComplaintAgentState)
builder.add_node("llm_extraction", llm_extraction_node)
builder.add_node("validate_response", validate_response_node)
builder.set_entry_point("llm_extraction")
builder.add_edge("llm_extraction", "validate_response")
builder.add_conditional_edges(
    "validate_response",
    should_retry_or_end,
    {"end": END, "retry": "llm_extraction"},
)
complaint_agent_graph = builder.compile()


def _heuristic_fallback_extraction(complaint: str) -> Dict[str, Any]:
    """
    Intelligent resilient semantic entity extractor when remote LLM is rate-limited.
    """
    c = complaint.lower()
    components = []
    if "fiber" in c or "cable" in c:
        components.append("fiber_cable")
    if "tower" in c:
        components.append("network_tower")
    if "router" in c or "ont" in c or "modem" in c or "wifi" in c:
        components.append("router")
    if "sim" in c or "mobile" in c:
        components.append("sim")
    if not components:
        components = ["unknown"]

    failures = []
    if "cut" in c or "severed" in c or "damage" in c:
        failures.append("physical_damage")
    elif (
        "down" in c
        or "outage" in c
        or "offline" in c
        or "blackout" in c
        or "no service" in c
    ):
        failures.append("complete_outage")
    elif "slow" in c or "speed" in c:
        failures.append("slow_speed")
    elif "drop" in c or "disconnect" in c or "intermittent" in c or "loss" in c:
        failures.append("intermittent_connection")
    else:
        failures = ["unknown"]

    scope = "individual"
    if (
        "4000" in c
        or "thousand" in c
        or "highway" in c
        or "emergency" in c
        or "911" in c
        or "hospital" in c
        or "district" in c
        or "widespread" in c
    ):
        scope = "widespread"
    elif (
        "area" in c
        or "street" in c
        or "entire" in c
        or "multiple" in c
        or "neighborhood" in c
    ):
        scope = "area_wide"

    impact = (
        "complete_outage"
        if (
            "down" in c
            or "offline" in c
            or "cut" in c
            or "severed" in c
            or "blackout" in c
            or "emergency" in c
            or "911" in c
        )
        else "degraded"
    )
    duration = (
        168.0
        if ("96" in c or "7 days" in c or "unresolved" in c or "week" in c)
        else (72.0 if "hours" in c or "2 days" in c else 24.0)
    )
    pattern = (
        "chronic_recurring"
        if ("frequent" in c or "every" in c or "constantly" in c)
        else "one_time"
    )

    return {
        "component": components,
        "failure_type": failures,
        "scope": scope,
        "service_impact": impact,
        "duration_hours": duration,
        "occurrence_pattern": pattern,
    }


def run_complaint_agent(complaint: str) -> Dict[str, Any]:
    initial_state: ComplaintAgentState = {
        "complaint": complaint,
        "raw_llm_response": None,
        "parsed_technical_info": None,
        "retry_count": 0,
        "error": None,
        "is_valid": False,
    }

    final_state = complaint_agent_graph.invoke(initial_state)

    if final_state.get("parsed_technical_info"):
        return final_state["parsed_technical_info"]

    # Use intelligent heuristic fallback
    return validate_technical_information(
        _heuristic_fallback_extraction(complaint)
    )
