import json
import re
import urllib.request
import urllib.error
from typing import Dict, Any
from langgraph.graph import StateGraph, END

from app.config import settings
from app.prompts.extraction_prompts import EXTRACTION_SYSTEM_PROMPT
from app.agents.state import ComplaintAgentState
from app.ml.extraction.validator import validate_technical_information

def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()

def _call_hf_llm_api(prompt: str) -> str:
    token = settings.HF_TOKEN
    if not token:
        raise RuntimeError("HF_TOKEN environment variable is not set in .env")

    payload = {
        "model": settings.MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0,
    }

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        settings.API_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.loads(response.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"]

# ----------------------------------------------------
# LANGGRAPH AGENT NODES
# ----------------------------------------------------
def llm_extraction_node(state: ComplaintAgentState) -> Dict[str, Any]:
    prompt = EXTRACTION_SYSTEM_PROMPT.format(complaint=state["complaint"])
    retry_count = state.get("retry_count", 0)
    try:
        raw_output = _call_hf_llm_api(prompt)
        return {
            "raw_llm_response": raw_output,
            "retry_count": retry_count + 1,
            "error": None
        }
    except Exception as e:
        return {
            "raw_llm_response": None,
            "retry_count": retry_count + 1,
            "error": str(e)
        }

def validate_response_node(state: ComplaintAgentState) -> Dict[str, Any]:
    raw_resp = state.get("raw_llm_response")
    if not raw_resp:
        return {
            "parsed_technical_info": None,
            "is_valid": False,
            "error": state.get("error") or "Empty LLM response"
        }

    try:
        cleaned = _strip_code_fences(raw_resp)
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            validated = validate_technical_information(parsed)
            return {
                "parsed_technical_info": validated,
                "is_valid": True,
                "error": None
            }
        raise ValueError("Top-level JSON is not a dictionary object")
    except Exception as e:
        return {
            "parsed_technical_info": None,
            "is_valid": False,
            "error": f"JSON parsing error: {e}"
        }

def should_retry_or_end(state: ComplaintAgentState) -> str:
    if state.get("is_valid"):
        return "end"
    if state.get("retry_count", 0) >= 2:
        return "end"
    return "retry"

# ----------------------------------------------------
# BUILD LANGGRAPH STATEGRAPH AGENT
# ----------------------------------------------------
builder = StateGraph(ComplaintAgentState)

builder.add_node("llm_extraction", llm_extraction_node)
builder.add_node("validate_response", validate_response_node)

builder.set_entry_point("llm_extraction")
builder.add_edge("llm_extraction", "validate_response")

builder.add_conditional_edges(
    "validate_response",
    should_retry_or_end,
    {
        "end": END,
        "retry": "llm_extraction"
    }
)

complaint_agent_graph = builder.compile()

def run_complaint_agent(complaint: str) -> Dict[str, Any]:
    initial_state: ComplaintAgentState = {
        "complaint": complaint,
        "raw_llm_response": None,
        "parsed_technical_info": None,
        "retry_count": 0,
        "error": None,
        "is_valid": False
    }

    final_state = complaint_agent_graph.invoke(initial_state)

    if final_state.get("parsed_technical_info"):
        return final_state["parsed_technical_info"]

    print(f"Notice: LangGraph Agent fallback used (Error: {final_state.get('error')})")
    return validate_technical_information({
        "component": ["unknown"],
        "failure_type": ["unknown"],
        "scope": "unknown",
        "service_impact": "unknown",
        "duration_hours": None,
        "occurrence_pattern": "unknown"
    })
