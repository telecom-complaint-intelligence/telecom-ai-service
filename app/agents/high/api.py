import json
from typing import Dict, Any

from app.agents.high.graph import graph
from app.agents.high.state import ComplaintState


def run_high_agent(complaint_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the High Agent pipeline with the given complaint data and 9 guardrails.
    """
    if not isinstance(complaint_data, dict):
        complaint_data = {"complaint": str(complaint_data)}

    complaint_id = str(complaint_data.get("complaint_id") or "CMP-AUTO")
    ticket_num = str(complaint_data.get("ticket_number") or complaint_data.get("customer_id") or "223464")
    customer_id = ticket_num if ticket_num.startswith("TICKET_") else f"TICKET_{ticket_num}"
    complaint_text = (
        complaint_data.get("complaint_text")
        or complaint_data.get("complaint")
        or ""
    )

    tech_info = complaint_data.get("technical_information") or {}
    if isinstance(tech_info, str):
        try:
            tech_info = json.loads(tech_info)
        except Exception:
            tech_info = {}
    if not isinstance(tech_info, dict):
        tech_info = {}

    duration_h = complaint_data.get("duration_hours")
    if duration_h is None and isinstance(tech_info, dict):
        duration_h = tech_info.get("duration_hours")
    try:
        duration_h = float(duration_h) if duration_h is not None else 48.0
    except (ValueError, TypeError):
        duration_h = 48.0

    scope_val = complaint_data.get("scope")
    if not scope_val and isinstance(tech_info, dict):
        scope_val = tech_info.get("scope")
    if not scope_val:
        scope_val = "area"

    complexity = (
        complaint_data.get("complexity")
        or complaint_data.get("severity")
        or "HIGH"
    )

    initial_state: ComplaintState = {
        "complaint_id": complaint_id,
        "customer_id": customer_id,
        "complaint_text": complaint_text,
        "domain": complaint_data.get("domain", "network_infrastructure"),
        "problem_type": complaint_data.get("problem_type", "network_disruption"),
        "issue_signature": complaint_data.get("issue_signature", "network_infrastructure.outage"),
        "severity": str(complexity).lower(),
        "scope": scope_val,
        "duration_hours": duration_h,
        "days_unresolved": int(complaint_data.get("days_unresolved", max(1, int(duration_h / 24)))),
        "affected_subscribers": int(complaint_data.get("affected_subscribers", 4200 if scope_val == "area" else 1)),
        "historical_occurrences": int(complaint_data.get("historical_occurrences", 1)),
        "occurrences_last_30_days": int(complaint_data.get("occurrences_last_30_days", 1)),
        "customer_previous_complaints": int(complaint_data.get("customer_previous_complaints", 0)),
        "last_occurrence": str(complaint_data.get("last_occurrence", "2015-08-05")),
        "similar_complaints": complaint_data.get("similar_complaints", []),
        "sentiment_score": float(complaint_data.get("sentiment_score", 0.85)),
        "retry_count": 0,
        "max_retries": 1,
        "current_node": "start",
    }

    # Run the graph
    result = graph.invoke(initial_state)
    return result
