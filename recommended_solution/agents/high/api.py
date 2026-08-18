from agents.high.graph import graph
from agents.high.state import ComplaintState

def run_high_agent(complaint_data: dict) -> dict:
    """
    Run the High Agent pipeline with the given complaint data.
    """
    tech_info = complaint_data.get("technical_information", {}) if complaint_data else {}
    if not isinstance(tech_info, dict):
        tech_info = {}
        
    components = tech_info.get("component", [])
    failures = tech_info.get("failure_type", [])
    comp = components[0] if components else "general"
    fail = failures[0] if failures else "general"
    
    domain = complaint_data.get("domain", complaint_data.get("category", "general"))
    problem_type = complaint_data.get("problem_type", f"{comp}_{fail}")
    
    service_impact = tech_info.get("service_impact", "disruption")
    issue_signature = f"{domain}.{fail}.{service_impact}"
    
    scope_val = tech_info.get("scope", "individual")
    if scope_val == "area":
        affected = 4500
    elif scope_val == "multiple_customers":
        affected = 50
    else:
        affected = 1
    
    age_raw = complaint_data.get("age_in_days")
    age = int(age_raw) if age_raw is not None else 1
    
    duration_hours_raw = tech_info.get("duration_hours")
    duration_hours = float(duration_hours_raw) if duration_hours_raw is not None else float(24 * age)
    
    sentiment_raw = complaint_data.get("weighted_negativity_score")
    if sentiment_raw is None:
        sentiment_raw = complaint_data.get("negativity_score")
    sentiment_score = float(sentiment_raw) if sentiment_raw is not None else 0.5
    
    complaint_count_raw = complaint_data.get("category_complaint_count")
    complaint_count = int(complaint_count_raw) if complaint_count_raw is not None else 1
    
    # Create the initial state conforming to agents/high/state.py:ComplaintState
    initial_state = ComplaintState(
        complaint_id=str(complaint_data.get("complaint_id", "UNKNOWN")),
        customer_id=str(complaint_data.get("customer_id", "UNKNOWN")),
        complaint_text=str(complaint_data.get("complaint", complaint_data.get("complaint_text", ""))),
        domain=domain,
        problem_type=problem_type,
        issue_signature=issue_signature,
        severity=str(complaint_data.get("complexity", "HIGH")).lower(),
        scope=scope_val,
        duration_hours=duration_hours,
        days_unresolved=age,
        affected_subscribers=affected,
        historical_occurrences=complaint_count,
        occurrences_last_30_days=complaint_count,
        customer_previous_complaints=0,
        last_occurrence="2026-08-16",
        similar_complaints=[],
        sentiment_score=sentiment_score,
        retry_count=0,
        max_retries=1,
        current_node="start"
    )

    # Run the graph
    result = graph.invoke(initial_state)
    
    # Add backward compatibility fields for the orchestrator
    if "final_confidence" in result and "confidence_score" not in result:
        result["confidence_score"] = result["final_confidence"]
        
    return result


