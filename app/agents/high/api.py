from app.agents.high.graph import graph
from app.agents.high.state import ComplaintState


def run_high_agent(complaint_data: dict) -> dict:
    """
    Run the High Agent pipeline with the given complaint data.
    """
    complaint_text = (
        complaint_data.get("complaint_text")
        or complaint_data.get("complaint")
        or "Broadband outage"
    )
    initial_state = ComplaintState(
        complaint_id=complaint_data.get("complaint_id", "TICK-HIGH-001"),
        customer_id=complaint_data.get("customer_id", "CUST-001"),
        complaint_text=complaint_text,
        domain=complaint_data.get("domain", "Broadband & Fiber"),
        problem_type=complaint_data.get("problem_type", "Network Outage"),
        severity=complaint_data.get("severity", "HIGH"),
        duration_hours=float(complaint_data.get("duration_hours", 24.0) or 24.0),
        days_unresolved=int(complaint_data.get("days_unresolved", 1) or 1),
        scope=complaint_data.get("scope", "area"),
        affected_subscribers=int(complaint_data.get("affected_subscribers", 50) or 50),
        sentiment_score=float(complaint_data.get("sentiment_score", 0.8) or 0.8),
        retry_count=0,
        max_retries=1,
    )

    # Run the graph
    result = graph.invoke(initial_state)
    return result

