from app.agents.high.graph import graph
from app.agents.high.state import ComplaintState


def run_high_agent(complaint_data: dict) -> dict:
    """
    Run the High Agent pipeline with the given complaint data.
    """
    # Create the initial state
    initial_state = ComplaintState(
        complaint_input=complaint_data,
        diagnosis="",
        root_cause="",
        policy_status="",
        risk_level="",
        proposed_action="",
        priority="",
        critic_feedback="",
        final_decision="",
        confidence_score=0.0,
        replan_required=False,
        replan_count=0,
        completed=False,
        messages=[]
    )

    # Run the graph
    result = graph.invoke(initial_state)
    return result
