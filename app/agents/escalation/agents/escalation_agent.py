from app.agents.escalation.engine.decision_engine import decision_engine_graph
from app.agents.escalation.schemas.input_schema import EscalationInput
from app.agents.escalation.schemas.output_schema import EscalationOutput


def run_escalation_agent(input_data: EscalationInput) -> EscalationOutput:
    initial_state = {
        "complaint": input_data.complaint,
        "current_severity": input_data.current_severity,
        "customer_feedback": input_data.customer_feedback,
        "solution_agent_output": input_data.solution_agent_output,
        "category": input_data.category,
        "technical_information": input_data.technical_information,
        "complexity": input_data.complexity,
        "complexity_score": input_data.complexity_score,
        "weighted_negativity_score": input_data.weighted_negativity_score,
        "age_in_days": input_data.age_in_days,
        "category_complaint_count": input_data.category_complaint_count,
    }
    result = decision_engine_graph.invoke(initial_state)
    return EscalationOutput(
        next_severity=result["next_severity"],
        escalated=result["escalated"],
        reasoning=result.get("reasoning", "")
    )
