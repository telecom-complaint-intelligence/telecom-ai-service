import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.agents.complaint_agent import complaint_agent_graph, run_complaint_agent


def test_langgraph_complaint_agent():
    complaint = "Our fiber cable got severed by road construction crew, whole area is down since morning."
    result = run_complaint_agent(complaint)

    assert isinstance(result, dict)
    assert "component" in result
    assert "failure_type" in result
    assert "scope" in result
    assert "service_impact" in result

def test_langgraph_state_graph_compilation():
    assert complaint_agent_graph is not None
