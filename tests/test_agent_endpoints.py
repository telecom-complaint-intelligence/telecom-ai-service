from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_solution_agent_endpoint():
    payload = {
        "complaint": "My wifi disconnects every 10 minutes",
        "complexity": "LOW",
        "category": "Internet / Connectivity",
    }
    response = client.post("/api/v1/agents/solution", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert (
        "customer_instructions" in data
        or "summary" in data
        or "recommended_actions" in data
    )


def test_escalation_agent_endpoint_escalates_on_false_feedback():
    payload = {
        "complaint": "The internet is completely down across our whole street",
        "previous_solution": "1. Restart router",
        "customer_feedback": False,
        "current_severity": "LOW",
        "category": "Internet / Connectivity",
        "technical_information": {
            "component": ["fiber_cable"],
            "failure_type": ["physical_damage"],
            "scope": "area",
        },
    }
    response = client.post("/api/v1/agents/escalate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "escalated" in data
    assert "next_severity" in data
    assert "reasoning" in data


def test_high_agent_endpoint():
    payload = {
        "complaint": "Main optical fiber cut during road construction",
        "complexity": "CRITICAL",
        "scope": "area",
        "duration_hours": 96.0,
    }
    response = client.post("/api/v1/agents/high", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert (
        "diagnosis" in data or "proposed_action" in data or "priority" in data
    )


def test_master_analyze_returns_agent_solution():
    payload = {"complaint": "Optical light on my ONT router is blinking red"}
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "category" in data
    assert "complexity" in data
    assert "solution_a" in data or "solution_high" in data
