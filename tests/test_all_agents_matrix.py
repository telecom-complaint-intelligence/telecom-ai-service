"""
Unit and Integration Test Matrix for all Agent Tiers:
1. LOW Complexity -> Solution Agent (Customer self-care)
2. MEDIUM Complexity -> Solution Agent (Technical diagnostics)
3. HIGH Complexity -> High Agent Council (5 Agents: Diagnosis, Policy, Risk, Planner, Critic)
4. CRITICAL Complexity -> High Agent Council with Emergency Field Dispatch
5. ESCALATION Agent -> Evaluates Customer Feedback = False -> Escalates to High Council
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_1_low_complexity_solution_agent():
    """Test LOW complexity complaint invokes Solution Agent with customer-safe instructions."""
    payload = {"complaint": "My mobile data is not connecting on my phone"}
    resp = client.post("/api/v1/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["complexity"] in ["LOW", "MEDIUM"]
    assert data["solution_a"] is not None
    assert len(data["solution_a"]) > 0
    assert "confidence_score" in data
    print("\n[TEST 1 LOW] Solution generated:\n", data["solution_a"])


def test_2_medium_complexity_solution_agent():
    """Test MEDIUM complexity complaint invokes Solution Agent for technical diagnostics."""
    payload = {
        "complaint": "Router WAN port shows intermittent sync loss every few hours for 2 days"
    }
    resp = client.post("/api/v1/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["complexity"] in ["LOW", "MEDIUM", "HIGH"]
    assert data["solution_a"] is not None
    print("\n[TEST 2 MEDIUM] Diagnostic solution:\n", data["solution_a"])


def test_3_high_complexity_high_agent():
    """Test HIGH complexity complaint triggers High Agent multi-agent council."""
    payload = {
        "complaint": "Optical fiber distribution terminal is damaged and offline across the entire street for 96 hours"
    }
    resp = client.post("/api/v1/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["complexity"] in ["HIGH", "CRITICAL"]
    assert data["diagnosis"] is not None
    assert data["root_cause"] is not None
    assert data["risk_level"] is not None
    assert data["final_decision"] is not None
    assert data["solution_high"] is not None or data["solution_a"] is not None
    print(
        f"\n[TEST 3 HIGH] Council Diagnosis: {data['diagnosis']} | Decision: {data['final_decision']}"
    )


def test_4_critical_complexity_emergency_council():
    """Test CRITICAL complexity complaint triggers Emergency High Agent Council."""
    payload = {
        "complaint": "Major optical fiber cable severed during construction on main highway, hospital 911 emergency lines down for 4000 users"
    }
    resp = client.post("/api/v1/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["complexity"] == "CRITICAL"
    assert data["total_complexity_score"] >= 70.0
    assert data["diagnosis"] is not None
    assert data["final_decision"] is not None
    print(
        f"\n[TEST 4 CRITICAL] Complexity Score: {data['total_complexity_score']} | Decision: {data['final_decision']}"
    )


def test_5_escalation_agent_pipeline():
    """
    Test Escalation Agent:
    Takes customer feedback = False on a LOW complaint and elevates it to HIGH,
    automatically executing the High Agent council.
    """
    payload = {
        "complaint": "Broadband optical signal light is blinking red",
        "previous_solution": "1. Restart the router\n2. Check WAN cable",
        "customer_feedback": False,
        "current_severity": "LOW",
        "category": "Internet / Connectivity",
        "technical_information": {
            "component": ["ont_router", "fiber_cable"],
            "failure_type": ["physical_damage"],
            "scope": "individual",
        },
    }
    resp = client.post("/api/v1/agents/escalate", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["escalated"] is True
    assert data["next_severity"] in ["MEDIUM", "HIGH"]
    assert data["reasoning"] is not None
    assert data["high_agent_result"] is not None
    assert data["high_agent_result"]["diagnosis"] is not None
    assert data["high_agent_result"]["final_decision"] is not None
    print(
        f"\n[TEST 5 ESCALATION] Escalated: {data['escalated']} -> {data['next_severity']} | High Council Decision: {data['high_agent_result']['final_decision']}"
    )


def test_6_other_category_bypasses_complexity():
    """
    Test that complaints classified as 'Other' bypass technical complexity
    scoring and return complexity = 'OTHER' with score = 0.
    """
    payload = {"complaint": "Where is the nearest headquarters office located?"}
    resp = client.post("/api/v1/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["category"] == "Other"
    assert data["complexity"] == "OTHER"
    assert data["complexity_score"] == 0
    assert data["total_complexity_score"] == 0.0
    assert data["final_decision"] == "ROUTE_TO_GENERAL_SUPPORT"
    print(f"\n[TEST 6 OTHER] Successfully short-circuited: {data['complexity']} | Decision: {data['final_decision']}")

