import os
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.extraction.hybrid_extractor import extract_technical_information
from app.main import app
from app.priority.complexity import calculate_complexity

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_analyze_complaint_pipeline():
    complaint = "Our local fiber cable is damaged, leaving many customers without service since this morning."
    extraction_res = extract_technical_information(complaint)
    tech = extraction_res["technical_information"]
    comp_res = calculate_complexity(tech)

    assert "component" in tech
    assert "failure_type" in tech
    assert comp_res["complexity"] in ["LOW", "MEDIUM", "CRITICAL"]

def test_post_complaint_endpoint():
    payload = {
        "complaint": "Our local fiber cable is damaged, leaving many customers without service since this morning."
    }
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["complaint"] == payload["complaint"]
    assert "technical_information" in data
    assert "complexity" in data
