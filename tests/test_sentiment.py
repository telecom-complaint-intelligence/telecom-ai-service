import os
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.main import app
from app.models.sentiment.sentiment_scorer import analyze_complaint_negativity, measure_negativity

client = TestClient(app)

def test_measure_negativity_function():
    complaint = "Our fiber internet service is completely broken and down since morning!"
    score = measure_negativity(complaint)
    assert 0.0 <= score <= 1.0

    analysis = analyze_complaint_negativity(complaint)
    assert analysis["negativity_score"] == score
    assert analysis["weighted_negativity_score"] == round(score * 15.0, 4)

def test_post_complaint_sentiment_endpoint():
    payload = {
        "complaint": "Our internet connection is extremely slow and unusable."
    }
    response = client.post("/api/v1/sentiment", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "negativity_score" in data
    assert "weighted_negativity_score" in data
    assert 0.0 <= data["negativity_score"] <= 1.0
    assert data["weighted_negativity_score"] == round(data["negativity_score"] * 15.0, 4)

def test_post_complaint_includes_sentiment():
    payload = {
        "complaint": "The router has no lights and we have complete outage."
    }
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "negativity_score" in data
    assert "weighted_negativity_score" in data
    assert data["weighted_negativity_score"] == round(data["negativity_score"] * 15.0, 4)
