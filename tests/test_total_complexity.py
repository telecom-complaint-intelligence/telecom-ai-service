import os
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.main import app
from app.priority.complexity import calculate_total_complexity

client = TestClient(app)

def test_calculate_total_complexity_formula():
    complexity_score = 75
    negativity_score = 0.4374

    result = calculate_total_complexity(complexity_score, negativity_score)

    assert result["complexity_score"] == 75
    assert result["negativity_score"] == 0.4374
    assert result["sentiment_score"] == 43.74
    assert result["weighted_complexity_score"] == 63.75
    assert result["weighted_negativity_score"] == 6.561
    assert result["total_complexity_score"] == 70.311

def test_post_complaint_returns_total_complexity_scores():
    payload = {
        "complaint": "Comcast Cable Internet Speeds are terribly slow and unusable"
    }

    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "complexity_score" in data
    assert "negativity_score" in data
    assert "sentiment_score" in data
    assert "weighted_complexity_score" in data
    assert "weighted_negativity_score" in data
    assert "total_complexity_score" in data

    # Verify formula relationship
    expected_sentiment = round(data["negativity_score"] * 100.0, 4)
    expected_weighted_comp = round(data["complexity_score"] * 0.85, 4)
    expected_weighted_neg = round(expected_sentiment * 0.15, 4)
    expected_total = round(expected_weighted_comp + expected_weighted_neg, 4)

    assert data["sentiment_score"] == expected_sentiment
    assert data["weighted_complexity_score"] == expected_weighted_comp
    assert data["weighted_negativity_score"] == expected_weighted_neg
    assert data["total_complexity_score"] == expected_total
