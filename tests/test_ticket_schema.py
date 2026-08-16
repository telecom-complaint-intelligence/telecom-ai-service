import os
import sys
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.main import app

client = TestClient(app)

def test_analyze_complaint_endpoint():
    payload = {
        "complaint": "the network tower near my house got bursted"
    }

    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["complaint"] == "the network tower near my house got bursted"
    assert "category" in data
    assert "negativity_score" in data
    assert "sentiment_score" in data
    assert "technical_information" in data
    assert "complexity" in data
    assert "total_complexity_score" in data

def test_analyze_empty_complaint_raises_error():
    payload = {
        "complaint": "   "
    }

    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 400
