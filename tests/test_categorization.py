import os
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.main import app
from app.models.categorization.category_predictor import predict_category

client = TestClient(app)

def test_predict_category_function():
    text = "the network tower near my house got bursted"
    category, confidence = predict_category(text)
    assert isinstance(category, str)
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0

def test_post_complaint_includes_category():
    payload = {
        "complaint": "the network tower near my house got bursted"
    }

    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "category" in data
    assert data["category"] is not None
    assert "category_confidence" in data
