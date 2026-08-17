import os
import re
from typing import Any

import joblib

from trained_models.train_dummy_model import train_and_save_models

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "trained_models")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")
MODELS_PATH = os.path.join(MODEL_DIR, "field_models.joblib")

FIELDS = ["component", "failure_type", "scope", "service_impact", "occurrence_pattern"]

_vectorizer = None
_models = None

def _get_models():
    global _vectorizer, _models
    if _vectorizer is None or _models is None:
        if not os.path.exists(VECTORIZER_PATH) or not os.path.exists(MODELS_PATH):
            print("Joblib model files not found. Bootstrapping trained_models directory...")
            train_and_save_models(MODEL_DIR)
        _vectorizer = joblib.load(VECTORIZER_PATH)
        _models = joblib.load(MODELS_PATH)
    return _vectorizer, _models

def extract_duration_hours(complaint: str) -> float | None:
    text = complaint.lower()

    if "since this morning" in text:
        return 12.0
    if "since yesterday" in text:
        return 24.0
    if "two days" in text or "2 days" in text:
        return 48.0
    if "three days" in text or "3 days" in text:
        return 72.0
    if "one week" in text or "a week" in text or "1 week" in text:
        return 168.0

    match = re.search(r"(\d+)\s*hours?", text)
    if match:
        return float(match.group(1))

    match = re.search(r"(\d+)\s*days?", text)
    if match:
        return float(match.group(1)) * 24.0

    return None

def predict_with_ml(complaint: str) -> dict[str, Any]:
    vectorizer, models = _get_models()
    X = vectorizer.transform([complaint])
    result = {}

    for field in FIELDS:
        if field in models:
            clf = models[field]
            proba = clf.predict_proba(X)[0]
            idx = proba.argmax()
            result[field] = {
                "value": str(clf.classes_[idx]),
                "confidence": round(float(proba[idx]), 3)
            }
        else:
            result[field] = {"value": "unknown", "confidence": 0.0}

    return result

def convert_ml_prediction(complaint: str, predictions: dict[str, Any]) -> dict[str, Any]:
    return {
        "component": [predictions.get("component", {}).get("value", "unknown")],
        "failure_type": [predictions.get("failure_type", {}).get("value", "unknown")],
        "scope": predictions.get("scope", {}).get("value", "unknown"),
        "service_impact": predictions.get("service_impact", {}).get("value", "unknown"),
        "duration_hours": extract_duration_hours(complaint),
        "occurrence_pattern": predictions.get("occurrence_pattern", {}).get("value", "unknown"),
    }
