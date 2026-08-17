from typing import Any

from app.agents.complaint_agent import run_complaint_agent
from app.config import settings
from app.extraction.ml_predictor import convert_ml_prediction, predict_with_ml
from app.extraction.validator import validate_technical_information


def extract_technical_information(complaint: str) -> dict[str, Any]:
    threshold = settings.CONFIDENCE_THRESHOLD

    # 1. ML Prediction
    ml_predictions = predict_with_ml(complaint)

    confidences = {
        field: pred["confidence"]
        for field, pred in ml_predictions.items()
    }

    lowest_confidence = min(confidences.values()) if confidences else 0.0

    # 2. Check Threshold
    if lowest_confidence >= threshold:
        raw_tech = convert_ml_prediction(complaint, ml_predictions)
        technical_information = validate_technical_information(raw_tech)
        extraction_source = "ml"
    else:
        technical_information = run_complaint_agent(complaint)
        extraction_source = "llm"

    # 3. Hybrid Result
    return {
        "technical_information": technical_information,
        "extraction_source": extraction_source,
        "ml_predictions": ml_predictions,
        "lowest_confidence": round(lowest_confidence, 3),
        "threshold": threshold,
    }
