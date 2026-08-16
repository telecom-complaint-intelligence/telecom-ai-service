from typing import Dict, Any
from app.config import settings
from app.ml.extraction.ml_predictor import predict_with_ml, convert_ml_prediction
from app.agents.complaint_agent import run_complaint_agent
from app.ml.extraction.validator import validate_technical_information

def extract_technical_information(complaint: str) -> Dict[str, Any]:
    threshold = settings.CONFIDENCE_THRESHOLD

    # ----------------------------------------
    # STEP 1: ML PREDICTION
    # ----------------------------------------
    ml_predictions = predict_with_ml(complaint)

    confidences = {
        field: pred["confidence"]
        for field, pred in ml_predictions.items()
    }

    lowest_confidence = min(confidences.values()) if confidences else 0.0

    # ----------------------------------------
    # STEP 2: CHECK CONFIDENCE THRESHOLD
    # ----------------------------------------
    if lowest_confidence >= threshold:
        raw_tech = convert_ml_prediction(complaint, ml_predictions)
        technical_information = validate_technical_information(raw_tech)
        extraction_source = "ml"
    else:
        technical_information = run_complaint_agent(complaint)
        extraction_source = "llm"

    # ----------------------------------------
    # FINAL HYBRID RESULT
    # ----------------------------------------
    return {
        "technical_information": technical_information,
        "extraction_source": extraction_source,
        "ml_predictions": ml_predictions,
        "lowest_confidence": round(lowest_confidence, 3),
        "threshold": threshold,
    }
