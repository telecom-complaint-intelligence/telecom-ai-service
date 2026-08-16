from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

from app.aggregator.aggregator import aggregate_complaint_features
from app.models.categorization.category_predictor import predict_category
from app.models.sentiment.sentiment_scorer import measure_negativity
from app.extraction.hybrid_extractor import extract_technical_information

router = APIRouter(prefix="/api/v1", tags=["AI Inference"])

class AnalyzeRequest(BaseModel):
    complaint: str = Field(..., description="Customer complaint text")

class CategorizeRequest(BaseModel):
    complaint: str

class SentimentRequest(BaseModel):
    complaint: str

class ExtractRequest(BaseModel):
    complaint: str

@router.post("/analyze")
def analyze_complaint_full(request: AnalyzeRequest) -> Dict[str, Any]:
    """
    Main AI Inference endpoint called by telecom-backend.
    Runs DistilBERT categorization + RoBERTa sentiment + Information Extraction + Technical Complexity.
    """
    if not request.complaint.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Complaint text cannot be empty"
        )
    return aggregate_complaint_features(request.complaint)

@router.post("/categorize")
def categorize_complaint(request: CategorizeRequest) -> Dict[str, Any]:
    cat, conf = predict_category(request.complaint)
    return {"category": cat, "category_confidence": conf}

@router.post("/sentiment")
def measure_complaint_sentiment(request: SentimentRequest) -> Dict[str, Any]:
    score = measure_negativity(request.complaint)
    return {
        "negativity_score": score,
        "weighted_negativity_score": round(score * 15.0, 4)
    }

@router.post("/extract")
def extract_complaint_info(request: ExtractRequest) -> Dict[str, Any]:
    return extract_technical_information(request.complaint)
