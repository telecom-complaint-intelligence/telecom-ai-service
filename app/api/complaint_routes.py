
from app.db.models import ComplaintSentiment
from app.db.session import get_db
from app.schemas.complaint_schema import (
    ComplaintRequest,
    ComplaintResponse,
    SentimentRequest,
    SentimentResponse,
)
from app.services.complaint_service import ComplaintService
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.ml.sentiment.sentiment_scorer import measure_negativity

router = APIRouter(prefix="/complaints", tags=["Complaints"])

@router.post("/sentiment", response_model=SentimentResponse, status_code=status.HTTP_201_CREATED)
def calculate_and_store_sentiment(
    request: SentimentRequest,
    db: Session = Depends(get_db)
):
    neg_score = measure_negativity(request.complaint)
    weighted_score = round(neg_score * 15.0, 4)

    db_sentiment = ComplaintSentiment(
        complaint_text=request.complaint,
        negativity_score=neg_score,
        weighted_negativity_score=weighted_score
    )

    db.add(db_sentiment)
    db.commit()
    db.refresh(db_sentiment)

    return SentimentResponse(
        id=db_sentiment.id,
        complaint_id=db_sentiment.complaint_id,
        complaint=db_sentiment.complaint_text,
        negativity_score=db_sentiment.negativity_score,
        weighted_negativity_score=db_sentiment.weighted_negativity_score
    )

@router.post("", response_model=ComplaintResponse, status_code=status.HTTP_201_CREATED)
def create_complaint_analysis(
    request: ComplaintRequest,
    db: Session = Depends(get_db)
):
    return ComplaintService.process_and_save_complaint(request, db)

@router.post("/batch", response_model=list[ComplaintResponse], status_code=status.HTTP_201_CREATED)
def create_batch_complaints(
    requests: list[ComplaintRequest],
    db: Session = Depends(get_db)
):
    return ComplaintService.process_batch_complaints(requests, db)

@router.get("", response_model=list[ComplaintResponse])
def list_complaints(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    return ComplaintService.get_all_complaints(db, skip=skip, limit=limit)

@router.get("/{complaint_id}", response_model=ComplaintResponse)
def get_complaint_by_id(
    complaint_id: int,
    db: Session = Depends(get_db)
):
    response = ComplaintService.get_complaint_by_id(complaint_id, db)
    if not response:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Complaint with ID {complaint_id} not found"
        )
    return response
