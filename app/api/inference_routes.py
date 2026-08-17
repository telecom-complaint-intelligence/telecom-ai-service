from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.agents.escalation.agents.escalation_agent import run_escalation_agent
from app.agents.escalation.schemas.input_schema import EscalationInput
from app.agents.high.api import run_high_agent
from app.agents.solution.graph.solution_graph import solution_graph
from app.aggregator.aggregator import aggregate_complaint_features
from app.extraction.hybrid_extractor import extract_technical_information
from app.models.categorization.category_predictor import predict_category
from app.models.sentiment.sentiment_scorer import measure_negativity

router = APIRouter(prefix="/api/v1", tags=["AI Inference & Agents"])


class AnalyzeRequest(BaseModel):
    complaint: str = Field(..., description="Customer complaint text")


class CategorizeRequest(BaseModel):
    complaint: str


class SentimentRequest(BaseModel):
    complaint: str


class ExtractRequest(BaseModel):
    complaint: str


class SolutionAgentRequest(BaseModel):
    complaint: str
    complaint_id: str | None = "CMP-001"
    complexity: str | None = "LOW"
    category: str | None = "Internet / Connectivity"
    technical_information: dict[str, Any] | str | None = None


class EscalationAgentRequest(BaseModel):
    complaint: str
    previous_solution: str | None = ""
    customer_feedback: bool | str = Field(
        ...,
        description="True/False or text indicating if previous solution worked",
    )
    current_severity: str | None = "LOW"
    complexity: str | None = "LOW"
    complexity_score: float | None = 0.5
    weighted_negativity_score: float | None = 0.5
    category: str | None = "General"
    technical_information: dict[str, Any] | str | None = None
    age_in_days: int | None = 1
    category_complaint_count: int | None = 1


class HighAgentRequest(BaseModel):
    complaint: str
    complaint_text: str | None = None
    complexity: str | None = "CRITICAL"
    scope: str | None = "area"
    duration_hours: float | None = 24.0
    severity: str | None = "high"
    technical_information: dict[str, Any] | str | None = None


@router.post("/analyze")
def analyze_complaint_full(request: AnalyzeRequest) -> dict[str, Any]:
    """
    Main AI Inference & Multi-Agent Triage endpoint called by telecom-backend.
    1. DistilBERT categorization + RoBERTa sentiment + Hybrid Information Extraction + Complexity Math.
    2. Automatically dispatches to Solution Agent (LOW/MED) or High Agent (HIGH/CRITICAL).
    """
    if not request.complaint.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Complaint text cannot be empty",
        )
    return aggregate_complaint_features(request.complaint)


@router.post("/categorize")
def categorize_complaint(request: CategorizeRequest) -> dict[str, Any]:
    cat, conf = predict_category(request.complaint)
    return {"category": cat, "category_confidence": conf}


@router.post("/sentiment")
def measure_complaint_sentiment(request: SentimentRequest) -> dict[str, Any]:
    score = measure_negativity(request.complaint)
    return {
        "negativity_score": score,
        "sentiment_score": round(score * 100.0, 2),
        "weighted_negativity_score": round(score * 15.0, 4),
    }


@router.post("/extract")
def extract_complaint_info(request: ExtractRequest) -> dict[str, Any]:
    return extract_technical_information(request.complaint)


@router.post("/agents/solution")
def run_solution_agent_endpoint(
    request: SolutionAgentRequest,
) -> dict[str, Any]:
    """
    Direct endpoint for Solution Agent (LOW/MEDIUM).
    Queries Qdrant vector KB and synthesizes customer-safe troubleshooting instructions.
    """
    try:
        sol_state = {
            "complaint_input": {
                "complaint": request.complaint,
                "complaint_id": request.complaint_id or "AUTO",
                "complexity": request.complexity or "LOW",
                "category": request.category,
                "technical_information": request.technical_information,
            }
        }
        res = solution_graph.invoke(sol_state)
        return res.get("final_solution", {})
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Solution Agent execution failed: {e!s}",
        ) from e


@router.post("/agents/escalate")
def run_escalation_agent_endpoint(
    request: EscalationAgentRequest,
) -> dict[str, Any]:
    """
    Direct endpoint for Escalation Agent.
    Evaluates customer feedback (True/False). If solution failed and triggers escalation,
    it automatically executes the High Agent and returns the high-severity diagnosis & dispatch plan.
    """
    try:
        esc_in = EscalationInput(
            complaint=request.complaint,
            current_severity=request.current_severity or "LOW",
            customer_feedback=str(request.customer_feedback).lower(),
            solution_agent_output=request.previous_solution or "",
            category=request.category,
            technical_information=request.technical_information,
            complexity=request.complexity,
            complexity_score=request.complexity_score,
            weighted_negativity_score=request.weighted_negativity_score,
            age_in_days=request.age_in_days,
            category_complaint_count=request.category_complaint_count,
        )
        esc_res = run_escalation_agent(esc_in)

        response_data = {
            "escalated": esc_res.escalated,
            "next_severity": esc_res.next_severity,
            "reasoning": esc_res.reasoning,
            "high_agent_result": None,
        }

        # If escalated to HIGH, automatically trigger High Agent
        if esc_res.next_severity == "HIGH" or esc_res.escalated:
            high_input = {
                "complaint": request.complaint,
                "complaint_text": request.complaint,
                "complexity": "HIGH",
                "category": request.category,
                "technical_information": request.technical_information,
                "escalation_reasoning": esc_res.reasoning,
                "scope": "area"
                if "multiple" in str(request.technical_information)
                else "individual",
            }
            high_res = run_high_agent(high_input)
            response_data["high_agent_result"] = {
                "diagnosis": high_res.get("diagnosis"),
                "root_cause": high_res.get("root_cause"),
                "risk_level": high_res.get("risk_level"),
                "policy_status": high_res.get("policy_status"),
                "final_decision": high_res.get("final_decision"),
                "critic_feedback": high_res.get("critic_reason"),
                "solution_high": high_res.get("proposed_action")
                or high_res.get("solution_high"),
                "confidence_score": high_res.get("confidence_score", 0.95),
            }

        return response_data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Escalation Agent execution failed: {e!s}",
        ) from e


@router.post("/agents/high")
def run_high_agent_endpoint(request: HighAgentRequest) -> dict[str, Any]:
    """
    Direct endpoint for High Agent multi-agent council (Diagnosis, Policy, Risk, Planner, Critic).
    """
    try:
        high_input = {
            "complaint": request.complaint,
            "complaint_text": request.complaint_text or request.complaint,
            "complexity": request.complexity or "CRITICAL",
            "scope": request.scope or "area",
            "duration_hours": request.duration_hours or 24.0,
            "severity": request.severity or "high",
            "technical_information": request.technical_information,
        }
        return run_high_agent(high_input)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"High Agent execution failed: {e!s}",
        ) from e
