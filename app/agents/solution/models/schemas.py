from typing import Any

from pydantic import BaseModel, Field


class CustomerHistoryEntry(BaseModel):
    complaint_id: str
    issue: str
    date: str

class PreviousResolutionEntry(BaseModel):
    complaint_id: str
    solution: str
    outcome: str

class ComplaintInput(BaseModel):
    complaint_id: str
    customer_id: str
    complaint_text: str
    priority: str  # "LOW", "MEDIUM", "HIGH"
    domain: str
    component: str
    failure_type: str
    scope: str
    impact: str
    
    # Optional fields for MEDIUM/HIGH priority
    sentiment: str | None = None
    sentiment_score: float | None = None
    customer_history: list[CustomerHistoryEntry] = Field(default_factory=list)
    previous_resolutions: list[PreviousResolutionEntry] = Field(default_factory=list)
    system_diagnostics: dict[str, Any] = Field(default_factory=dict)
    additional_context: dict[str, Any] = Field(default_factory=dict)

class EvidenceEntry(BaseModel):
    knowledge_id: str
    title: str

class LowSolution(BaseModel):
    complaint_id: str
    summary: str
    customer_instructions: list[str]
    expected_result: str
    warnings: list[str] = Field(default_factory=list)
    confidence: float
    evidence: list[EvidenceEntry] = Field(default_factory=list)
    status: str = "READY"

class MediumSolution(BaseModel):
    complaint_id: str
    problem_summary: str
    probable_causes: list[str]
    diagnostic_findings: list[str]
    recommended_actions: list[str]
    previous_resolution_analysis: str
    rationale: str
    confidence: float
    risks: list[str] = Field(default_factory=list)
    evidence: list[EvidenceEntry] = Field(default_factory=list)
    status: str = "READY"
