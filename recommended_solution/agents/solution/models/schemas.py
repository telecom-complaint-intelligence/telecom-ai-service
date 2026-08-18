from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

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
    sentiment: Optional[str] = None
    sentiment_score: Optional[float] = None
    customer_history: List[CustomerHistoryEntry] = Field(default_factory=list)
    previous_resolutions: List[PreviousResolutionEntry] = Field(default_factory=list)
    system_diagnostics: Dict[str, Any] = Field(default_factory=dict)
    additional_context: Dict[str, Any] = Field(default_factory=dict)

class EvidenceEntry(BaseModel):
    knowledge_id: str
    title: str

class LowSolution(BaseModel):
    complaint_id: str
    summary: str
    customer_instructions: List[str]
    expected_result: str
    warnings: List[str] = Field(default_factory=list)
    confidence: float
    evidence: List[EvidenceEntry] = Field(default_factory=list)
    status: str = "READY"

class MediumSolution(BaseModel):
    complaint_id: str
    problem_summary: str
    probable_causes: List[str]
    diagnostic_findings: List[str]
    recommended_actions: List[str]
    previous_resolution_analysis: str
    rationale: str
    confidence: float
    risks: List[str] = Field(default_factory=list)
    evidence: List[EvidenceEntry] = Field(default_factory=list)
    status: str = "READY"
