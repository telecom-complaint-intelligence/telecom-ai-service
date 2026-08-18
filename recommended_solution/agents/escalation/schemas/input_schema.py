from pydantic import BaseModel, Field
from typing import Any, Optional, Dict
class EscalationInput(BaseModel):
    complaint: str = Field(..., description="Original complaint text")
    current_severity: str = Field(..., description="Current severity")
    customer_feedback: str = Field(..., description="Customer outcome")
    solution_agent_output: Any = Field(..., description="Output from Solution Agent")
    category: Optional[str] = Field(default=None, description="Complaint category")
    technical_information: Optional[Dict[str, Any]] = Field(default=None, description="Technical metadata")
    complexity: Optional[str] = Field(default=None, description="Complexity tier")
    complexity_score: Optional[float] = Field(default=None, description="Complexity score")
    weighted_negativity_score: Optional[float] = Field(default=None, description="Negativity score")
    age_in_days: Optional[int] = Field(default=None, description="Age of the complaint in days")
    category_complaint_count: Optional[int] = Field(default=None, description="Count of open complaints in the same category")

