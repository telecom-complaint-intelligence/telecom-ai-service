from typing import Any

from pydantic import BaseModel, Field


class EscalationInput(BaseModel):
    complaint: str = Field(..., description="Original complaint text")
    current_severity: str = Field(
        default="LOW", description="Current severity"
    )
    customer_feedback: bool | str = Field(
        ...,
        description="Customer outcome (True/False or 'worked'/'still broken')",
    )
    solution_agent_output: Any = Field(
        default="", description="Output from Solution Agent"
    )
    category: str | None = Field(
        default=None, description="Complaint category"
    )
    technical_information: dict[str, Any] | str | None = Field(
        default=None, description="Technical metadata"
    )
    complexity: str | None = Field(
        default=None, description="Complexity tier"
    )
    complexity_score: float | None = Field(
        default=None, description="Complexity score"
    )
    weighted_negativity_score: float | None = Field(
        default=None, description="Negativity score"
    )
    age_in_days: int | None = Field(
        default=None, description="Age of the complaint in days"
    )
    category_complaint_count: int | None = Field(
        default=None,
        description="Count of open complaints in the same category",
    )
