from pydantic import BaseModel, Field
from typing import Optional

class EscalationOutput(BaseModel):
    next_severity: str = Field(..., description="The calculated next severity (LOW, MEDIUM, or HIGH)")
    escalated: bool = Field(..., description="True if escalated to HIGH, False otherwise")
    reasoning: Optional[str] = Field(None, description="Policy reasoning for the decision")
