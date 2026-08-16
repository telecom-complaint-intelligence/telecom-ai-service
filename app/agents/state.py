from typing import TypedDict, Optional, Dict, Any

class ComplaintAgentState(TypedDict):
    complaint: str
    raw_llm_response: Optional[str]
    parsed_technical_info: Optional[Dict[str, Any]]
    retry_count: int
    error: Optional[str]
    is_valid: bool
