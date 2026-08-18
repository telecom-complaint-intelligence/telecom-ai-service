from typing import Any, TypedDict


class ComplaintAgentState(TypedDict):
    complaint: str
    raw_llm_response: str | None
    parsed_technical_info: dict[str, Any] | None
    retry_count: int
    error: str | None
    is_valid: bool
