"""
state.py

Shared state for the telecom complaint LangGraph workflow.
"""

from typing import TypedDict, Optional, Any


class ComplaintState(TypedDict, total=False):

    # ============================================================
    # ORIGINAL COMPLAINT DATA
    # ============================================================

    complaint_id: str
    customer_id: str

    complaint_text: str

    domain: str
    problem_type: str
    issue_signature: str

    severity: str

    historical_occurrences: int
    occurrences_last_30_days: int
    customer_previous_complaints: int

    last_occurrence: str
    similar_complaints: list[str]

    sentiment_score: float

    # Impact & Timeline Metrics
    days_unresolved: Optional[int]
    duration_hours: Optional[float]
    scope: Optional[str]
    affected_subscribers: Optional[int]
    future_impact_days: Optional[int]
    impact_level: Optional[str]
    impact_reason: Optional[str]
    critical_triggers: Optional[list[str]]
    company_recommendations: Optional[list[str]]


    # ============================================================
    # DIAGNOSIS AGENT
    # ============================================================

    diagnosis: Optional[str]
    root_cause: Optional[str]
    diagnosis_confidence: Optional[float]


    # ============================================================
    # POLICY AGENT
    # ============================================================

    policy_status: Optional[str]
    policy_reason: Optional[str]
    policy_confidence: Optional[float]


    # ============================================================
    # RISK AGENT
    # ============================================================

    risk_level: Optional[str]
    risk_reason: Optional[str]
    risk_confidence: Optional[float]


    # ============================================================
    # PLANNER AGENT
    # ============================================================

    proposed_decision: Optional[str]
    priority: Optional[str]
    proposed_action: Optional[str]
    planner_reason: Optional[str]
    planner_confidence: Optional[float]


    # ============================================================
    # CRITIC AGENT
    # ============================================================

    critic_decision: Optional[str]
    critic_reason: Optional[str]
    critic_confidence: Optional[float]
    replan_required: Optional[bool]


    # ============================================================
    # REPLAN AGENT
    # ============================================================

    revised_decision: Optional[str]
    revised_priority: Optional[str]
    revised_action: Optional[str]
    revised_reason: Optional[str]
    replan_confidence: Optional[float]


    # ============================================================
    # FINAL RESULT
    # ============================================================

    final_decision: Optional[str]
    final_priority: Optional[str]
    final_action: Optional[str]
    final_reason: Optional[str]
    final_confidence: Optional[float]
    moved_to_critical: Optional[bool]


    # ============================================================
    # GRAPH CONTROL
    # ============================================================

    current_node: Optional[str]

    retry_count: int
    max_retries: int

    error: Optional[str]