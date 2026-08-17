"""
state.py

Shared state for the telecom complaint LangGraph workflow.
"""

from typing import TypedDict


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
    days_unresolved: int | None
    duration_hours: float | None
    scope: str | None
    affected_subscribers: int | None
    future_impact_days: int | None
    impact_level: str | None
    impact_reason: str | None
    critical_triggers: list[str] | None
    company_recommendations: list[str] | None


    # ============================================================
    # DIAGNOSIS AGENT
    # ============================================================

    diagnosis: str | None
    root_cause: str | None
    diagnosis_confidence: float | None


    # ============================================================
    # POLICY AGENT
    # ============================================================

    policy_status: str | None
    policy_reason: str | None
    policy_confidence: float | None


    # ============================================================
    # RISK AGENT
    # ============================================================

    risk_level: str | None
    risk_reason: str | None
    risk_confidence: float | None


    # ============================================================
    # PLANNER AGENT
    # ============================================================

    proposed_decision: str | None
    priority: str | None
    proposed_action: str | None
    planner_reason: str | None
    planner_confidence: float | None


    # ============================================================
    # CRITIC AGENT
    # ============================================================

    critic_decision: str | None
    critic_reason: str | None
    critic_confidence: float | None
    replan_required: bool | None


    # ============================================================
    # REPLAN AGENT
    # ============================================================

    revised_decision: str | None
    revised_priority: str | None
    revised_action: str | None
    revised_reason: str | None
    replan_confidence: float | None


    # ============================================================
    # FINAL RESULT
    # ============================================================

    final_decision: str | None
    final_priority: str | None
    final_action: str | None
    final_reason: str | None
    final_confidence: float | None
    moved_to_critical: bool | None


    # ============================================================
    # GRAPH CONTROL & ENTERPRISE GUARDRAILS
    # ============================================================

    current_node: str | None

    retry_count: int
    max_retries: int

    error: str | None

    # Guardrails tracking
    guardrail_status: dict | None
    guardrail_violations: list[str] | None
    guardrail_warnings: list[str] | None
    execution_token: str | None
    human_review_dossier: dict | None