"""
retrieval/historical.py

Module for searching and retrieving historical resolved telecom complaints
and proven recommendations.
"""

import json
from pathlib import Path
from typing import Any

from app.agents.high.retrieval.qdrant_retriever import high_agent_retriever

DATA_PATHS = [
    Path(__file__).resolve().parent.parent / "data" / "complaints.json",
    Path(__file__).resolve().parent.parent.parent / "data" / "complaints.json",
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "complaints.json",
]


def load_historical_complaints() -> list[dict[str, Any]]:
    """
    Load the historical resolved complaints dataset from disk.
    """
    for p in DATA_PATHS:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            except Exception as e:
                print(f"⚠️ Error loading historical complaints from {p}: {e}")
    return []


def find_similar_resolved_complaints(
    query_text: str,
    domain: str | None = None,
    problem_type: str | None = None,
    top_k: int = 3
) -> list[dict[str, Any]]:
    """
    Find most similar historical resolved complaints based on keyword overlap
    and domain/problem_type matching.

    Parameters
    ----------
    query_text : str
        Complaint description or search text.
    domain : Optional[str]
        Telecom domain filter (e.g. network_infrastructure).
    problem_type : Optional[str]
        Specific failure type (e.g. router_equipment_failure).
    top_k : int
        Maximum number of matching records to return.

    Returns
    -------
    List[Dict[str, Any]]
        Ranked list of similar historical resolved complaints.
    """
    complaints = load_historical_complaints()
    if not complaints:
        return []

    query_tokens = set(query_text.lower().replace(",", " ").replace(".", " ").split())
    scored_records = []

    for item in complaints:
        score = 0.0
        item_text = (
            item.get("complaint_text", "") + " " +
            item.get("problem_type", "") + " " +
            item.get("domain", "") + " " +
            item.get("resolution_details", {}).get("root_cause_found", "")
        ).lower()
        item_tokens = set(item_text.replace(",", " ").replace(".", " ").split())

        # Keyword token overlap
        overlap = len(query_tokens.intersection(item_tokens))
        score += overlap * 2.0

        # Domain bonus
        if domain and domain.lower() == item.get("domain", "").lower():
            score += 5.0

        # Problem type exact bonus
        if problem_type and problem_type.lower() == item.get("problem_type", "").lower():
            score += 10.0

        scored_records.append((score, item))

    # Sort descending by similarity score
    scored_records.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, item in scored_records[:top_k]:
        if score > 0:
            results.append(item)

    return results


def get_recommendations_for_issue(
    query_text: str,
    domain: str | None = None,
    problem_type: str | None = None
) -> dict[str, Any]:
    """
    Retrieve historical resolution recommendations for a given telecom issue.

    Parameters
    ----------
    query_text : str
        Incoming complaint text.
    domain : Optional[str]
        Domain identifier.
    problem_type : Optional[str]
        Problem type identifier.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing matched historical case and operational recommendations.
    """
    matches = []
    try:
        qdrant_results = high_agent_retriever.search(
            domain=domain or "",
            problem_type=problem_type or "",
            complaint_text=query_text,
            top_k=1
        )
        if qdrant_results:
            matches = qdrant_results
            # Format qdrant result to match expected dictionary keys
            # if necessary. For now, assuming the Qdrant payload is a valid complaint dict.
    except Exception as e:
        print(f"⚠️ High agent Qdrant retrieval failed, falling back: {e}")

    if not matches:
        matches = find_similar_resolved_complaints(
            query_text=query_text,
            domain=domain,
            problem_type=problem_type,
            top_k=1
        )

    if not matches:
        return {
            "matched_historical_id": None,
            "similar_root_cause": "No exact historical match found.",
            "future_impact_days": 3,
            "days_unresolved": 1,
            "recommendations": {
                "immediate_action": "Follow standard tier-1 support protocol and initiate diagnostic trace.",
                "preventive_engineering": "Log issue signature for recurring pattern monitoring.",
                "customer_care_policy": "Provide ticket reference number and standard resolution ETA."
            },
            "company_recommendations": [
                "1. Operational Triage: Follow standard tier-1 support protocol and initiate diagnostic trace.",
                "2. Systemic Monitoring: Log issue signature for recurring pattern monitoring across the district.",
                "3. Customer Communication: Provide ticket reference number and standard resolution ETA.",
                "4. 7-Day Threshold Escalation: If issue remains unresolved at day 7, automatically move priority to CRITICAL."
            ]
        }

    best_match = matches[0]
    return {
        "matched_historical_id": best_match.get("complaint_id"),
        "historical_ticket": best_match.get("ticket_number"),
        "similar_root_cause": best_match.get("resolution_details", {}).get("root_cause_found"),
        "historical_action_taken": best_match.get("resolution_details", {}).get("resolution_action_taken"),
        "historical_priority": best_match.get("resolution_details", {}).get("final_priority"),
        "future_impact_days": best_match.get("future_impact_days", 5),
        "days_unresolved": best_match.get("days_unresolved", 1),
        "recommendations": best_match.get("recommendations", {}),
        "company_recommendations": best_match.get("company_recommendations", [])
    }


def assess_complaint_multidimensional_impact(state: dict[str, Any]) -> dict[str, Any]:
    """
    Evaluate multidimensional impact based on:
    1. Scope: 'area' / 'district' (Huge Problem affecting multiple consumers) vs 'individual'.
    2. Days Unresolved: Whether complaint has exceeded or reached the 7-day maximum SLA threshold.
    3. Specific Consumer Recurrence: Repeat complaints in last 30 days.
    4. Infrastructure & Life-Safety Emergency.

    Parameters
    ----------
    state : Dict[str, Any]
        Current complaint state.

    Returns
    -------
    Dict[str, Any]
        Structured impact assessment dictionary.
    """
    scope = str(state.get("scope", "")).lower()
    complaint_text = (str(state.get("complaint_text") or state.get("complaint") or "")).lower()
    domain = str(state.get("domain", "")).lower()
    problem_type = str(state.get("problem_type", "")).lower()
    severity = str(state.get("severity", "low")).lower()

    duration_hours = float(state.get("duration_hours") or 0.0)
    days_unresolved = state.get("days_unresolved")
    if days_unresolved is None:
        days_unresolved = max(1, int(duration_hours / 24)) if duration_hours else 1

    customer_previous_complaints = int(state.get("customer_previous_complaints") or 0)
    occurrences_last_30_days = int(state.get("occurrences_last_30_days") or 0)
    affected_subscribers = int(state.get("affected_subscribers") or (4200 if "tower" in complaint_text else (1850 if "fiber" in complaint_text else (120 if "neighborhood" in complaint_text or "area" in complaint_text else 1))))

    # Core Boolean Detectors
    is_area_outage = (
        scope in ["area", "district", "neighborhood"] or
        "area" in complaint_text or
        "neighborhood" in complaint_text or
        "district" in complaint_text or
        affected_subscribers > 50
    )

    is_infrastructure_damage = (
        "tower" in complaint_text or
        "fiber" in complaint_text or
        "cable" in complaint_text or
        "physical_damage" in problem_type or
        "physical damage" in complaint_text or
        "infrastructure" in domain
    )

    is_life_safety = (
        "911" in complaint_text or
        "hospital" in complaint_text or
        "clinic" in complaint_text or
        "emergency" in complaint_text or
        "public_safety" in domain
    )

    is_7_days_overdue = (days_unresolved >= 7 or duration_hours >= 168.0)
    is_high_recurrence = (customer_previous_complaints >= 3 or occurrences_last_30_days >= 5)

    critical_triggers = []
    should_move_to_critical = False

    if is_life_safety:
        should_move_to_critical = True
        critical_triggers.append("Life-Safety Emergency (911 / Public Safety circuits unreachable)")
        impact_level = "CRITICAL"
        impact_summary = "Life-critical emergency requiring instantaneous failover and zero-wait incident response."

    elif is_area_outage and is_infrastructure_damage:
        should_move_to_critical = True
        critical_triggers.append(f"Huge Area Problem: Physical infrastructure damage causing complete blackout across entire area ({affected_subscribers:,} affected consumers)")
        critical_triggers.append("Urgent Human On-Site Field Intervention Required: Emergency dispatch of specialized rigging/splicing crew")
        impact_level = "CRITICAL"
        impact_summary = f"Severe Area Outage ({affected_subscribers:,} consumers impacted) with physical hardware destruction requiring immediate on-site human crew dispatch."

    elif is_7_days_overdue:
        should_move_to_critical = True
        critical_triggers.append(f"7-Day SLA Breach: Problem has remained unresolved for {days_unresolved} days (breached maximum 7-day SLA threshold)")
        critical_triggers.append("Executive Retention Escalation: Mandatory senior engineering dispatch and executive care assignment")
        impact_level = "CRITICAL"
        impact_summary = f"Maximum SLA Violation ({days_unresolved} days unresolved) triggering automatic escalation to CRITICAL priority."

    elif is_high_recurrence and days_unresolved >= 4:
        should_move_to_critical = True
        critical_triggers.append(f"Chronic Consumer Problem Recurrence: {occurrences_last_30_days} drops in 30 days and {days_unresolved} days unresolved")
        critical_triggers.append("Executive Churn Escalation: Extreme subscriber churn probability requiring senior line engineer inspection")
        impact_level = "CRITICAL"
        impact_summary = f"Severe Chronic Recurrence ({occurrences_last_30_days} occurrences in 30 days, {days_unresolved} days open) escalated to prevent imminent subscriber churn."

    else:
        should_move_to_critical = False
        if is_area_outage or days_unresolved >= 4 or severity == "high":
            impact_level = "HIGH"
            impact_summary = f"High severity localized complaint ({days_unresolved} days open, {affected_subscribers} consumer impacted); managed under priority technical escalation."
        elif severity == "medium":
            impact_level = "MEDIUM"
            impact_summary = "Moderate impact issue managed under standard operational workflows."
        else:
            impact_level = "LOW"
            impact_summary = "Low impact localized inquiry resolved through standard self-care guide."

    return {
        "should_move_to_critical": should_move_to_critical,
        "impact_level": impact_level,
        "impact_summary": impact_summary,
        "critical_triggers": critical_triggers,
        "is_area_outage": is_area_outage,
        "affected_subscribers": affected_subscribers,
        "days_unresolved": days_unresolved,
        "duration_hours": duration_hours,
        "is_7_days_overdue": is_7_days_overdue,
        "is_high_recurrence": is_high_recurrence
    }


if __name__ == "__main__":
    print("Testing historical retrieval & impact evaluation...")
    test_cases = [
        {"complaint_text": "the network tower is damaged and area has no service", "scope": "area", "duration_hours": 96},
        {"complaint_text": "my home router has been down for 4 days", "scope": "individual", "duration_hours": 96},
        {"complaint_text": "internet has been down for 8 days unsolved", "scope": "individual", "duration_hours": 192}
    ]
    for tc in test_cases:
        res = assess_complaint_multidimensional_impact(tc)
        print("\n" + "=" * 60)
        print("Complaint:", tc["complaint_text"])
        print("Move to Critical:", res["should_move_to_critical"])
        print("Impact Level:", res["impact_level"])
        print("Triggers:", res["critical_triggers"])

