from typing import Any


def format_list_items(items: Any) -> str:
    """Helper to convert list or string of components/failures into a clean string."""
    if isinstance(items, list):
        clean = [
            str(i).replace("_", " ").title()
            for i in items
            if str(i).lower() not in ["unknown", "none"]
        ]
        return ", ".join(clean) if clean else "Equipment / Service"
    if isinstance(items, str) and items.lower() not in ["unknown", "none"]:
        return items.replace("_", " ").title()
    return "Equipment / Service"


def generate_summary(data: dict[str, Any]) -> dict[str, Any]:
    """
    Non-LLM Deterministic Feature-Aggregated Summary Engine.
    Dynamically generates structured and text summaries across LOW, MEDIUM, HIGH, CRITICAL, and OTHER tiers.
    In LOW severity, embeds the exact customer troubleshooting solution directly in the summary.
    """
    complexity = data.get("complexity", "LOW").upper()
    category = data.get("category", "General")
    tech_info = data.get("technical_information", {}) or {}

    comp_str = format_list_items(tech_info.get("component", []))
    fail_str = format_list_items(tech_info.get("failure_type", []))
    scope_str = str(tech_info.get("scope", "individual")).replace("_", " ")
    impact_str = str(tech_info.get("service_impact", "minor")).replace("_", " ")
    duration = tech_info.get("duration_hours")
    pattern = str(tech_info.get("occurrence_pattern", "one_time")).replace("_", " ")

    solution_a = data.get("solution_a") or ""
    solution_high = data.get("solution_high") or ""
    diagnosis = data.get("diagnosis") or f"{comp_str} Issue"
    root_cause = data.get("root_cause") or "N/A (Pending Inspection)"
    final_decision = data.get("final_decision") or "SELF_CARE"
    total_score = float(data.get("total_complexity_score", 0.0))

    # 1. LOW COMPLEXITY: Customer / L1 Helpdesk with direct solution steps
    if complexity == "LOW":
        clean_sol = solution_a.strip()
        if not clean_sol:
            clean_sol = "1. Power cycle device for 30 seconds.\n2. Verify optical/LAN cables."

        summary_text = (
            f"[{category}] {comp_str} - {fail_str} ({scope_str} impact).\n"
            f"Recommended Solution:\n{clean_sol}"
        )
        headline = f"Low Complexity: {comp_str} {fail_str}"
        solution_snippet = clean_sol
        structured = {
            "headline": headline,
            "category": category,
            "affected_component": comp_str,
            "symptom": fail_str,
            "scope": scope_str,
            "solution_steps": clean_sol,
        }

    # 2. MEDIUM COMPLEXITY: Tier-2 Diagnostic Synopsis
    elif complexity == "MEDIUM":
        dur_str = f" lasting {duration}h" if duration else ""
        action = solution_a.strip() or "Perform remote ONT line check and optical power diagnostic."
        summary_text = (
            f"[{category} | Score: {total_score:.1f}/100] {comp_str} experiencing {fail_str} ({pattern}){dur_str} "
            f"with {impact_str} impact.\n"
            f"Recommended Diagnostic Action:\n{action}"
        )
        headline = f"Medium Diagnostic: {comp_str} ({fail_str})"
        solution_snippet = action
        structured = {
            "headline": headline,
            "category": category,
            "component": comp_str,
            "failure_pattern": pattern,
            "duration_hours": duration,
            "service_impact": impact_str,
            "diagnostic_action": action,
        }

    # 3. HIGH COMPLEXITY: Tier-3 Operational Work Order
    elif complexity == "HIGH":
        action = solution_high.strip() if solution_high else solution_a.strip() or "Dispatch field engineer for on-site line inspection."
        dur_str = f" (Unresolved: {duration}h)" if duration else ""
        summary_text = (
            f"🚨 HIGH INCIDENT [Score: {total_score:.1f}/100]: {diagnosis} — {fail_str} on {comp_str} affecting {scope_str}{dur_str}.\n"
            f"Field Action Required ({final_decision}):\n{action}"
        )
        headline = f"High Incident: {diagnosis} ({comp_str})"
        solution_snippet = action
        structured = {
            "headline": headline,
            "diagnosis": diagnosis,
            "affected_scope": scope_str,
            "root_cause_hypothesis": root_cause,
            "final_decision": final_decision,
            "field_action": action,
        }

    # 4. CRITICAL COMPLEXITY: NOC Outage Briefing
    elif complexity == "CRITICAL":
        action = solution_high.strip() if solution_high else solution_a.strip() or "Emergency field restoration unit dispatched."
        dur_str = f" for {duration}h" if duration else ""
        summary_text = (
            f"🚨 CRITICAL OUTAGE BRIEFING [Score: {total_score:.1f}/100]: {scope_str.upper()} blackout due to {fail_str} on {comp_str}{dur_str}.\n"
            f"Diagnosis: {diagnosis}\n"
            f"Root Cause: {root_cause}\n"
            f"Emergency Dispatch ({final_decision}):\n{action}"
        )
        headline = f"CRITICAL OUTAGE: {scope_str.upper()} ({comp_str})"
        solution_snippet = action
        structured = {
            "headline": headline,
            "incident_level": "CRITICAL",
            "scope": scope_str,
            "infrastructure": comp_str,
            "root_cause": root_cause,
            "emergency_action": action,
            "final_decision": final_decision,
        }

    # 5. OTHER: General Customer Care Inquiry
    else:
        summary_text = (
            f"General Inquiry: Request classified under '{category}' (Score: 0.0/100).\n"
            f"Action: Forwarded to general Customer Care desk."
        )
        headline = f"General Inquiry: {category}"
        solution_snippet = "Forwarded to general Customer Care desk."
        structured = {
            "headline": headline,
            "category": category,
            "action": "ROUTE_TO_GENERAL_SUPPORT",
        }

    return {
        "headline": headline,
        "summary": summary_text,
        "solution_snippet": solution_snippet,
        "summary_structured": structured,
    }
