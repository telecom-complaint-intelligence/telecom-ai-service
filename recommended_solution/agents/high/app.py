import sys

# Configure UTF-8 encoding for Windows standard streams
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from agents.high.graph import graph


# ============================================================
# INPUT COMPLAINT DATA
# ============================================================

raw_input = {
  "complaint_id": 23,
  "ticket_number": "223463",
  "complaint": "the network tower in my area has physical damage and has been down for four days",
  "received_via": "Internet",
  "city": "acworth",
  "state": "georgia",
  "zip_code": "30102",
  "date_parsed": "2015-08-05",
  "time_parsed": "11:35:20",
  "status": 0,
  "extraction_source": "llm",
  "lowest_confidence": 0.14,
  "technical_information": {
    "component": [
      "network_tower"
    ],
    "failure_type": [
      "physical_damage"
    ],
    "scope": "area",
    "service_impact": "complete_outage",
    "duration_hours": 96,
    "occurrence_pattern": "ongoing"
  },
  "complexity": "HIGH",
  "complexity_score": 85,
  "negativity_score": 0.91,
  "weighted_negativity_score": 11.05
}

duration_h = raw_input["technical_information"].get("duration_hours", 96)
scope_val = raw_input["technical_information"].get("scope", "area")

initial_state = {
    "complaint_id": str(raw_input["complaint_id"]),
    "customer_id": f"TICKET_{raw_input['ticket_number']}",
    "complaint_text": raw_input["complaint"],
    "domain": "network_infrastructure",
    "problem_type": f"{raw_input['technical_information']['component'][0]}_{raw_input['technical_information']['failure_type'][0]}",
    "issue_signature": f"network_infrastructure.{raw_input['technical_information']['failure_type'][0]}.{raw_input['technical_information']['service_impact']}",
    "severity": raw_input["complexity"].lower(),
    "scope": scope_val,
    "duration_hours": duration_h,
    "days_unresolved": max(1, int(duration_h / 24)),
    "affected_subscribers": 4200 if scope_val == "area" else 1,
    "historical_occurrences": 1,
    "occurrences_last_30_days": 1,
    "customer_previous_complaints": 0,
    "last_occurrence": raw_input["date_parsed"],
    "similar_complaints": [],
    "sentiment_score": raw_input["negativity_score"],
    "retry_count": 0,
    "max_retries": 1,
    "current_node": "start",
}


# ============================================================
# RUN GRAPH WITH TIMING
# ============================================================

import time

print()
print("=" * 70)
print("STARTING TELECOM COMPLAINT AGENTIC WORKFLOW")
print("=" * 70)

start_time = time.perf_counter()

result = graph.invoke(initial_state)

end_time = time.perf_counter()
execution_time_sec = end_time - start_time
execution_time_ms = execution_time_sec * 1000


# ============================================================
# FINAL RESULT & IMPACT BREAKDOWN
# ============================================================

print()
print("=" * 70)
print("FINAL DECISION & MULTIDIMENSIONAL IMPACT BREAKDOWN")
print("=" * 70)

is_critical = result.get("moved_to_critical")
if is_critical is None:
    is_critical = (
        str(result.get("final_priority")).strip().upper() == "CRITICAL" or
        str(result.get("final_decision")).strip().upper() == "CRITICAL"
    )

triggers = result.get("critical_triggers", [])
if is_critical:
    trigger_desc = triggers[0] if triggers else "Human Field Intervention / Critical SLA Threshold"
    critical_status_str = f"YES ({trigger_desc})"
else:
    critical_status_str = "NO (Preserved HIGH Priority - Single Customer Issue Under 7 Days)"

print("Moved to Critical :", critical_status_str)
print("Decision          :", result.get("final_decision"))
print("Priority          :", result.get("final_priority"))
print("Problem Scope     :", f"{str(result.get('scope', scope_val)).upper()} (Affected Consumers: {result.get('affected_subscribers', 1):,})")
print("Days Unresolved   :", f"{result.get('days_unresolved', 1)} days ({result.get('duration_hours', 0.0):.0f} hours)")
print("Future Impact Est :", f"{result.get('future_impact_days', 5)} days of projected disruption & churn risk")
print("Impact Level      :", result.get("impact_level", "HIGH"))
print("Impact Summary    :", result.get("impact_reason", "Standard operational impact."))
print("Action            :", result.get("final_action"))
print("Reason            :", result.get("final_reason"))
print("Confidence        :", result.get("final_confidence"))
print("Execution Time    :", f"{execution_time_sec:.3f} seconds ({execution_time_ms:.1f} ms)")

if triggers and len(triggers) > 1:
    print()
    print("Critical Triggers :")
    for trg in triggers:
        print(f"  ⚡ {trg}")

company_recs = result.get("company_recommendations", [])
if company_recs:
    print()
    print("-" * 70)
    print("COMPANY RECOMMENDATIONS (3-5 Strategic Action Points):")
    print("-" * 70)
    for rec in company_recs:
        print(f"  • {rec}")

print()
print("=" * 70)
print("WORKFLOW COMPLETED")
print("=" * 70)