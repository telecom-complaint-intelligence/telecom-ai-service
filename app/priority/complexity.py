from typing import Any

from app.ml.extraction.validator import contains_any

SCOPE_SCORE = {
    "individual": 1, "household": 2,
    "multiple_customers": 3, "multiple_users": 3,
    "widespread": 4, "area_wide": 5, "regional": 6, "nationwide": 7,
    "unknown": 2,
}

IMPACT_SCORE = {
    "minor": 1, "degraded": 2,
    "partial_outage": 3, "unavailable": 3,
    "complete_outage": 4, "unknown": 2,
}

COMPONENT_SCORE = {
    "router": 1, "wifi_router": 1, "modem": 1, "ont": 1,
    "network_cable": 2, "fiber_cable": 2, "exchange": 2,
    "base_station": 3, "network_infrastructure": 3,
    "network_tower": 4, "core_network": 4, "backbone_network": 4,
    "unknown": 2,
}

FAILURE_SCORE = {
    "slow_speed": 1, "intermittent_connection": 1, "unstable_connection": 1,
    "device_instability": 2, "device_failure": 2,
    "network_failure": 3, "cable_cut": 3,
    "infrastructure_failure": 4, "major_network_failure": 4, "physical_damage": 4,
    "complete_outage": 4,
    "unknown": 2,
}

CRITICAL_COMPONENTS = {"network_tower", "core_network", "backbone_network", "network_infrastructure"}
CRITICAL_FAILURE_TYPES = {"physical_damage", "infrastructure_failure", "major_network_failure", "cable_cut", "network_failure"}
CRITICAL_ELIGIBLE_SCOPES = {"widespread", "area_wide", "regional", "nationwide"}

RAW_SCORE_MIN = 6
RAW_SCORE_MAX = 30
MODIFIER_RAW_POINTS = 7

def check_critical_override(component: list[str], failure_type: list[str], scope: str, impact: str) -> dict[str, Any]:
    has_critical_component = contains_any(component, CRITICAL_COMPONENTS)
    has_severe_failure = contains_any(failure_type, CRITICAL_FAILURE_TYPES)

    # Path A: explicit large-scale scope + complete outage + severe cause
    if (
        scope in CRITICAL_ELIGIBLE_SCOPES
        and impact == "complete_outage"
        and (has_critical_component or has_severe_failure)
    ):
        return {
            "critical": True,
            "reason": (
                f"Large-scale ({scope}) complete outage involving "
                f"critical infrastructure and/or a severe failure type"
            ),
        }

    # Path B: critical component suffered severe failure with complete outage
    if has_critical_component and has_severe_failure and impact in {"complete_outage", "unavailable"}:
        return {
            "critical": True,
            "reason": (
                "Critical infrastructure component suffered a severe "
                "failure — real-world scope is almost certainly wider "
                "than what a single complainant reported"
            ),
        }

    return {"critical": False, "reason": None}

def calculate_base_complexity(component: list[str], failure_type: list[str], scope: str, impact: str) -> dict[str, Any]:
    scope_s = SCOPE_SCORE.get(scope, 2)
    impact_s = IMPACT_SCORE.get(impact, 2)
    component_s = max([COMPONENT_SCORE.get(c, 2) for c in component]) if component else 2
    failure_s = max([FAILURE_SCORE.get(f, 2) for f in failure_type]) if failure_type else 2

    raw_total = (scope_s * 2) + (impact_s * 2) + component_s + failure_s

    reason = (
        f"Score breakdown -> scope({scope})={scope_s}, "
        f"impact({impact})={impact_s}, component_max={component_s}, "
        f"failure_max={failure_s}, raw_total={raw_total}"
    )

    return {"raw_total": raw_total, "reason": reason}

def raw_score_to_label(raw_total: float) -> str:
    if raw_total <= 13:
        return "LOW"
    elif raw_total <= 20:
        return "MEDIUM"
    else:
        return "CRITICAL"

def raw_score_to_100(raw_total: float) -> int:
    clamped = max(RAW_SCORE_MIN, min(raw_total, RAW_SCORE_MAX))
    pct = (clamped - RAW_SCORE_MIN) / (RAW_SCORE_MAX - RAW_SCORE_MIN) * 100
    return round(pct)

def calculate_modifier(base_complexity: str, duration_hours: float | None, occurrence_pattern: str) -> dict[str, Any]:
    modifier = 0
    reasons = []

    long_duration = duration_hours is not None and duration_hours > 72
    persistent = occurrence_pattern in {"recurring", "continuous"}

    if long_duration:
        modifier = 1
        reasons.append(f"Long unresolved duration: {duration_hours} hours")

    if persistent and not long_duration:
        modifier = 1
        reasons.append(f"Persistent occurrence pattern: {occurrence_pattern}")

    return {"modifier": min(modifier, 1), "reasons": reasons}

def calculate_complexity(technical_information: dict[str, Any]) -> dict[str, Any]:
    component = technical_information.get("component", ["unknown"])
    failure_type = technical_information.get("failure_type", ["unknown"])
    scope = technical_information.get("scope", "unknown")
    impact = technical_information.get("service_impact", "unknown")
    duration = technical_information.get("duration_hours")
    occurrence = technical_information.get("occurrence_pattern", "unknown")

    critical_result = check_critical_override(component, failure_type, scope, impact)
    if critical_result["critical"]:
        return {
            "complexity": "CRITICAL",
            "complexity_score": 100,
            "base_complexity": "CRITICAL",
            "modifier": 0,
            "critical_override": True,
            "decision_reason": critical_result["reason"],
            "score_breakdown": technical_information,
        }

    base_result = calculate_base_complexity(component, failure_type, scope, impact)
    raw_total = base_result["raw_total"]
    base_complexity = raw_score_to_label(raw_total)

    modifier_result = calculate_modifier(base_complexity, duration, occurrence)
    modifier = modifier_result["modifier"]

    combined_raw = min(raw_total + (modifier * MODIFIER_RAW_POINTS), RAW_SCORE_MAX)
    final_complexity = raw_score_to_label(combined_raw)
    complexity_score = raw_score_to_100(combined_raw)

    decision_reason = base_result["reason"]
    if modifier_result["reasons"]:
        decision_reason += " | " + "; ".join(modifier_result["reasons"])
    decision_reason += f" | combined_raw={combined_raw} -> {final_complexity} ({complexity_score}/100)"

    return {
        "complexity": final_complexity,
        "complexity_score": complexity_score,
        "base_complexity": base_complexity,
        "modifier": modifier,
        "critical_override": False,
        "decision_reason": decision_reason,
        "score_breakdown": technical_information,
    }

def calculate_total_complexity(complexity_score: int, negativity_score: float) -> dict[str, float]:
    negativity = max(0.0, min(1.0, float(negativity_score)))
    sentiment_score = round(negativity * 100.0, 4)
    weighted_comp = round(float(complexity_score) * 0.85, 4)
    weighted_neg = round(sentiment_score * 0.15, 4)
    total_score = round(weighted_comp + weighted_neg, 4)

    return {
        "complexity_score": complexity_score,
        "negativity_score": round(negativity, 4),
        "sentiment_score": sentiment_score,
        "weighted_complexity_score": weighted_comp,
        "weighted_negativity_score": weighted_neg,
        "total_complexity_score": total_score
    }
