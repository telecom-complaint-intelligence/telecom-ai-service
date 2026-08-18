from typing import Any

ALLOWED_COMPONENT = {
    "router", "wifi_router", "modem", "ont",
    "fiber_cable", "network_cable",
    "network_tower", "core_network", "backbone_network",
    "network_infrastructure", "base_station", "exchange",
    "unknown",
}

ALLOWED_FAILURE_TYPE = {
    "slow_speed", "intermittent_connection",
    "device_failure", "device_instability",
    "physical_damage", "infrastructure_failure",
    "network_failure", "major_network_failure",
    "cable_cut", "complete_outage", "unstable_connection",
    "unknown",
}

ALLOWED_SCOPE = {
    "individual", "household",
    "multiple_customers", "multiple_users",
    "widespread", "area_wide", "regional", "nationwide",
    "unknown",
}

ALLOWED_IMPACT = {
    "degraded", "minor", "partial_outage",
    "complete_outage", "unavailable", "unknown",
}

ALLOWED_OCCURRENCE = {
    "one_time", "recurring", "intermittent", "continuous", "unknown",
}

def normalize_string(value: Any) -> str:
    if value is None:
        return "unknown"
    val_str = str(value).strip().lower()
    return val_str if val_str else "unknown"

def normalize_list(value: Any) -> list[str]:
    if value is None:
        return ["unknown"]
    if isinstance(value, str):
        val_str = value.strip().lower()
        return [val_str] if val_str else ["unknown"]
    if isinstance(value, list):
        cleaned = [str(v).strip().lower() for v in value if v is not None and str(v).strip()]
        return cleaned if cleaned else ["unknown"]
    return ["unknown"]

def validate_value(value: str, allowed: set) -> str:
    return value if value in allowed else "unknown"

def validate_list(values: list[str], allowed: set) -> list[str]:
    cleaned = [validate_value(v, allowed) for v in values]
    cleaned = [v for v in cleaned if v != "unknown"] or ["unknown"]
    return cleaned

def safe_duration(value: Any) -> float | None:
    if value is None:
        return None
    try:
        val_float = float(value)
        return val_float if val_float >= 0 else None
    except (ValueError, TypeError):
        return None

def contains_any(values: list[str], targets: set) -> bool:
    return any(v in targets for v in values)

ALLOWED_COMPLEXITY = {"low", "medium", "high", "critical", "other", "unknown"}

def validate_technical_information(raw: dict[str, Any]) -> dict[str, Any]:
    component = validate_list(normalize_list(raw.get("component")), ALLOWED_COMPONENT)
    failure_type = validate_list(normalize_list(raw.get("failure_type")), ALLOWED_FAILURE_TYPE)
    scope = validate_value(normalize_string(raw.get("scope")), ALLOWED_SCOPE)
    service_impact = validate_value(normalize_string(raw.get("service_impact")), ALLOWED_IMPACT)
    occurrence_pattern = validate_value(normalize_string(raw.get("occurrence_pattern")), ALLOWED_OCCURRENCE)
    duration_hours = safe_duration(raw.get("duration_hours"))
    complexity = validate_value(normalize_string(raw.get("complexity")), ALLOWED_COMPLEXITY)

    return {
        "component": component,
        "failure_type": failure_type,
        "scope": scope,
        "service_impact": service_impact,
        "duration_hours": duration_hours,
        "occurrence_pattern": occurrence_pattern,
        "complexity": complexity.upper() if complexity != "unknown" else "unknown"
    }
