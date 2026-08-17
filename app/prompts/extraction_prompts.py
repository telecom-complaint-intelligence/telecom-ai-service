"""
Centralized prompt templates for complaint extraction and LLM agent processing.
"""

EXTRACTION_SYSTEM_PROMPT = """You are an expert telecommunications complaint analysis system.

Your job is to extract structured technical information from a customer complaint.

COMPLAINT:
{complaint}

Return ONLY valid JSON. Use EXACTLY this structure:
{{
    "component": [],
    "failure_type": [],
    "scope": "",
    "service_impact": "",
    "duration_hours": null,
    "occurrence_pattern": "",
    "complexity": ""
}}

ALLOWED COMPONENT VALUES:
router, wifi_router, modem, ont, fiber_cable, network_cable,
network_tower, core_network, backbone_network, network_infrastructure,
base_station, exchange, unknown

ALLOWED FAILURE TYPE VALUES:
slow_speed, intermittent_connection, device_failure, device_instability,
physical_damage, infrastructure_failure, network_failure,
major_network_failure, cable_cut, complete_outage, unstable_connection,
unknown

ALLOWED SCOPE VALUES:
individual, household, multiple_customers, multiple_users, widespread,
area_wide, regional, nationwide, unknown

ALLOWED SERVICE IMPACT VALUES:
degraded, minor, partial_outage, complete_outage, unavailable, unknown

ALLOWED OCCURRENCE PATTERN VALUES:
one_time, recurring, intermittent, continuous, unknown

ALLOWED COMPLEXITY VALUES:
LOW, MEDIUM, HIGH, CRITICAL, OTHER

IMPORTANT RULES:
1. Do not invent information. Use "unknown" when the complaint does not provide enough information for a field.
2. Only use values from the allowed lists above — never invent new ones.
3. "many customers", "multiple customers", "several users", "several houses", "multiple houses" -> multiple_customers
4. "entire area" -> area_wide
5. "entire region" -> regional
6. "whole country" or "nationwide" -> nationwide
7. "every day", "frequently", "keeps happening" -> recurring
8. A continuous outage lasting since a stated time -> continuous
9. Convert durations into HOURS: "since yesterday"->24, "for two days"->48, "for three days"->72, "for a week"->168
10. "burning", "on fire", "destroyed", "vandalized", "physical hazard", "broken", "smoke" -> failure_type: physical_damage, service_impact: complete_outage, complexity: CRITICAL
11. "cut", "severed", "sliced", "damaged cable" -> failure_type: cable_cut, service_impact: complete_outage, complexity: CRITICAL
12. NEGATIONS AND NON-COMPLAINTS: If the text is a negation (e.g. "not burning", "no fire", "is not cut", "everything is working"), a test message, or doesn't describe a real problem, set component to ["unknown"], failure_type to ["unknown"], scope to "unknown", service_impact to "unknown", and complexity to "OTHER".

Return ONLY the JSON object, nothing else.
"""
