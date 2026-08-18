from typing import Any

from app.agents.high.api import run_high_agent
from app.agents.solution.graph.solution_graph import solution_graph
from app.extraction.hybrid_extractor import extract_technical_information
from app.models.categorization.category_predictor import predict_category
from app.models.sentiment.sentiment_scorer import measure_negativity
from app.priority.complexity import (
    calculate_complexity,
    calculate_total_complexity,
)


def aggregate_complaint_features(complaint_text: str) -> dict[str, Any]:
    """
    Feature Aggregator (F.A.) & Multi-Agent Orchestrator —
    1. DistilBERT Category Prediction
    2. RoBERTa Sentiment Negativity Scorer
    3. Hybrid Technical Information Extraction
    4. Priority & Complexity Calculation (85% Tech + 15% Sentiment)
    5. Agentic Resolution:
       - If LOW / MEDIUM: Synthesizes customer instructions via Solution Agent.
       - If HIGH / CRITICAL: Executes multi-agent diagnosis & dispatch via High Agent.
    """
    # 1. DistilBERT Category Prediction
    category, category_confidence = predict_category(complaint_text)

    # 2. RoBERTa Sentiment Negativity Scorer
    negativity_score = measure_negativity(complaint_text)

    # Short-circuit if Category is a non-technical / CRM inquiry (Other, Account, Billing, Cancellation, Customer Support, Service/Plan)
    category_lower = category.strip().lower()
    non_tech_info = {
        "other": {
            "decision": "ROUTE_TO_GENERAL_SUPPORT",
            "solution": "Thank you for contacting us. Your request has been categorized as a general inquiry and forwarded to our Customer Care team.",
            "diagnosis": "Non-Technical / General Customer Inquiry",
            "root_cause": "N/A (Non-Technical)"
        },
        "account": {
            "decision": "ROUTE_TO_ACCOUNT_SUPPORT",
            "solution": "Your account access inquiry has been received and routed to our Account Services team. A support representative will assist you with your login credentials/profile shortly.",
            "diagnosis": "Non-Technical / Account Management Inquiry",
            "root_cause": "N/A (Account Access)"
        },
        "billing / payment": {
            "decision": "ROUTE_TO_BILLING_SUPPORT",
            "solution": "Your billing and invoice inquiry has been routed to our Billing Operations desk. We will review your charges and payment history and contact you shortly.",
            "diagnosis": "Non-Technical / Billing & Payments Inquiry",
            "root_cause": "N/A (Billing Inquiry)"
        },
        "cancellation": {
            "decision": "ROUTE_TO_RETENTION_TEAM",
            "solution": "Your account cancellation request has been received and routed to our Customer Relations desk. A representative will contact you to confirm termination details.",
            "diagnosis": "Non-Technical / Service Cancellation Request",
            "root_cause": "N/A (Service Termination)"
        },
        "customer support": {
            "decision": "ROUTE_TO_GENERAL_SUPPORT",
            "solution": "Your request has been routed to our general Customer Support team. A service representative will reach out to you shortly.",
            "diagnosis": "Non-Technical / General Support Request",
            "root_cause": "N/A (Support Inquiry)"
        },
        "service / plan": {
            "decision": "ROUTE_TO_SALES_SUPPORT",
            "solution": "Your plan change or subscription inquiry has been routed to our Service Management team. A sales representative will help you review plan validity and upgrade options.",
            "diagnosis": "Non-Technical / Plan & Subscription Inquiry",
            "root_cause": "N/A (Service Upgrade/Plan)"
        }
    }

    if category_lower in non_tech_info:
        info = non_tech_info[category_lower]
        return {
            "complaint": complaint_text,
            "category": category,
            "category_confidence": category_confidence,
            "negativity_score": negativity_score,
            "sentiment_score": round(negativity_score * 100.0, 2),
            "extraction_source": "none",
            "lowest_confidence": 1.0,
            "technical_information": {
                "component": ["none"],
                "failure_type": ["none"],
                "scope": "individual",
                "service_impact": "none",
                "duration_hours": None,
                "occurrence_pattern": "none",
            },
            "complexity": "OTHER",
            "complexity_score": 0,
            "base_complexity": "OTHER",
            "modifier": 0,
            "critical_override": False,
            "decision_reason": f"Category classified as '{category}'. Technical complexity scoring bypassed.",
            "weighted_complexity_score": 0.0,
            "weighted_negativity_score": 0.0,
            "total_complexity_score": 0.0,
            "solution_a": info["solution"],
            "solution_high": None,
            "warnings": [],
            "evidence": [],
            "confidence_score": round(category_confidence, 2),
            "diagnosis": info["diagnosis"],
            "root_cause": info["root_cause"],
            "risk_level": "OTHER",
            "policy_status": "STANDARD",
            "final_decision": info["decision"],
            "critic_feedback": "Non-technical inquiry; no engineering escalation required.",
        }

    # 3. Information Extraction (ML / LangGraph Agent)
    extraction_result = extract_technical_information(complaint_text)
    tech_info = extraction_result["technical_information"]
    extraction_source = extraction_result["extraction_source"]
    lowest_confidence = extraction_result["lowest_confidence"]

    # 4. Priority & Complexity Engine
    llm_complexity = tech_info.get("complexity")

    if llm_complexity and llm_complexity != "unknown":
        complexity = llm_complexity
        complexity_score = 0
        if complexity == "LOW":
            complexity_score = 25
        elif complexity == "MEDIUM":
            complexity_score = 50
        elif complexity == "HIGH":
            complexity_score = 75
        elif complexity == "CRITICAL":
            complexity_score = 100
        
        complexity_result = {
            "complexity": complexity,
            "complexity_score": complexity_score,
            "base_complexity": complexity,
            "modifier": 0,
            "critical_override": complexity == "CRITICAL",
            "decision_reason": f"Complexity classified directly by LLM model to {complexity}.",
        }
    else:
        complexity_result = calculate_complexity(tech_info)
        complexity = complexity_result["complexity"]

    if complexity == "OTHER":
        return {
            "complaint": complaint_text,
            "category": category,
            "category_confidence": category_confidence,
            "negativity_score": negativity_score,
            "sentiment_score": round(negativity_score * 100.0, 2),
            "extraction_source": extraction_source,
            "lowest_confidence": lowest_confidence,
            "technical_information": tech_info,
            "complexity": "OTHER",
            "complexity_score": 0,
            "base_complexity": "OTHER",
            "modifier": 0,
            "critical_override": False,
            "decision_reason": complexity_result.get("decision_reason") or "Bypassed technical complexity scoring.",
            "weighted_complexity_score": 0.0,
            "weighted_negativity_score": 0.0,
            "total_complexity_score": 0.0,
            "solution_a": "No active technical failure was detected in this complaint. Your request has been routed to our Customer Care team for general support.",
            "solution_high": None,
            "warnings": [],
            "evidence": [],
            "confidence_score": 0.90,
            "diagnosis": "Non-Technical / General Inquiry",
            "root_cause": "N/A (No active issue reported)",
            "risk_level": "OTHER",
            "policy_status": "STANDARD",
            "final_decision": "ROUTE_TO_GENERAL_SUPPORT",
            "critic_feedback": "No active technical failure reported.",
        }

    # 5. Total Complexity Calculation (85% Complexity + 15% Sentiment Negativity)
    total_complexity = calculate_total_complexity(
        complexity_result["complexity_score"], negativity_score
    )

    # 6. Multi-Agent Solution Generation
    solution_a = None
    solution_high = None
    warnings = []
    evidence = []
    confidence_score = 0.85
    diagnosis = None
    root_cause = None
    risk_level = None
    policy_status = None
    final_decision = None
    critic_feedback = None

    if complexity in ["LOW", "MEDIUM"]:
        try:
            sol_state = {
                "complaint_input": {
                    "complaint": complaint_text,
                    "complaint_id": "AUTO",
                    "complexity": complexity,
                    "technical_information": tech_info,
                    "category": category,
                }
            }
            sol_res = solution_graph.invoke(sol_state)
            final_sol = sol_res.get("final_solution", {})

            if "customer_instructions" in final_sol:
                inst_list = final_sol.get("customer_instructions", [])
                solution_a = "\n".join(inst_list) if isinstance(inst_list, list) else str(inst_list)
            elif "recommended_actions" in final_sol:
                actions = final_sol.get("recommended_actions", [])
                solution_a = (
                    "\n".join(actions)
                    if isinstance(actions, list)
                    else str(actions)
                )
            else:
                solution_a = final_sol.get(
                    "summary", "Standard self-care steps applied."
                )

            warnings = final_sol.get("warnings", [])
            evidence = final_sol.get("evidence", [])
            confidence_score = float(final_sol.get("confidence", 0.90))
        except Exception as e:
            print(f"⚠️ Solution Agent fallback: {e}")
            solution_a = "1. Power cycle device for 30 seconds.\n2. Verify optical cable link light.\n3. Check subscription status."

    else:
        # HIGH or CRITICAL Complexity
        try:
            high_input = {
                "complaint": complaint_text,
                "complaint_text": complaint_text,
                "complexity": complexity,
                "category": category,
                "technical_information": tech_info,
                "scope": tech_info.get("scope", "individual"),
                "duration_hours": tech_info.get("duration_hours", 24.0),
                "severity": complexity.lower(),
            }
            high_res = run_high_agent(high_input)
            diagnosis = high_res.get(
                "diagnosis", "Network Infrastructure Disruption"
            )
            root_cause = high_res.get("root_cause", "Hardware/Fiber Anomaly")
            risk_level = high_res.get("risk_level", "HIGH")
            policy_status = high_res.get("policy_status", "ELEVATED")
            final_decision = high_res.get(
                "final_decision", "DISPATCH_FIELD_TECH"
            )
            critic_feedback = high_res.get("critic_reason", "")
            solution_high = high_res.get(
                "proposed_action"
            ) or high_res.get("solution_high", "Field engineering dispatched.")
            confidence_score = float(high_res.get("confidence_score", 0.95))
            solution_a = solution_high  # Fallback for display
        except Exception as e:
            print(f"⚠️ High Agent fallback: {e}")
            diagnosis = "Critical Network Incident"
            root_cause = "Hardware / Physical Line Failure"
            solution_high = "Urgent Field Intervention: Emergency technician dispatched for on-site inspection."
            solution_a = solution_high
            final_decision = "DISPATCH_FIELD_TECH"

    return {
        "complaint": complaint_text,
        "category": category,
        "category_confidence": category_confidence,
        "negativity_score": negativity_score,
        "sentiment_score": total_complexity["sentiment_score"],
        "extraction_source": extraction_source,
        "lowest_confidence": lowest_confidence,
        "technical_information": tech_info,
        "complexity": complexity_result["complexity"],
        "complexity_score": complexity_result["complexity_score"],
        "base_complexity": complexity_result.get("base_complexity"),
        "modifier": complexity_result.get("modifier", 0),
        "critical_override": complexity_result.get("critical_override", False),
        "decision_reason": complexity_result.get("decision_reason"),
        "weighted_complexity_score": total_complexity[
            "weighted_complexity_score"
        ],
        "weighted_negativity_score": total_complexity[
            "weighted_negativity_score"
        ],
        "total_complexity_score": total_complexity["total_complexity_score"],
        # Multi-Agent Solutions & Diagnosis
        "solution_a": solution_a,
        "solution_high": solution_high,
        "warnings": warnings,
        "evidence": evidence,
        "confidence_score": confidence_score,
        "diagnosis": diagnosis,
        "root_cause": root_cause,
        "risk_level": risk_level,
        "policy_status": policy_status,
        "final_decision": final_decision,
        "critic_feedback": critic_feedback,
    }
