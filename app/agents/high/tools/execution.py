"""
tools/execution.py

Intervention Execution Engine
Safely executes and dispatches authorized telecom operational actions,
field engineering work orders, SLA escalations, and customer care interventions.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import uuid


def execute_telecom_intervention(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the authorized final intervention plan.

    Parameters
    ----------
    state : Dict[str, Any]
        Current LangGraph state with final decision, action, and authorization token.

    Returns
    -------
    Dict[str, Any]
        Execution receipt and operational dispatch summary.
    """
    print()
    print("=" * 60)
    print("TOOL EXECUTION: DISPATCHING TELECOM INTERVENTION")
    print("=" * 60)

    is_authorized = state.get("execution_authorized", False)
    auth_token = state.get("authorization_token", "UNAUTHORIZED")
    final_action = state.get("final_action", "No action specified")
    final_decision = state.get("final_decision", "NORMAL")
    final_priority = state.get("final_priority", "MEDIUM")
    customer_id = state.get("customer_id", "UNKNOWN")
    complaint_id = state.get("complaint_id", "UNKNOWN")

    if not is_authorized:
        print("❌ EXECUTION BLOCKED: Execution Guardrail has not authorized this intervention.")
        return {
            "status": "BLOCKED",
            "message": "Execution blocked by Execution Guardrail.",
            "authorization_token": None,
            "timestamp": datetime.now().isoformat(),
        }

    action_lower = str(final_action).lower()
    dispatch_type = "STANDARD_OPERATION"
    dispatch_details: Dict[str, Any] = {}

    # 1. Human On-Site Field Intervention / Dispatch
    if any(k in action_lower for k in ["dispatch", "field", "crew", "technician", "on-site", "tower", "cable", "repair"]):
        dispatch_type = "EMERGENCY_FIELD_DISPATCH"
        dispatch_details = {
            "work_order_id": f"WO-FIELD-{uuid.uuid4().hex[:6].upper()}",
            "assigned_crew": "North District Emergency Fiber & Rigging Unit",
            "priority": final_priority,
            "sla_target_hours": 4 if final_priority == "CRITICAL" else 12,
            "equipment_required": ["Fiber Splicer", "OTDR Analyzer", "Replacement SFP Module / Router"],
            "status": "DISPATCHED_TO_FIELD_CREW"
        }

    # 2. Senior Tier-2 / Tier-3 Engineering Escalation
    elif any(k in action_lower for k in ["tier-2", "tier-3", "engineering", "escalat", "trace", "investigat"]):
        dispatch_type = "TIER2_ENGINEERING_ESCALATION"
        dispatch_details = {
            "escalation_ticket_id": f"ENG-ESC-{uuid.uuid4().hex[:6].upper()}",
            "assigned_team": "Core Network Reliability & Infrastructure Team",
            "priority": final_priority,
            "target_sla": "2 Hours" if final_priority == "CRITICAL" else "8 Hours",
            "status": "ASSIGNED_TO_SENIOR_ENGINEER"
        }

    # 3. Customer Retention & Executive Care
    elif any(k in action_lower for k in ["retention", "executive", "care", "contact", "callback", "outreach"]):
        dispatch_type = "EXECUTIVE_CUSTOMER_CARE"
        dispatch_details = {
            "case_id": f"CARE-VIP-{uuid.uuid4().hex[:6].upper()}",
            "assigned_agent": "Priority Customer Retention Specialist",
            "outreach_channel": "Direct Priority Phone & SMS",
            "status": "SCHEDULED_OUTREACH"
        }

    else:
        dispatch_type = "AUTOMATED_CRM_UPDATE"
        dispatch_details = {
            "ticket_id": f"CRM-ACT-{uuid.uuid4().hex[:6].upper()}",
            "action_logged": final_action,
            "status": "COMPLETED"
        }

    execution_receipt = {
        "execution_id": f"EXEC-{uuid.uuid4().hex[:8].upper()}",
        "status": "SUCCESSFULLY_EXECUTED",
        "authorization_token": auth_token,
        "dispatch_type": dispatch_type,
        "target_ticket": customer_id,
        "complaint_id": complaint_id,
        "executed_action": final_action,
        "priority_level": final_priority,
        "dispatch_details": dispatch_details,
        "timestamp": datetime.now().isoformat(),
    }

    print(f"✅ EXECUTION SUCCESS: Receipt ID {execution_receipt['execution_id']}")
    print(f"   Dispatch Type : {dispatch_type}")
    print(f"   Ticket Target : {customer_id}")
    print(f"   Auth Token    : {auth_token}")

    return execution_receipt
