import json
import sys
import os
from typing import Dict, Any

from agents.solution.graph.solution_graph import solution_graph
from agents.escalation.schemas.input_schema import EscalationInput
from agents.escalation.agents.escalation_agent import run_escalation_agent
from agents.high.api import run_high_agent

def main():
    if len(sys.argv) < 2:
        print("Usage: python orchestrator.py <input.json>")
        sys.exit(1)

    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found.")
        sys.exit(1)

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            complaint_data = json.load(f)
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        sys.exit(1)

    complexity = str(complaint_data.get("complexity", "LOW")).upper()
    print(f"\n🚀 Starting Orchestrator Pipeline for {complexity} complaint...")

    # HIGH Pipeline
    if complexity == "HIGH":
        print("\n[Orchestrator] Directing to HIGH Agent...")
        high_result = run_high_agent(complaint_data)
        print("\n=== HIGH AGENT FINAL REPORT ===")
        print(f"Diagnosis: {high_result.get('diagnosis')}")
        print(f"Root Cause: {high_result.get('root_cause')}")
        print(f"Priority: {high_result.get('priority')}")
        print(f"Final Decision: {high_result.get('final_decision')}")
        print(f"Confidence: {high_result.get('confidence_score')}")
        sys.exit(0)

    # LOW/MEDIUM Pipeline -> Solution Agent
    print(f"\n[Orchestrator] Step 1: Running Solution Agent ({complexity})...")
    solution_state = {"complaint_input": complaint_data}
    solution_result = solution_graph.invoke(solution_state)
    
    final_solution = solution_result.get("final_solution", {})
    print("\n[Orchestrator] Solution Generated.")
    print(json.dumps(final_solution, indent=2))

    # Escalation Pipeline
    print("\n[Orchestrator] Step 2: Running Escalation Agent...")
    
    # Extract mock fields from input for escalation testing
    # Simulated customer feedback (default "yes" if not provided to simulate success)
    customer_feedback = complaint_data.get("customer_feedback", "yes")
    age_in_days = int(complaint_data.get("age_in_days", 1))
    category_complaint_count = int(complaint_data.get("category_complaint_count", 1))
    
    escalation_input = EscalationInput(
        complaint=complaint_data.get("complaint", complaint_data.get("complaint_text", "")),
        current_severity=complexity,
        customer_feedback=customer_feedback,
        solution_agent_output=json.dumps(final_solution),
        category=complaint_data.get("category", "General"),
        technical_information=complaint_data.get("technical_information", "None"),
        complexity=complexity,
        complexity_score=complaint_data.get("complexity_score", 0.5),
        weighted_negativity_score=complaint_data.get("weighted_negativity_score", 0.5),
        age_in_days=age_in_days,
        category_complaint_count=category_complaint_count
    )

    escalation_result = run_escalation_agent(escalation_input)
    next_severity = escalation_result.next_severity
    
    print("\n=== ESCALATION RESULT ===")
    print(f"Original Severity: {complexity}")
    print(f"Next Severity:     {next_severity}")
    print(f"Escalated?         {escalation_result.escalated}")
    print(f"Reasoning:         {escalation_result.reasoning}")

    if next_severity == "HIGH":
        print("\n[Orchestrator] Step 3: Escalated to HIGH! Running High Agent...")
        # Add the escalation reasoning to the complaint data so the High Agent has context
        complaint_data["escalation_reasoning"] = escalation_result.reasoning
        high_result = run_high_agent(complaint_data)
        print("\n=== HIGH AGENT FINAL REPORT ===")
        print(f"Diagnosis: {high_result.get('diagnosis')}")
        print(f"Root Cause: {high_result.get('root_cause')}")
        print(f"Priority: {high_result.get('priority')}")
        print(f"Final Decision: {high_result.get('final_decision')}")
        print(f"Confidence: {high_result.get('confidence_score')}")

    elif next_severity == "MEDIUM" and complexity == "LOW":
        print("\n[Orchestrator] Complaint escalated from LOW to MEDIUM. It will be picked up by the MEDIUM pipeline in the next cycle.")
    else:
        print("\n[Orchestrator] Pipeline completed. No further escalation.")

if __name__ == "__main__":
    main()
