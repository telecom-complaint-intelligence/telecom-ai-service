import json
from typing import Any

from app.agents.solution.agents.llm import get_llm_response


def synthesize_solution(state: dict[str, Any]) -> dict[str, Any]:
    """
    Synthesizes the final solution matching the LowSolution or MediumSolution schema.
    Applies safety checks inherently without needing a separate critic.
    """
    complaint = state["complaint_input"]
    priority = complaint.get("complexity", "LOW")
    complaint_id = str(complaint.get("complaint_id", "UNKNOWN"))
    
    analysis = state.get("analysis", {})
    knowledge = state.get("retrieved_knowledge", [])
    
    evidence_list = [{"knowledge_id": k["id"], "title": k["title"]} for k in knowledge]
    kb_text = "\n".join([f"ID: {k['id']}, Title: {k['title']}, Safe: {k['customer_safe']}\nContent: {k['content']}" for k in knowledge])
    
    if priority == "LOW":
        system_prompt = """
        You are the Solution Agent for LOW priority complaints.
        Format the final response for the customer based on the analysis and retrieved knowledge.
        
        CRITICAL SAFETY RULE: Only suggest actions that are safe for a non-technical customer. 
        Reject anything requiring technical access, backend changes, or destructive actions (like factory reset without explicit KB support).
        
        CRITICAL: You MUST output ONLY a single valid JSON object starting with { and ending with }.
        
        Output JSON matching exactly this schema:
        {
          "complaint_id": "...",
          "summary": "...",
          "customer_instructions": ["step 1", "step 2"],
          "expected_result": "...",
          "warnings": ["..."],
          "confidence": 0.0 to 1.0,
          "status": "READY"
        }
        """
    else:
        system_prompt = """
        You are the Solution Agent for MEDIUM priority complaints.
        Format the final response for internal technical teams based on the analysis and retrieved knowledge.
        Ensure recommendations use the supplied technical information.
        
        CRITICAL: You MUST output ONLY a single valid JSON object starting with { and ending with }.
        
        Output JSON matching exactly this schema:
        {
          "complaint_id": "...",
          "problem_summary": "...",
          "probable_causes": ["..."],
          "diagnostic_findings": ["..."],
          "recommended_actions": ["..."],
          "previous_resolution_analysis": "...",
          "rationale": "...",
          "confidence": 0.0 to 1.0,
          "risks": ["..."],
          "status": "READY"
        }
        """
        
    user_prompt = f"""
    Complaint ID: {complaint_id}
    
    Analysis (Summary, Symptoms, Root Causes): {json.dumps(analysis)}
    
    Retrieved Knowledge:
    {kb_text}
    """
    
    response = get_llm_response(system_prompt, user_prompt, json_mode=True)
    
    if "error" not in response:
        response["evidence"] = evidence_list
        response["complaint_id"] = complaint_id
        if "status" not in response:
            response["status"] = "READY"
        state["final_solution"] = response
    else:
        state["final_solution"] = {"error": "Failed to synthesize solution."}
        
    return state
