import json
from typing import Dict, Any
from app.agents.solution.agents.llm import get_llm_response

def analyze_complaint(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes the raw complaint text, extracts symptoms, and identifies root causes.
    Combines the previous Understanding and Root Cause Analysis logic.
    """
    complaint = state["complaint_input"]
    complaint_text = complaint.get("complaint", "")
    priority = complaint.get("complexity", "LOW")
    
    tech_info = complaint.get("technical_information") or {}
    complexity_score = complaint.get("complexity_score", 0)
    negativity_score = complaint.get("negativity_score", 0)
    
    system_prompt = f"""
    You are the Complaint Analysis Agent.
    Your task is to analyze a telecom complaint, extract structured information, and identify root causes based on the provided context.
    
    Priority Mode: {priority}
    For MEDIUM priority, you MUST heavily weigh technical_information, complexity_score, and negativity_score.
    If the negativity score is high, ensure the analysis highlights the urgency.
    
    CRITICAL: You MUST output ONLY a single valid JSON object starting with {{ and ending with }}. Do not output lists of arrays.
    
    Output JSON format:
    {{
      "problem_summary": "Short summary of the issue",
      "symptoms": ["list", "of", "symptoms"],
      "likely_causes": ["..."],
      "alternative_causes": ["..."],
      "evidence_supporting": ["..."],
      "evidence_against": ["..."]
    }}
    """
    
    user_prompt = f"""
    Complaint Text: {complaint_text}
    
    Technical Information: {json.dumps(tech_info)}
    Complexity Score: {complexity_score}
    Negativity Score: {negativity_score}
    """
    
    response = get_llm_response(system_prompt, user_prompt, json_mode=True)
    
    if "error" not in response:
        state["analysis"] = response
    else:
        state["analysis"] = {
            "problem_summary": "Error analyzing complaint.",
            "symptoms": [],
            "likely_causes": [],
            "alternative_causes": [],
            "evidence_supporting": [],
            "evidence_against": []
        }
        
    return state
