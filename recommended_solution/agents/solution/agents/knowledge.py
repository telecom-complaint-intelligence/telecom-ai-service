from typing import Dict, Any
from agents.solution.knowledge.retriever import VectorKnowledgeRetriever

retriever = VectorKnowledgeRetriever()

def retrieve_knowledge(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieves knowledge base articles based on the complaint inputs.
    """
    complaint = state["complaint_input"]
    
    category = complaint.get("category", "")
    tech_info = complaint.get("technical_information", {})
    
    components = tech_info.get("component", [])
    component = components[0] if components else ""
    
    failure_types = tech_info.get("failure_type", [])
    failure_type = failure_types[0] if failure_types else ""
    
    docs = retriever.search(
        domain=category,
        component=component,
        failure_type=failure_type,
        complaint_text=complaint.get("complaint", ""),
        top_k=3
    )
    
    state["retrieved_knowledge"] = docs
    return state
