from typing import Any, Dict
from app.agents.solution.knowledge.retriever import VectorKnowledgeRetriever

retriever = VectorKnowledgeRetriever()


def retrieve_knowledge(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieves knowledge base articles based on the complaint inputs.
    """
    complaint = state["complaint_input"]

    category = complaint.get("category", "")
    tech_info = complaint.get("technical_information") or {}
    if isinstance(tech_info, str):
        tech_info = {"component": [tech_info]}

    components = tech_info.get("component", [])
    component = (
        components[0]
        if isinstance(components, list) and components
        else str(components)
    )

    failure_types = tech_info.get("failure_type", [])
    failure_type = (
        failure_types[0]
        if isinstance(failure_types, list) and failure_types
        else str(failure_types)
    )

    docs = retriever.search(
        domain=category,
        component=component,
        failure_type=failure_type,
        complaint_text=complaint.get("complaint", ""),
        top_k=3,
    )

    state["retrieved_knowledge"] = docs
    return state
