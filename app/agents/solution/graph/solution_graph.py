from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.agents.solution.agents.analysis import analyze_complaint
from app.agents.solution.agents.knowledge import retrieve_knowledge
from app.agents.solution.agents.solution import synthesize_solution


class GraphState(TypedDict):
    complaint_input: dict[str, Any]
    retrieved_knowledge: list
    analysis: dict[str, Any]
    final_solution: dict[str, Any]

def create_solution_graph():
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("retrieve_kb", retrieve_knowledge)
    workflow.add_node("analyze", analyze_complaint)
    workflow.add_node("synthesize", synthesize_solution)
    
    # Define edges
    # We sequence them: retrieve_kb -> analyze -> synthesize
    workflow.set_entry_point("retrieve_kb")
    workflow.add_edge("retrieve_kb", "analyze")
    workflow.add_edge("analyze", "synthesize")
    workflow.add_edge("synthesize", END)
    
    return workflow.compile()

solution_graph = create_solution_graph()
