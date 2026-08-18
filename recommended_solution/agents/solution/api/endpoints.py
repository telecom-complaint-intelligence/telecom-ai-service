from fastapi import APIRouter, HTTPException
from agents.solution.models.schemas import ComplaintInput, LowSolution, MediumSolution
from agents.solution.graph.solution_graph import solution_graph

router = APIRouter()

@router.post("/solution/low", response_model=LowSolution)
async def get_low_solution(complaint: ComplaintInput):
    if complaint.priority != "LOW":
        raise HTTPException(status_code=400, detail="Priority must be LOW for this endpoint")
        
    initial_state = {"complaint_input": complaint.model_dump()}
    final_state = solution_graph.invoke(initial_state)
    
    solution_data = final_state.get("final_solution", {})
    if "error" in solution_data:
        raise HTTPException(status_code=500, detail=solution_data["error"])
        
    return LowSolution(**solution_data)

@router.post("/solution/medium", response_model=MediumSolution)
async def get_medium_solution(complaint: ComplaintInput):
    if complaint.priority != "MEDIUM":
        raise HTTPException(status_code=400, detail="Priority must be MEDIUM for this endpoint")
        
    initial_state = {"complaint_input": complaint.model_dump()}
    final_state = solution_graph.invoke(initial_state)
    
    solution_data = final_state.get("final_solution", {})
    if "error" in solution_data:
        raise HTTPException(status_code=500, detail=solution_data["error"])
        
    return MediumSolution(**solution_data)
