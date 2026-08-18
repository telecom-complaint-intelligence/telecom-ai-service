from fastapi import FastAPI, HTTPException, status
from agents.escalation.schemas.input_schema import EscalationInput
from agents.escalation.schemas.output_schema import EscalationOutput
from agents.escalation.agents.escalation_agent import run_escalation_agent
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("escalation_agent")
app = FastAPI(
    title="Escalation Agent API",
    description="FastAPI service for the LangGraph-based customer complaint escalation agent",
    version="1.0.0"
)
@app.post(
    "/api/v1/escalation/evaluate",
    response_model=EscalationOutput,
    status_code=status.HTTP_200_OK,
    summary="Evaluate complaint outcome",
    description="Processes complaint, category, severity, solution, customer response and decides on the next action based on policies and guardrails."
)
def evaluate_complaint(payload: EscalationInput):
    try:
        logger.info(f"Received evaluation request: severity={payload.current_severity}")
        output = run_escalation_agent(payload)
        return output
    except ValueError as ve:
        logger.error(f"Validation/Input Error during execution: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation/Input Error: {str(ve)}"
        )
    except Exception as e:
        logger.error(f"Agent Execution Failure: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Agent Execution Failure (LLM/Service down): {str(e)}"
        )
@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "healthy", "service": "escalation-agent"}
