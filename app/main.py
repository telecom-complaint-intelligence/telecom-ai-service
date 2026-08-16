from fastapi import FastAPI
from app.api.inference_routes import router as inference_router

app = FastAPI(
    title="Telecom AI Service",
    description="Stateless AI/ML/NLP Inference Microservice for Complaint Categorization, Sentiment Analysis, Technical Extraction, Priority Routing, and LangGraph Triage.",
    version="1.0.0"
)

app.include_router(inference_router)

@app.get("/")
def read_root():
    return {
        "status": "running",
        "service": "telecom-ai-service",
        "port": 8001,
        "docs": "/docs",
        "endpoints": ["/api/v1/analyze", "/api/v1/categorize", "/api/v1/sentiment", "/api/v1/extract"]
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}
