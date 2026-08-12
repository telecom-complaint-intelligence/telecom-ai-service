from fastapi import FastAPI

app = FastAPI(title="Telecom Complaint Intelligence & Automated Resolution Assistant - AI Service")

@app.get("/")
def read_root():
    return {"message": "Hello from telecom-ai-service API!"}

@app.get("/health")
def health_check():
    return {"status": "ok"}
