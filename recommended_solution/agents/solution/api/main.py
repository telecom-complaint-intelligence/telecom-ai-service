from fastapi import FastAPI
from agents.solution.api.endpoints import router
import uvicorn

app = FastAPI(title="Telecom Solution Agent API", version="0.1.0")

app.include_router(router)

@app.get("/")
def root():
    return {"message": "Welcome to the Telecom Solution Agent API Prototype"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
