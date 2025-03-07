from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class ServiceStatus(BaseModel):
    service_name: str
    status: str
    version: str = "1.0.0"

@app.get("/")
async def root():
    return {"message": "Service is running"}

@app.get("/status")
async def status():
    return ServiceStatus(
        service_name="QuantumService",
        status="operational"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)