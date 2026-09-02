from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Resume-Job Matching API",
    description="NLP-based Resume–Job Matching and Ranking System",
    version="1.0.0"
)


class HealthResponse(BaseModel):
    status: str


@app.get("/", response_model=HealthResponse)
def root():
    return {
        "status": "Resume-Job Matching API is running"
    }


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "healthy"
    }
