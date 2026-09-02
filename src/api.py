from fastapi import FastAPI
from pydantic import BaseModel

from .model import load_model


app = FastAPI(
    title="Resume-Job Matching API",
    description="NLP-based Resume–Job Matching and Ranking System",
    version="1.0.0"
)


# Load trained model once when the API starts
model, features = load_model()


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


@app.get("/model-info")
def model_info():
    return {
        "model": "Logistic Regression V2",
        "number_of_features": len(features),
        "features": features
    }
