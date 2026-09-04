from fastapi import FastAPI
from pydantic import BaseModel

from .model import load_model
from .pipeline import build_candidate_features

app = FastAPI(
    title="Resume-Job Matching API",
    description="NLP-based Resume–Job Matching and Ranking System",
    version="1.0.0"
)


# Load trained model once when the API starts
model, features = load_model()


class HealthResponse(BaseModel):
    status: str


class ResumeInput(BaseModel):
    resume_id: str
    role: str
    seniority: str
    years_experience: float
    industry: str
    skills: list[str]
    text: str


class JobInput(BaseModel):
    job_id: str
    job_title: str
    seniority: str
    industry: str
    must_have_skills: list[str]
    nice_to_have_skills: list[str]
    text: str


class MatchCandidatesRequest(BaseModel):
    job: JobInput
    resumes: list[ResumeInput]


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


@app.post("/predict")
def predict(request: MatchRequest):

    import pandas as pd

    X = pd.DataFrame(
        [[request.feature_values[feature] for feature in features]],
        columns=features
    )

    probability = model.predict_proba(X)[0, 1]

    return {
        "match_score": round(float(probability), 4)
    }
@app.post("/match")
def match_candidates(request: MatchCandidatesRequest):

    return {
        "job_id": request.job.job_id,
        "number_of_resumes": len(request.resumes),
        "message": "Matching endpoint is ready."
    }
