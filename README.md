# NLP Resume–Job Matching & Ranking System

An end-to-end NLP-based system for matching candidates with job opportunities and ranking the most relevant resumes for a given job description.

The system combines **semantic similarity**, **structured candidate-job features**, and a **learned ranking model** to retrieve and rank candidates. It also provides **explainable matching information** and an optional **LLM-powered explanation layer**.

---

## 🚀 Project Overview

Recruiters often need to evaluate a large number of resumes against job descriptions. Traditional keyword-based matching can miss candidates whose experience is semantically relevant but expressed using different terminology.

This project addresses the problem as a **ranking task** rather than a simple classification problem.

Given a job description, the system:

1. Represents jobs and resumes using semantic embeddings.
2. Retrieves a candidate pool using semantic similarity.
3. Extracts structured matching features.
4. Applies a learned ranking model.
5. Produces a ranked list of candidates.
6. Provides interpretable matching information.
7. Optionally generates natural-language explanations using a local LLM.

---

## 🏗️ System Architecture

```text
Resume / Job Data
        │
        ▼
   Text Processing
        │
        ▼
 Sentence Embeddings
        │
        ▼
 Semantic Retrieval
        │
        ▼
 Candidate Pool
        │
        ▼
 Feature Engineering
        │
        ├── Semantic Similarity
        ├── Skill Matching
        ├── Experience Matching
        ├── Seniority Matching
        ├── Industry Matching
        └── Skill Coverage Features
        │
        ▼
 Learned Ranking Model
        │
        ▼
   Top-K Candidates
        │
        ▼
   Explainability
        │
        ▼
 Optional LLM Explanation
```

---

## 🎯 Problem Formulation

The system is designed as a **Resume–Job Ranking Problem**.

For a given job:

$$
Job \rightarrow Ranked\ Candidates
$$

Instead of predicting only whether a resume is relevant or irrelevant, the system assigns a ranking score to candidates and returns the most relevant resumes at the top.

---

## 📊 Dataset

The project uses the following synthetic resume–job matching dataset:

**Hugging Face Dataset:** `michaelozon/candidate-matching-synthetic`

The dataset contains:

* 10,000 resumes
* 2,500 jobs
* 2,500 matching records
* 24 job roles
* 10 industries
* 73 skills
* Seniority information
* Years of experience
* Structured and unstructured resume/job information

### Working Dataset

For experimentation and development, the project uses:

* **2,000 resumes**
* **150 jobs**

The resume dataset contains fields such as:

```text
resume_id
role
seniority
years_experience
industry
education
skills
summary
experience_bullets
text
```

The job dataset contains:

```text
job_id
job_title
seniority
industry
must_have_skills
nice_to_have_skills
description
responsibilities
requirements
text
```

---

## 🧠 NLP & Semantic Representation

The project uses the Sentence Transformers framework with:

```text
all-MiniLM-L6-v2
```

Each resume and job description is converted into a **384-dimensional embedding**.

```text
Resume → 384-dimensional vector
Job    → 384-dimensional vector
```

Cosine similarity is then used to measure semantic similarity between jobs and resumes.

```python
similarity_matrix = cosine_similarity(
    job_embeddings,
    resume_embeddings
)
```

The resulting similarity matrix has the shape:

```text
(150 jobs, 2000 resumes)
```

---

## 🔎 Semantic Retrieval

Semantic similarity is first used to retrieve a candidate pool before applying the more detailed ranking model.

The retrieval stage initially used:

```text
Top-700 candidates
```

The retrieval evaluation showed:

| Recall@K   |  Score |
| ---------- | -----: |
| Recall@10  | 0.0563 |
| Recall@50  | 0.2027 |
| Recall@100 | 0.3813 |
| Recall@200 | 0.6164 |
| Recall@500 | 0.8957 |
| Recall@700 | 0.9406 |

A validation-only evaluation showed:

| Recall@K   | Validation Score |
| ---------- | ---------------: |
| Recall@100 |           0.4031 |
| Recall@200 |           0.6367 |
| Recall@500 |           0.9281 |
| Recall@700 |           0.9741 |

This demonstrates that semantic retrieval is effective at producing a high-quality candidate pool, while the ranking stage is responsible for ordering those candidates.

---

## ⚙️ Feature Engineering

The ranking model combines semantic and structured features.

### Final Feature Set

The final model uses 12 features:

```text
semantic_score
skill_score
experience_score
experience_gap
seniority_score
seniority_gap
industry_score
missing_must_have_count
must_have_coverage
nice_to_have_coverage
exact_must_have_match_count
skill_count
```

### Skill Features

The system evaluates:

* Must-have skill coverage
* Nice-to-have skill coverage
* Exact must-have matches
* Missing must-have skills
* Overall skill matching

### Experience Features

The system considers:

* Candidate years of experience
* Experience gap relative to job seniority

### Seniority Features

Candidate and job seniority levels are compared using:

```text
Junior
Mid
Senior
```

### Industry Features

Candidate and job industries are compared to provide an additional matching signal.

---

## 🤖 Ranking Models

Several ranking approaches were evaluated during development.

### 1. Semantic Baseline

Uses semantic similarity directly for ranking.

### 2. Manual Hybrid

Combines semantic, skill, experience, seniority, and industry signals using manually selected weights.

### 3. Logistic Regression V1

A learned ranking score based on the initial feature set.

### 4. Logistic Regression V2

The final selected model using the engineered 12-feature representation.

### 5. LambdaRank

A learning-to-rank approach using LightGBM.

### 6. Pairwise Logistic Ranking

A pairwise ranking approach trained on positive and negative candidate pairs.

---

## 📈 Model Comparison

| Model                      | Precision@10 |  Recall@10 |    NDCG@10 |
| -------------------------- | -----------: | ---------: | ---------: |
| Semantic Baseline          |       0.0333 |     0.0579 |     0.1469 |
| Manual Hybrid              |       0.0300 |     0.0500 |     0.1328 |
| Logistic Regression V1     |       0.0400 |     0.0790 |     0.1804 |
| **Logistic Regression V2** |   **0.0467** | **0.0929** | **0.2228** |
| LambdaRank                 |       0.0367 |     0.0614 |     0.1456 |
| Pairwise Logistic          |       0.0467 |     0.0887 | **0.2246** |

---

## 🏆 Final Model

The selected **Champion Model** is:

```text
Logistic Regression V2
```

It was selected based on the overall ranking performance, particularly its stronger Recall@10 while maintaining a simple and reproducible architecture.

### Final Validation Performance

```text
Precision@10 : 0.0467
Recall@10    : 0.0929
NDCG@10      : 0.2228
MRR@10       : 0.1619
MAP@10       : 0.0342
```

The final model uses 12 engineered features combining semantic and structured matching signals.

---

## 💡 Explainability

The system does not only return a ranking score.

For every candidate, it can provide information such as:

* Candidate role
* Seniority
* Industry
* Years of experience
* Must-have skill coverage
* Matched skills
* Missing must-have skills

Example output:

```text
Rank: 1
Resume ID: R_007142
Match Score: 97.13%

Role: Customer Success Associate
Seniority: Mid
Industry: E-commerce
Years Experience: 3

Must-Have Coverage: 100%

Matched Skills:
- Escalations
- Customer Satisfaction
- Root Cause Analysis
- Ticketing

Missing Skills:
None
```

This makes the ranking more interpretable than a black-box score alone.

---

## 🧠 LLM-Powered Explanation

An optional local LLM layer is included to transform structured matching information into a natural-language explanation.

The project uses:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

The LLM is **not used as the core ranking model**.

Instead:

```text
Ranking Model
      ↓
Structured Explanation
      ↓
Local LLM
      ↓
Natural-Language Explanation
```

This separation keeps the core ranking system:

* Reproducible
* Quantitatively evaluable
* Independent of LLM generation
* Easier to debug

The structured explanation remains the trusted source of matching information.

---

## 🔄 End-to-End Example

A job can be passed to the final ranking pipeline:

```python
final_output = get_final_ranking(
    job_id="J_002480",
    top_k=10
)

final_output
```

The system returns a ranked DataFrame containing:

```text
Rank
Resume ID
Match Score
Role
Seniority
Industry
Years Experience
Must-Have Coverage
Matched Skills
Missing Skills
```

---

## 💾 Saved Model

The final trained model is saved as:

```text
final_resume_job_ranker.pkl
```

The final feature configuration is saved as:

```text
final_features.pkl
```

They can be loaded later using:

```python
import joblib

model = joblib.load(
    "final_resume_job_ranker.pkl"
)

features = joblib.load(
    "final_features.pkl"
)
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Sentence Transformers
* LightGBM
* Hugging Face Transformers
* Qwen2.5
* Matplotlib
* Jupyter / Google Colab

---

## 📁 Project Structure

```text
resume-job-matching/
│
├── README.md
│
├── notebooks/
│   └── resume_job_matching.ipynb
│
├── models/
│   ├── final_resume_job_ranker.pkl
│   └── final_features.pkl
│
├── data/
│   └── README.md
│
├── src/
│   ├── retrieval.py
│   ├── features.py
│   ├── ranking.py
│   └── explainability.py
│
├── requirements.txt
│
└── .gitignore
```

---

## ▶️ Running the Project

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd resume-job-matching
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the notebook

```text
notebooks/resume_job_matching.ipynb
```

The notebook contains the complete workflow:

```text
Data Loading
→ Data Exploration
→ Text Processing
→ Embeddings
→ Retrieval
→ Feature Engineering
→ Ranking
→ Evaluation
→ Explainability
→ LLM Explanation
→ Final Demo
```

---

## 📌 Key Engineering Decisions

### Ranking instead of Classification

The project treats candidate selection as a ranking problem because recruiters need the **best candidates ordered by relevance**, not only binary relevance predictions.

### Retrieval + Ranking

The system separates:

```text
Candidate Retrieval
```

from:

```text
Candidate Ranking
```

This makes the architecture scalable and allows the ranking model to operate on a smaller candidate pool.

### Structured Features + Semantic Similarity

Semantic embeddings capture contextual similarity, while structured features capture explicit hiring requirements such as skills, seniority, experience, and industry.

### Explainability Layer

The system exposes the factors contributing to candidate-job matching instead of returning only a numerical score.

### LLM as an Explanation Layer

The LLM is intentionally separated from the core scoring pipeline so that ranking remains deterministic and measurable.

---

## 📊 Limitations

The current implementation is an experimental ranking system using a synthetic dataset and a sampled working dataset.

Some structured features, such as experience gaps, use heuristic definitions because explicit required-years fields are not available for every job.

The LLM explanation layer is optional and should not be treated as the source of truth for candidate matching.

---

## 🚀 Future Improvements

Potential extensions include:

* Larger-scale candidate retrieval
* More advanced embedding models
* Cross-encoder reranking
* Better skill normalization and ontology matching
* Hard/soft requirement handling
* Hyperparameter optimization
* FastAPI inference service
* Docker deployment
* MLflow experiment tracking
* Production database integration
* Resume PDF ingestion pipeline
* Monitoring and model drift detection
* Human-in-the-loop recruiter feedback
* Learning-to-rank models trained on real recruiter decisions

---

## 👨‍💻 Project Goal

This project demonstrates an end-to-end NLP and machine learning workflow for a realistic **Resume–Job Matching and Candidate Ranking** problem, combining semantic search, feature engineering, learned ranking, evaluation, explainability, and optional LLM-based natural-language generation.
