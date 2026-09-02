import pandas as pd

from .features import (
    skill_match_score,
    calculate_must_have_coverage,
    calculate_nice_to_have_coverage,
    missing_must_have_count,
    calculate_experience_gap,
    calculate_seniority_gap,
    exact_must_have_match_count,
    calculate_skill_count,
    seniority_match_score,
    industry_match_score,
    experience_match_score,
)


def build_candidate_features(job, resume, semantic_score):
    """
    Build the final feature vector for one
    job-resume pair.
    """

    skill_result = skill_match_score(
        resume["skills"],
        job["must_have_skills"],
        job["nice_to_have_skills"]
    )

    features = {
        "semantic_score": semantic_score,

        "skill_score": skill_result["skill_score"],

        "experience_score": experience_match_score(
            resume["years_experience"],
            job["seniority"]
        ),

        "experience_gap": calculate_experience_gap(
            resume["years_experience"],
            job["seniority"]
        ),

        "seniority_score": seniority_match_score(
            resume["seniority"],
            job["seniority"]
        ),

        "seniority_gap": calculate_seniority_gap(
            resume["seniority"],
            job["seniority"]
        ),

        "industry_score": industry_match_score(
            resume["industry"],
            job["industry"]
        ),

        "missing_must_have_count": missing_must_have_count(
            resume["skills"],
            job["must_have_skills"]
        ),

        "must_have_coverage": calculate_must_have_coverage(
            resume["skills"],
            job["must_have_skills"]
        ),

        "nice_to_have_coverage": calculate_nice_to_have_coverage(
            resume["skills"],
            job["nice_to_have_skills"]
        ),

        "exact_must_have_match_count": exact_must_have_match_count(
            resume["skills"],
            job["must_have_skills"]
        ),

        "skill_count": calculate_skill_count(
            resume["skills"]
        ),
    }

    return features

def score_candidates(
    model,
    features,
    job,
    resumes,
    candidate_indices,
    semantic_scores
):
    """
    Score and rank retrieved candidates.
    """

    results = []

    for resume_idx, semantic_score in zip(
        candidate_indices,
        semantic_scores
    ):

        resume = resumes.iloc[resume_idx]

        feature_values = build_candidate_features(
            job=job,
            resume=resume,
            semantic_score=semantic_score
        )

        X = pd.DataFrame(
            [[feature_values[feature] for feature in features]],
            columns=features
        )

        score = model.predict_proba(X)[0, 1]

        results.append({
            "resume_idx": resume_idx,
            "resume_id": resume["resume_id"],
            "ranking_score": float(score)
        })

    results = sorted(
        results,
        key=lambda x: x["ranking_score"],
        reverse=True
    )

    for rank, result in enumerate(results, start=1):
        result["rank"] = rank

    return results
