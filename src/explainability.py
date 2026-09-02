from .features import (
    skill_match_score,
    calculate_must_have_coverage,
    calculate_nice_to_have_coverage,
)


def generate_candidate_explanation(
    job,
    resume,
    ranking_score
):
    """
    Generate structured, human-readable
    information explaining a candidate match.
    """

    skill_result = skill_match_score(
        resume["skills"],
        job["must_have_skills"],
        job["nice_to_have_skills"]
    )

    explanation = {
        "resume_id": resume["resume_id"],
        "job_id": job["job_id"],
        "candidate_role": resume["role"],
        "candidate_seniority": resume["seniority"],
        "candidate_industry": resume["industry"],
        "years_experience": int(
            resume["years_experience"]
        ),

        "ranking_score": round(
            float(ranking_score),
            4
        ),

        "must_have_coverage": round(
            calculate_must_have_coverage(
                resume["skills"],
                job["must_have_skills"]
            ),
            4
        ),

        "nice_to_have_coverage": round(
            calculate_nice_to_have_coverage(
                resume["skills"],
                job["nice_to_have_skills"]
            ),
            4
        ),

        "matched_must_have": skill_result[
            "matched_must"
        ],

        "matched_nice_to_have": skill_result[
            "matched_nice"
        ],

        "missing_must_have": skill_result[
            "missing_must"
        ]
    }

    return explanation
