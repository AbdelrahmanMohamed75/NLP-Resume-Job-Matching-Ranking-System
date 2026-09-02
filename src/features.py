def normalize_skills(skills):
    """
    Convert skills into a normalized lowercase set.
    """
    return set(
        str(skill).strip().lower()
        for skill in skills
    )


def skill_match_score(
    resume_skills,
    must_have_skills,
    nice_to_have_skills
):
    """
    Calculate skill matching scores between a resume and a job.
    """

    resume = normalize_skills(resume_skills)
    must_have = normalize_skills(must_have_skills)
    nice_to_have = normalize_skills(nice_to_have_skills)

    if len(must_have) > 0:
        matched_must = resume.intersection(must_have)
        must_score = len(matched_must) / len(must_have)
    else:
        matched_must = set()
        must_score = 0.0

    if len(nice_to_have) > 0:
        matched_nice = resume.intersection(nice_to_have)
        nice_score = len(matched_nice) / len(nice_to_have)
    else:
        matched_nice = set()
        nice_score = 0.0

    final_score = (
        0.8 * must_score +
        0.2 * nice_score
    )

    return {
        "skill_score": final_score,
        "must_score": must_score,
        "nice_score": nice_score,
        "matched_must": list(matched_must),
        "matched_nice": list(matched_nice),
        "missing_must": list(
            must_have - resume
        )
    }


def seniority_match_score(
    candidate_seniority,
    job_seniority
):
    """
    Compare candidate and job seniority.
    """

    candidate = str(
        candidate_seniority
    ).strip().lower()

    job = str(
        job_seniority
    ).strip().lower()

    levels = {
        "junior": 1,
        "mid": 2,
        "senior": 3
    }

    if candidate not in levels or job not in levels:
        return 0.0

    difference = abs(
        levels[candidate] -
        levels[job]
    )

    if difference == 0:
        return 1.0

    elif difference == 1:
        return 0.5

    return 0.0


def industry_match_score(
    candidate_industry,
    job_industry
):
    """
    Compare candidate and job industry.
    """

    candidate = str(
        candidate_industry
    ).strip().lower()

    job = str(
        job_industry
    ).strip().lower()

    return 1.0 if candidate == job else 0.0


def experience_match_score(
    candidate_years,
    job_seniority
):
    """
    Calculate experience compatibility
    based on job seniority.
    """

    candidate_years = float(
        candidate_years
    )

    job_seniority = str(
        job_seniority
    ).strip().lower()

    if job_seniority == "junior":

        if candidate_years <= 2:
            return 1.0

        elif candidate_years <= 5:
            return 0.5

        return 0.0

    elif job_seniority == "mid":

        if 3 <= candidate_years <= 6:
            return 1.0

        elif (
            2 <= candidate_years < 3
            or
            7 <= candidate_years <= 9
        ):
            return 0.5

        return 0.0

    elif job_seniority == "senior":

        if candidate_years >= 7:
            return 1.0

        elif candidate_years >= 5:
            return 0.5

        return 0.0

    return 0.0


def missing_must_have_count(
    resume_skills,
    must_have_skills
):
    """
    Count missing must-have skills.
    """

    resume = normalize_skills(
        resume_skills
    )

    must_have = normalize_skills(
        must_have_skills
    )

    return len(
        must_have - resume
    )


def calculate_experience_gap(
    candidate_years,
    job_seniority
):
    """
    Calculate the difference between
    candidate experience and expected experience.
    """

    candidate_years = float(
        candidate_years
    )

    job_seniority = str(
        job_seniority
    ).strip().lower()

    expected_years = {
        "junior": 1,
        "mid": 4,
        "senior": 7
    }

    if job_seniority not in expected_years:
        return 0.0

    return (
        candidate_years -
        expected_years[job_seniority]
    )


def calculate_must_have_coverage(
    resume_skills,
    must_have_skills
):
    """
    Calculate percentage of must-have
    skills covered by the candidate.
    """

    resume = normalize_skills(
        resume_skills
    )

    must_have = normalize_skills(
        must_have_skills
    )

    if len(must_have) == 0:
        return 0.0

    matched = resume.intersection(
        must_have
    )

    return len(matched) / len(must_have)


def calculate_nice_to_have_coverage(
    resume_skills,
    nice_to_have_skills
):
    """
    Calculate percentage of nice-to-have
    skills covered by the candidate.
    """

    resume = normalize_skills(
        resume_skills
    )

    nice_to_have = normalize_skills(
        nice_to_have_skills
    )

    if len(nice_to_have) == 0:
        return 0.0

    matched = resume.intersection(
        nice_to_have
    )

    return len(matched) / len(nice_to_have)


def calculate_seniority_gap(
    candidate_seniority,
    job_seniority
):
    """
    Calculate seniority difference.
    """

    levels = {
        "junior": 1,
        "mid": 2,
        "senior": 3
    }

    candidate = str(
        candidate_seniority
    ).strip().lower()

    job = str(
        job_seniority
    ).strip().lower()

    if candidate not in levels or job not in levels:
        return 0

    return (
        levels[candidate] -
        levels[job]
    )


def exact_must_have_match_count(
    resume_skills,
    must_have_skills
):
    """
    Count exact must-have skill matches.
    """

    resume = normalize_skills(
        resume_skills
    )

    must_have = normalize_skills(
        must_have_skills
    )

    return len(
        resume.intersection(
            must_have
        )
    )


def calculate_skill_count(
    resume_skills
):
    """
    Count unique candidate skills.
    """

    return len(
        normalize_skills(
            resume_skills
        )
    )


FINAL_FEATURES = [
    "semantic_score",
    "skill_score",
    "experience_score",
    "experience_gap",
    "seniority_score",
    "seniority_gap",
    "industry_score",
    "missing_must_have_count",
    "must_have_coverage",
    "nice_to_have_coverage",
    "exact_must_have_match_count",
    "skill_count"
]
