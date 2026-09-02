"""
Skill Gap & Succession Planning Engine
────────────────────────────────────────
- Maps employee job roles → O*NET codes
- Identifies skill gaps (essential + software) per role
- Generates upskilling recommendations
- Ranks succession candidates by performance + tenure + potential
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import SKILL_GAP_HIGH, SKILL_GAP_MEDIUM


# ═══════════════════════════════════════════════════════════════════════════════
#   ROLE → O*NET MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

# Manual mapping from IBM HR job roles → O*NET SOC codes (best-fit)
ROLE_ONET_MAP = {
    "Sales Executive": "41-4012.00",
    "Research Scientist": "19-1042.00",
    "Laboratory Technician": "19-4021.00",
    "Manufacturing Director": "11-3051.00",
    "Healthcare Representative": "41-9041.00",
    "Manager": "11-1021.00",
    "Sales Representative": "41-4012.00",
    "Research Director": "11-9121.01",
    "Human Resources": "13-1071.00",
    "Auditor": "13-2011.00",
    "Production Technician": "51-9199.00",
    "Marketing Executive": "11-2011.00",
    "Data Scientist": "15-2051.00",
    "Software Engineer": "15-1252.00",
    "Data Analyst": "15-1211.00",
    "ML Engineer": "15-2051.00",
    "Backend Engineer": "15-1252.00",
}

# Upskilling course recommendations per skill domain
COURSE_RECOMMENDATIONS = {
    "Machine Learning": ["Machine Learning Specialization (Coursera)", "Applied ML with Python (edX)"],
    "Data Analysis": ["Google Data Analytics Certificate", "Data Analysis with Python (IBM)"],
    "Cloud": ["AWS Solutions Architect Associate", "Google Cloud Professional Data Engineer"],
    "Leadership": ["Leadership Foundations (LinkedIn Learning)", "Strategic Leadership (Coursera)"],
    "Communication": ["Business Communication Skills (Coursera)", "Technical Writing (Google)"],
    "Programming": ["Python for Data Science (DataCamp)", "Advanced Python (Real Python)"],
    "Statistics": ["Statistics for Data Science (Coursera)", "Applied Statistics (edX)"],
    "Project Management": ["PMP Certification Prep", "Agile Project Management (Google)"],
    "Database": ["SQL for Data Analysis (Udacity)", "PostgreSQL Mastery (Udemy)"],
    "Security": ["Cybersecurity Fundamentals (IBM)", "CompTIA Security+"],
}


def get_onet_code(job_role: str) -> str | None:
    """Return best-match O*NET code for a job role."""
    # Direct match
    if job_role in ROLE_ONET_MAP:
        return ROLE_ONET_MAP[job_role]
    # Partial match
    for role, code in ROLE_ONET_MAP.items():
        if role.lower() in job_role.lower() or job_role.lower() in role.lower():
            return code
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#   ORG-WIDE SKILL GAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_org_skill_gaps(
    essential_skills: pd.DataFrame,
    software_skills: pd.DataFrame,
    occupations: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute organisation-wide skill gaps using O*NET importance scores.
    Returns a DataFrame of skills sorted by gap severity.
    """
    logger.info("Computing org-wide skill gaps …")

    # High-demand roles in the org
    target_roles = list(ROLE_ONET_MAP.values())

    # Filter essential skills for target roles
    es = essential_skills[
        essential_skills["O*NET-SOC Code"].isin(target_roles)
    ].copy() if "O*NET-SOC Code" in essential_skills.columns else essential_skills.copy()

    if es.empty:
        logger.warning("No matching essential skills found — using top skills")
        es = essential_skills.head(200).copy()

    # Aggregate by skill element
    gap_cols = [c for c in ["Element Name", "Data Value"] if c in es.columns]
    if len(gap_cols) < 2:
        logger.warning("Expected columns not found in essential_skills")
        return pd.DataFrame()

    skill_summary = (
        es.groupby("Element Name")["Data Value"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_importance", "count": "role_count"})
        .sort_values("avg_importance", ascending=False)
    )

    skill_summary["gap_severity"] = skill_summary["avg_importance"].apply(
        lambda x: "High" if x >= SKILL_GAP_HIGH
        else "Medium" if x >= SKILL_GAP_MEDIUM
        else "Low"
    )
    skill_summary["gap_score"] = (
        (skill_summary["avg_importance"] / skill_summary["avg_importance"].max()) * 100
    ).round(1)

    # Top skills from software list
    if "Element Name" in software_skills.columns:
        sw_hot = software_skills[
            software_skills.get("Hot Technology", pd.Series()) == "Y"
        ] if "Hot Technology" in software_skills.columns else software_skills.head(50)

        hot_skills = sw_hot["Element Name"].value_counts().head(10).reset_index()
        hot_skills.columns = ["Element Name", "freq"]
        hot_skills["avg_importance"] = 4.2
        hot_skills["gap_severity"] = "High"
        hot_skills["gap_score"] = 85.0
        hot_skills["role_count"] = hot_skills["freq"]
        skill_summary = pd.concat(
            [skill_summary, hot_skills[["Element Name", "avg_importance",
                                        "role_count", "gap_severity", "gap_score"]]],
            ignore_index=True
        ).drop_duplicates("Element Name")

    logger.success(f"Org skill gaps computed: {len(skill_summary)} skills")
    return skill_summary


def get_role_required_skills(
    job_role: str,
    essential_skills: pd.DataFrame,
    software_skills: pd.DataFrame,
    top_n: int = 15
) -> dict:
    """Return top required skills for a given job role."""
    onet_code = get_onet_code(job_role)
    result = {"job_role": job_role, "onet_code": onet_code,
              "essential_skills": [], "software_skills": []}

    if onet_code and "O*NET-SOC Code" in essential_skills.columns:
        es = essential_skills[
            essential_skills["O*NET-SOC Code"] == onet_code
        ]
        if not es.empty and "Element Name" in es.columns and "Data Value" in es.columns:
            top_es = (
                es.sort_values("Data Value", ascending=False)
                  .head(top_n)[["Element Name", "Data Value"]]
                  .rename(columns={"Data Value": "importance"})
                  .to_dict("records")
            )
            result["essential_skills"] = top_es

    if onet_code and "O*NET-SOC Code" in software_skills.columns:
        sw = software_skills[
            software_skills["O*NET-SOC Code"] == onet_code
        ]
        if not sw.empty and "Element Name" in sw.columns:
            result["software_skills"] = sw["Element Name"].unique().tolist()[:top_n]

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#   INDIVIDUAL UPSKILLING PLAN
# ═══════════════════════════════════════════════════════════════════════════════

def generate_upskilling_plan(
    employee: dict,
    essential_skills: pd.DataFrame,
    software_skills: pd.DataFrame,
    max_recommendations: int = 5
) -> dict:
    """Generate a personalised upskilling plan for one employee."""
    job_role = employee.get("JobRole", employee.get("Job Role", "Unknown"))
    emp_id = employee.get("EmployeeNumber", employee.get("EmployeeID", "N/A"))
    training_hours = float(employee.get("TrainingTimesLastYear",
                                        employee.get("Training Hours", 0)) or 0)
    perf_score = float(employee.get("PerformanceRating",
                                    employee.get("Performance Score", 3)) or 3)

    role_skills = get_role_required_skills(job_role, essential_skills, software_skills)
    essential = [s["Element Name"] for s in role_skills["essential_skills"][:10]]

    # Map skill names to course domains
    recommendations = []
    for skill in essential[:max_recommendations]:
        matched_domain = None
        for domain, courses in COURSE_RECOMMENDATIONS.items():
            if domain.lower() in skill.lower() or skill.lower() in domain.lower():
                matched_domain = domain
                break
        if not matched_domain:
            # Default generic recommendation
            matched_domain = list(COURSE_RECOMMENDATIONS.keys())[
                hash(skill) % len(COURSE_RECOMMENDATIONS)
            ]
        courses = COURSE_RECOMMENDATIONS[matched_domain]
        recommendations.append({
            "skill": skill,
            "domain": matched_domain,
            "recommended_courses": courses,
            "priority": "High" if perf_score < 3 else "Medium",
            "estimated_hours": 20 + int(hash(skill) % 40)
        })

    urgency = "High" if training_hours < 2 else "Medium" if training_hours < 5 else "Low"

    return {
        "employee_id": emp_id,
        "job_role": job_role,
        "onet_code": role_skills["onet_code"],
        "training_urgency": urgency,
        "current_training_times": training_hours,
        "skill_recommendations": recommendations,
        "software_to_learn": role_skills["software_skills"][:5]
    }


def batch_upskilling_plans(
    employees_df: pd.DataFrame,
    essential_skills: pd.DataFrame,
    software_skills: pd.DataFrame
) -> list[dict]:
    """Generate plans for all employees."""
    logger.info(f"Generating upskilling plans for {len(employees_df)} employees …")
    plans = []
    for _, row in employees_df.iterrows():
        plan = generate_upskilling_plan(
            row.to_dict(), essential_skills, software_skills
        )
        plans.append(plan)
    logger.success(f"Upskilling plans generated: {len(plans)}")
    return plans


# ═══════════════════════════════════════════════════════════════════════════════
#   SUCCESSION PLANNING
# ═══════════════════════════════════════════════════════════════════════════════

def compute_succession_scores(
    attrition_df: pd.DataFrame,
    performance_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Compute succession readiness scores for all employees.
    Score = weighted combo of performance, tenure, seniority, engagement.
    """
    logger.info("Computing succession readiness scores …")

    df = attrition_df.copy()

    # Normalise features (0–10 scale)
    def norm(series):
        mn, mx = series.min(), series.max()
        return (series - mn) / (mx - mn + 1e-9) * 10

    score = pd.Series(0.0, index=df.index)

    if "PerformanceRating" in df.columns:
        score += norm(df["PerformanceRating"]) * 3.0   # 30% weight

    if "TotalWorkingYears" in df.columns:
        score += norm(df["TotalWorkingYears"]) * 2.0   # 20%

    if "JobLevel" in df.columns:
        score += norm(df["JobLevel"]) * 2.0             # 20%

    if "YearsInCurrentRole" in df.columns:
        score += norm(df["YearsInCurrentRole"]) * 1.5  # 15%

    if "TrainingTimesLastYear" in df.columns:
        score += norm(df["TrainingTimesLastYear"]) * 1.5  # 15%

    df["SuccessionScore"] = score.round(2)
    df["ReadinessTier"] = pd.cut(
        df["SuccessionScore"],
        bins=[0, 3, 5, 7.5, 10],
        labels=["Not Ready", "Developing", "Ready (1-2yr)", "Ready Now"],
        include_lowest=True
    )

    result_cols = [c for c in [
        "EmployeeNumber", "Age", "Department", "JobRole", "JobLevel",
        "TotalWorkingYears", "PerformanceRating", "TrainingTimesLastYear",
        "SuccessionScore", "ReadinessTier", "Attrition"
    ] if c in df.columns]

    result = df[result_cols].sort_values("SuccessionScore", ascending=False)
    logger.success(f"Succession scores computed for {len(result)} employees")
    return result


def get_succession_pipeline(
    attrition_df: pd.DataFrame,
    department: str | None = None,
    top_n: int = 20
) -> list[dict]:
    """Return top succession candidates, optionally filtered by department."""
    scored = compute_succession_scores(attrition_df)

    if department and "Department" in scored.columns:
        scored = scored[scored["Department"] == department]

    # Only those not leaving
    if "Attrition" in scored.columns:
        scored = scored[scored["Attrition"] != "Yes"]

    return scored.head(top_n).to_dict("records")
