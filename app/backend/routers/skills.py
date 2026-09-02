"""Skills & Upskilling Router."""
import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backend.models.schemas import UpskillingRequest
from app.backend.services.data_pipeline import get_data
from app.backend.services.skill_engine import (
    compute_org_skill_gaps, generate_upskilling_plan, get_role_required_skills
)

router = APIRouter(prefix="/skills", tags=["Skill Gap & Upskilling"])


@router.get("/org-gap-analysis")
def org_skill_gap_analysis(top_n: int = 25):
    """Organisation-wide skill gap analysis using O*NET data."""
    try:
        data = get_data()
        gaps = compute_org_skill_gaps(
            data["essential_skills"],
            data["software_skills"],
            data["occupations"]
        )
        return gaps.head(top_n).fillna("").to_dict("records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/role-skills/{job_role}")
def role_required_skills(job_role: str, top_n: int = 15):
    """Get top required skills for a specific job role."""
    try:
        data = get_data()
        result = get_role_required_skills(
            job_role,
            data["essential_skills"],
            data["software_skills"],
            top_n=top_n
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upskilling-plan")
def get_upskilling_plan(req: UpskillingRequest):
    """Generate personalised upskilling plan for one employee."""
    try:
        data = get_data()
        employee_dict = {
            "EmployeeNumber": req.employee_id,
            "JobRole": req.job_role,
            "TrainingTimesLastYear": req.training_times_last_year,
            "PerformanceRating": req.performance_rating,
        }
        plan = generate_upskilling_plan(
            employee_dict,
            data["essential_skills"],
            data["software_skills"]
        )
        return plan
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hot-technologies")
def hot_technologies(top_n: int = 20):
    """Return in-demand/hot technology skills from O*NET."""
    try:
        data = get_data()
        sw = data["software_skills"]
        if "Hot Technology" not in sw.columns:
            return {"message": "Hot Technology column not available", "data": []}
        hot = sw[sw["Hot Technology"] == "Y"]
        if "Element Name" in hot.columns:
            top = hot["Element Name"].value_counts().head(top_n).reset_index()
            top.columns = ["technology", "frequency"]
            return top.to_dict("records")
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
