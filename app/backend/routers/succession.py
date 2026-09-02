"""Succession Planning Router."""
import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException
from typing import Optional

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backend.services.data_pipeline import get_data
from app.backend.services.skill_engine import compute_succession_scores, get_succession_pipeline

router = APIRouter(prefix="/succession", tags=["Succession Planning"])


@router.get("/candidates")
def succession_candidates(department: Optional[str] = None, top_n: int = 20):
    """Return top succession candidates, optionally filtered by department."""
    try:
        data = get_data()
        candidates = get_succession_pipeline(
            data["attrition"], department=department, top_n=top_n
        )
        return {"total": len(candidates), "candidates": candidates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/readiness-matrix")
def readiness_matrix():
    """Succession readiness breakdown by department and tier."""
    try:
        import pandas as pd
        data = get_data()
        scored = compute_succession_scores(data["attrition"])

        if "Department" not in scored.columns or "ReadinessTier" not in scored.columns:
            raise HTTPException(status_code=404, detail="Required columns missing.")

        matrix = (
            scored.groupby(["Department", "ReadinessTier"], observed=True)
            .size()
            .reset_index(name="count")
        )
        return matrix.to_dict("records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/high-potential")
def high_potential_employees(top_n: int = 30):
    """Employees with top succession scores who are NOT at attrition risk."""
    try:
        data = get_data()
        from app.backend.services.ml_engine import batch_predict_attrition
        df_attr = batch_predict_attrition(data["attrition"])
        scored = compute_succession_scores(df_attr)

        # High succession score + Low/Medium attrition risk
        if "AttritionProbability" in scored.columns:
            high_pot = scored[scored["AttritionProbability"] < 0.4]
        else:
            high_pot = scored

        cols = [c for c in [
            "EmployeeNumber", "Department", "JobRole", "JobLevel",
            "TotalWorkingYears", "PerformanceRating", "SuccessionScore",
            "ReadinessTier"
        ] if c in high_pot.columns]

        return high_pot[cols].head(top_n).fillna("").to_dict("records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
