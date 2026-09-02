"""Attrition Risk Router."""
import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backend.models.schemas import (
    AttritionPredictRequest, AttritionPredictResponse, DashboardSummary
)
from app.backend.services.data_pipeline import get_data
from app.backend.services.ml_engine import (
    batch_predict_attrition, get_attrition_feature_importance, predict_attrition
)

router = APIRouter(prefix="/attrition", tags=["Attrition Risk"])


@router.post("/predict", response_model=AttritionPredictResponse)
def predict_single_attrition(req: AttritionPredictRequest):
    """Predict attrition risk for one employee."""
    try:
        result = predict_attrition(req.model_dump())
        messages = {
            "Low": "Employee is likely to stay. Keep monitoring engagement.",
            "Medium": "Moderate risk. Consider engagement initiatives.",
            "High": "High attrition risk. Recommend manager intervention.",
            "Critical": "CRITICAL — Immediate retention action required.",
        }
        result["message"] = messages.get(result["risk_tier"], "")
        return result
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet. Run setup first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk-summary")
def attrition_risk_summary():
    """Department-level attrition risk breakdown."""
    try:
        data = get_data()
        df = batch_predict_attrition(data["attrition"])

        dept_summary = (
            df.groupby("Department")
            .agg(
                total=("EmployeeNumber", "count"),
                high_risk=("RiskTier", lambda x: (x.isin(["High", "Critical"])).sum()),
                avg_risk=("AttritionProbability", "mean"),
            )
            .reset_index()
        )
        dept_summary["risk_pct"] = (
            dept_summary["high_risk"] / dept_summary["total"] * 100
        ).round(1)

        tier_counts = df["RiskTier"].value_counts().to_dict()

        return {
            "total_employees": len(df),
            "tier_breakdown": tier_counts,
            "department_summary": dept_summary.to_dict("records"),
            "overall_attrition_rate": round(
                df["Attrition_enc"].mean() * 100 if "Attrition_enc" in df.columns else 0, 1
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feature-importance")
def feature_importance():
    """Return attrition model feature importances."""
    try:
        fi = get_attrition_feature_importance()
        return fi.head(20).to_dict("records")
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/high-risk-employees")
def high_risk_employees(limit: int = 50):
    """List employees flagged as High or Critical risk."""
    try:
        data = get_data()
        df = batch_predict_attrition(data["attrition"])
        hr = df[df["RiskTier"].isin(["High", "Critical"])].copy()
        cols = [c for c in [
            "EmployeeNumber", "Age", "Department", "JobRole",
            "MonthlyIncome", "YearsAtCompany", "JobSatisfaction",
            "OverTime", "AttritionProbability", "RiskTier"
        ] if c in hr.columns]
        hr = hr[cols].sort_values("AttritionProbability", ascending=False)
        return hr.head(limit).to_dict("records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
