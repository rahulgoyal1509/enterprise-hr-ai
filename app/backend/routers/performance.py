"""Performance Analytics Router."""
import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backend.models.schemas import PerformancePredictRequest, PerformancePredictResponse
from app.backend.services.data_pipeline import get_data
from app.backend.services.ml_engine import predict_performance

router = APIRouter(prefix="/performance", tags=["Performance Analytics"])


@router.post("/predict", response_model=PerformancePredictResponse)
def predict_performance_score(req: PerformancePredictRequest):
    """Predict performance score for one employee."""
    try:
        payload = {
            "Performance Score": req.performance_score,
            "KPI Score": req.kpi_score,
            "Attendance (%)": req.attendance_pct,
            "Peer Rating": req.peer_rating,
            "Task Completion (%)": req.task_completion_pct,
            "Work Hours Logged": req.work_hours_logged,
            "Training Hours": req.training_hours,
        }
        return predict_performance(payload)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/leaderboard")
def performance_leaderboard(department: str = None, top_n: int = 20):
    """Top performers ranked by KPI + performance score."""
    try:
        data = get_data()
        df = data["performance_kpi"].copy()
        if department and "Department" in df.columns:
            df = df[df["Department"] == department]

        score_col = next(
            (c for c in ["Performance Score", "KPI Score"] if c in df.columns), None
        )
        if score_col:
            df = df.sort_values(score_col, ascending=False)

        cols = [c for c in [
            "Employee ID", "Name", "Department", "Job Role",
            "Performance Score", "KPI Score", "Attendance (%)",
            "Peer Rating", "Task Completion (%)", "Promotion Eligibility"
        ] if c in df.columns]
        return df[cols].head(top_n).to_dict("records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/department-summary")
def department_performance_summary():
    """Average performance metrics by department."""
    try:
        data = get_data()
        df = data["performance_kpi"]
        if "Department" not in df.columns:
            raise HTTPException(status_code=404, detail="Department column not found.")

        numeric_cols = [c for c in [
            "Performance Score", "KPI Score", "Attendance (%)",
            "Peer Rating", "Task Completion (%)", "Training Hours"
        ] if c in df.columns]

        summary = df.groupby("Department")[numeric_cols].mean().round(2).reset_index()
        return summary.to_dict("records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training-impact")
def training_impact_analysis():
    """Correlation between training hours and performance."""
    try:
        data = get_data()
        df = data["performance_kpi"]
        cols = [c for c in ["Training Hours", "Performance Score", "KPI Score",
                             "Promotion Eligibility"] if c in df.columns]
        df_clean = df[cols].dropna()

        bins = [0, 10, 20, 30, 50, 200]
        labels = ["0-10h", "10-20h", "20-30h", "30-50h", "50h+"]
        df_clean = df_clean.copy()
        df_clean["training_band"] = pd.cut(
            df_clean["Training Hours"], bins=bins, labels=labels
        )
        agg = df_clean.groupby("training_band", observed=True).agg(
            avg_performance=("Performance Score", "mean"),
            avg_kpi=("KPI Score", "mean"),
            employee_count=("Training Hours", "count"),
        ).reset_index()
        return agg.to_dict("records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# need pandas for training impact
import pandas as pd  # noqa: E402
