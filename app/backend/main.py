"""
FastAPI Main Application
─────────────────────────
Enterprise HR AI Workforce Intelligence Platform
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import API_PREFIX, APP_TITLE, APP_VERSION, CORS_ORIGINS
from app.backend.routers import attrition, performance, skills, succession, policy_qa


# ── Lifespan: run pipeline + train models on startup ─────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Enterprise HR AI Platform …")
    try:
        from app.backend.services.data_pipeline import get_data
        from app.backend.services.ml_engine import train_all_models
        from app.backend.services.nlp_engine import initialize_nlp
        from config.settings import ATTRITION_MODEL_PATH

        data = get_data()

        if not ATTRITION_MODEL_PATH.exists():
            logger.info("Models not found — training now …")
            train_all_models(data)
        else:
            logger.info("Pre-trained models found ✔")

        initialize_nlp()
        logger.success("Platform ready ✔")
    except Exception as e:
        logger.error(f"Startup error: {e}")
    yield
    logger.info("Shutting down …")


# ── App instance ──────────────────────────────────────────────────────────────
app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=(
        "AI-powered HR Analytics platform with Attrition Risk ML, "
        "Performance Prediction, Skill Gap Analysis, Succession Planning, "
        "RAG Policy Q&A and Resume Screening."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(attrition.router,  prefix=API_PREFIX)
app.include_router(performance.router, prefix=API_PREFIX)
app.include_router(skills.router,     prefix=API_PREFIX)
app.include_router(succession.router, prefix=API_PREFIX)
app.include_router(policy_qa.router,  prefix=API_PREFIX)


# ── Root & Health ─────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "platform": APP_TITLE,
        "version": APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "attrition":   f"{API_PREFIX}/attrition",
            "performance": f"{API_PREFIX}/performance",
            "skills":      f"{API_PREFIX}/skills",
            "succession":  f"{API_PREFIX}/succession",
            "ai":          f"{API_PREFIX}/ai",
        },
    }


@app.get(f"{API_PREFIX}/health", tags=["Health"])
def health_check():
    from config.settings import ATTRITION_MODEL_PATH, DATABASE_PATH
    return {
        "status": "healthy",
        "database": "connected" if DATABASE_PATH.exists() else "not initialised",
        "attrition_model": "loaded" if ATTRITION_MODEL_PATH.exists() else "not trained",
    }


@app.get(f"{API_PREFIX}/analytics/dashboard-summary", tags=["Analytics"])
def dashboard_summary():
    """Aggregated KPIs for the executive dashboard."""
    try:
        from app.backend.services.data_pipeline import get_data
        from app.backend.services.ml_engine import batch_predict_attrition
        from app.backend.services.skill_engine import compute_org_skill_gaps

        data = get_data()
        df = batch_predict_attrition(data["attrition"])

        high_risk = int((df["RiskTier"].isin(["High", "Critical"])).sum())
        critical   = int((df["RiskTier"] == "Critical").sum())

        eng_cols = [c for c in ["Engagement Score", "Satisfaction Score"] if c in data["hr_performance"].columns]
        avg_eng  = float(data["hr_performance"][eng_cols[0]].mean()) if eng_cols else 72.0

        perf_col = next((c for c in ["Performance Score", "KPI Score"]
                         if c in data["performance_kpi"].columns), None)
        avg_perf = float(data["performance_kpi"][perf_col].mean()) if perf_col else 75.0

        dept_risk = df.groupby("Department")["AttritionProbability"].mean()
        top_risk_dept = dept_risk.idxmax() if not dept_risk.empty else "N/A"

        gaps = compute_org_skill_gaps(data["essential_skills"], data["software_skills"], data["occupations"])
        high_gaps = int((gaps["gap_severity"] == "High").sum()) if not gaps.empty else 0

        attr_rate = round(float(df["Attrition_enc"].mean() * 100) if "Attrition_enc" in df.columns else 16.1, 1)

        return {
            "total_employees":       len(df),
            "high_risk_employees":   high_risk,
            "critical_risk_employees": critical,
            "avg_engagement_score":  round(avg_eng, 1),
            "avg_performance_score": round(avg_perf, 1),
            "attrition_rate":        attr_rate,
            "top_risk_department":   top_risk_dept,
            "total_skill_gaps_high": high_gaps,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.backend.main:app", host="0.0.0.0", port=8000, reload=True)
