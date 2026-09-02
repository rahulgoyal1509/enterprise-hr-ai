"""Policy Q&A and Resume Screening Router."""
import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backend.models.schemas import (
    PolicyQueryRequest, PolicyQueryResponse,
    ResumeMatchRequest, ResumeMatchResponse
)
from app.backend.services.nlp_engine import (
    analyze_sentiment, query_policy, rank_all_resumes, match_resume_to_jd
)
from config.settings import JD_DIR

router = APIRouter(prefix="/ai", tags=["AI / NLP Engine"])


@router.post("/policy/query", response_model=PolicyQueryResponse)
def policy_query(req: PolicyQueryRequest):
    """Query HR policy documents using RAG."""
    try:
        result = query_policy(req.question, top_k=req.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resume/match", response_model=ResumeMatchResponse)
def resume_match(req: ResumeMatchRequest):
    """Match a resume text against a job description."""
    try:
        result = match_resume_to_jd(req.resume_text, req.jd_name)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/resume/rank/{jd_name}")
def rank_resumes(jd_name: str):
    """Rank all candidate resumes against a job description."""
    try:
        results = rank_all_resumes(jd_name)
        return {"jd": jd_name, "ranked_candidates": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sentiment")
def sentiment_analysis(text: str):
    """Analyse sentiment of any text (manager feedback, reviews)."""
    try:
        return analyze_sentiment(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/job-descriptions")
def list_job_descriptions():
    """List available job description files."""
    try:
        jds = [f.name for f in JD_DIR.glob("*.txt")]
        return {"job_descriptions": jds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
