"""
Pydantic Schemas for FastAPI request/response models.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Attrition ─────────────────────────────────────────────────────────────────

class AttritionPredictRequest(BaseModel):
    Age: int = 35
    DailyRate: float = 800
    DistanceFromHome: int = 5
    Education: int = 3
    EnvironmentSatisfaction: int = 3
    HourlyRate: float = 60
    JobInvolvement: int = 3
    JobLevel: int = 2
    JobSatisfaction: int = 3
    MonthlyIncome: float = 5000
    MonthlyRate: float = 15000
    NumCompaniesWorked: int = 2
    OverTime_enc: int = 0
    PercentSalaryHike: float = 12
    PerformanceRating: int = 3
    RelationshipSatisfaction: int = 3
    StockOptionLevel: int = 1
    TotalWorkingYears: int = 8
    TrainingTimesLastYear: int = 3
    WorkLifeBalance: int = 3
    YearsAtCompany: int = 5
    YearsInCurrentRole: int = 3
    YearsSinceLastPromotion: int = 1
    YearsWithCurrManager: int = 3
    BusinessTravel_enc: int = 1
    Department_enc: int = 1
    Gender_enc: int = 1
    MaritalStatus_enc: int = 0


class AttritionPredictResponse(BaseModel):
    attrition_probability: float
    risk_tier: str
    will_leave: bool
    message: str


# ── Performance ───────────────────────────────────────────────────────────────

class PerformancePredictRequest(BaseModel):
    performance_score: float = Field(75, alias="Performance Score")
    kpi_score: float = Field(80, alias="KPI Score")
    attendance_pct: float = Field(90, alias="Attendance (%)")
    peer_rating: float = Field(4.0, alias="Peer Rating")
    task_completion_pct: float = Field(85, alias="Task Completion (%)")
    work_hours_logged: float = Field(44, alias="Work Hours Logged")
    training_hours: float = Field(20, alias="Training Hours")

    class Config:
        populate_by_name = True


class PerformancePredictResponse(BaseModel):
    predicted_performance_score: float
    performance_tier: str
    promotion_eligible: bool


# ── Skills ────────────────────────────────────────────────────────────────────

class SkillGapRow(BaseModel):
    element_name: str
    avg_importance: float
    gap_severity: str
    gap_score: float


class UpskillingRequest(BaseModel):
    employee_id: Any
    job_role: str
    training_times_last_year: Optional[float] = 2
    performance_rating: Optional[float] = 3


# ── Policy Q&A ────────────────────────────────────────────────────────────────

class PolicyQueryRequest(BaseModel):
    question: str
    top_k: int = 5


class PolicyQueryResponse(BaseModel):
    question: str
    answer: str
    num_sources: int
    sources: list[dict]


# ── Resume ────────────────────────────────────────────────────────────────────

class ResumeMatchRequest(BaseModel):
    resume_text: str
    jd_name: str = "ml_engineer.txt"


class ResumeMatchResponse(BaseModel):
    jd_file: str
    match_score: float
    match_percentage: str
    recommendation: str
    matched_keywords: list[str]
    shortlist: bool


# ── Dashboard Summary ─────────────────────────────────────────────────────────

class DashboardSummary(BaseModel):
    total_employees: int
    high_risk_employees: int
    critical_risk_employees: int
    avg_engagement_score: float
    avg_performance_score: float
    attrition_rate: float
    top_risk_department: str
    total_skill_gaps_high: int


# ── Generic ───────────────────────────────────────────────────────────────────

class SuccessResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
