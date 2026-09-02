"""Central configuration for Enterprise HR AI Platform."""
from pathlib import Path

# ── Base paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
POLICIES_DIR = DATA_DIR / "hr_policies"
RESUMES_DIR = DATA_DIR / "resumes"
JD_DIR = DATA_DIR / "job_descriptions"
MODELS_DIR = BASE_DIR / "models"

# Ensure all required directories exist on disk
for _dir in [DATA_DIR, RAW_DIR, PROCESSED_DIR, POLICIES_DIR, RESUMES_DIR, JD_DIR, MODELS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL = f"sqlite:///{BASE_DIR}/data/hr_platform.db"
DATABASE_PATH = BASE_DIR / "data" / "hr_platform.db"

# ── Raw data files ────────────────────────────────────────────────────────────
ATTRITION_CSV = RAW_DIR / "employee_attrition.csv"
HR_PERFORMANCE_CSV = RAW_DIR / "Cleaned_HR_Data_Analysis.csv"
OCCUPATION_CSV = RAW_DIR / "occupation_data.csv"
ESSENTIAL_SKILLS_CSV = RAW_DIR / "essential_skills.csv"
SOFTWARE_SKILLS_CSV = RAW_DIR / "software_skills.csv"
PERFORMANCE_PRO_CSV = RAW_DIR / "employee_performance_pro.csv"
PERFORMANCE_DATASET_CSV = RAW_DIR / "Employee_Performance_Dataset.csv"

# ── Model files ───────────────────────────────────────────────────────────────
ATTRITION_MODEL_PATH = MODELS_DIR / "attrition_model.pkl"
PERFORMANCE_MODEL_PATH = MODELS_DIR / "performance_model.pkl"
ENGAGEMENT_MODEL_PATH = MODELS_DIR / "engagement_cluster_model.pkl"
LABEL_ENCODERS_PATH = MODELS_DIR / "label_encoders.pkl"
SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"

# ── Vector store ──────────────────────────────────────────────────────────────
FAISS_INDEX_PATH = MODELS_DIR / "policy_faiss.index"
FAISS_CHUNKS_PATH = MODELS_DIR / "policy_chunks.pkl"

# ── ML Hyperparameters ────────────────────────────────────────────────────────
ATTRITION_FEATURES = [
    "Age", "DailyRate", "DistanceFromHome", "Education",
    "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement",
    "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate",
    "NumCompaniesWorked", "OverTime_enc", "PercentSalaryHike",
    "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel",
    "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance",
    "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
    "YearsWithCurrManager", "BusinessTravel_enc", "Department_enc",
    "Gender_enc", "MaritalStatus_enc"
]

ATTRITION_RISK_TIERS = {
    (0.0, 0.25): "Low",
    (0.25, 0.50): "Medium",
    (0.50, 0.75): "High",
    (0.75, 1.01): "Critical"
}

PERFORMANCE_FEATURES = [
    "Performance Score", "KPI Score", "Attendance (%)",
    "Peer Rating", "Task Completion (%)", "Work Hours Logged",
    "Training Hours"
]

# ── Skill gap thresholds ──────────────────────────────────────────────────────
SKILL_GAP_HIGH = 4.0
SKILL_GAP_MEDIUM = 3.0

# ── App settings ──────────────────────────────────────────────────────────────
APP_TITLE = "AI Workforce Intelligence Platform"
APP_VERSION = "1.0.0"
API_PREFIX = "/api/v1"
CORS_ORIGINS = ["http://localhost:8501", "http://127.0.0.1:8501"]
