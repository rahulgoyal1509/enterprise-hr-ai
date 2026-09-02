"""
Data Pipeline Service
─────────────────────
Loads, validates, cleans, and persists all HR data sources into SQLite.
Produces normalized tables: employees, performance, skills, occupations.
"""

import sys
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text

# ── path resolution so this module can be run directly ───────────────────────
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    ATTRITION_CSV, HR_PERFORMANCE_CSV, OCCUPATION_CSV,
    ESSENTIAL_SKILLS_CSV, SOFTWARE_SKILLS_CSV,
    PERFORMANCE_PRO_CSV, PERFORMANCE_DATASET_CSV,
    PROCESSED_DIR, DATABASE_URL, LABEL_ENCODERS_PATH
)

# ── logger setup ──────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")


# ═══════════════════════════════════════════════════════════════════════════════
#   LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_attrition() -> pd.DataFrame:
    """Load IBM HR Attrition dataset."""
    logger.info("Loading employee_attrition.csv …")
    df = pd.read_csv(ATTRITION_CSV)

    # Drop constant / redundant columns
    df.drop(columns=["EmployeeCount", "Over18", "StandardHours"], errors="ignore", inplace=True)

    # Encode target
    df["Attrition_enc"] = (df["Attrition"] == "Yes").astype(int)

    # Encode categoricals
    cat_cols = {
        "OverTime": {"Yes": 1, "No": 0},
        "Gender": {"Male": 1, "Female": 0},
    }
    for col, mapping in cat_cols.items():
        if col in df.columns:
            df[f"{col}_enc"] = df[col].map(mapping).fillna(0).astype(int)

    ordinal_cols = ["BusinessTravel", "Department", "EducationField",
                    "JobRole", "MaritalStatus"]
    encoders = {}
    for col in ordinal_cols:
        if col in df.columns:
            uniq = df[col].dropna().unique()
            enc_map = {v: i for i, v in enumerate(sorted(uniq))}
            df[f"{col}_enc"] = df[col].map(enc_map).fillna(0).astype(int)
            encoders[col] = enc_map

    # Age bands
    df["AgeBand"] = pd.cut(
        df["Age"],
        bins=[18, 25, 35, 45, 60, 100],
        labels=["18-25", "26-35", "36-45", "46-60", "60+"]
    )

    # Income band
    df["IncomeBand"] = pd.cut(
        df["MonthlyIncome"],
        bins=5,
        labels=["Very Low", "Low", "Medium", "High", "Very High"]
    )

    joblib.dump(encoders, LABEL_ENCODERS_PATH)
    logger.success(f"Attrition data loaded: {len(df)} rows, {len(df.columns)} columns")
    return df


def load_hr_performance() -> pd.DataFrame:
    """Load performance/engagement/training data."""
    logger.info("Loading Cleaned_HR_Data_Analysis.csv …")
    df = pd.read_csv(HR_PERFORMANCE_CSV)

    # Normalise column names
    df.columns = [c.strip() for c in df.columns]

    # Parse dates where possible
    for col in ["StartDate", "Survey Date", "Training Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

    # Numeric coercions
    for col in ["Engagement Score", "Satisfaction Score", "Work-Life Balance Score",
                "Training Duration(Days)", "Training Cost"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Encode performance score
    perf_map = {
        "Exceeds": 5, "Fully Meets": 4, "Needs Improvement": 2,
        "PIP": 1, "Does not meet": 1
    }
    if "Performance Score" in df.columns:
        df["PerformanceScore_enc"] = df["Performance Score"].map(perf_map).fillna(3)

    logger.success(f"HR Performance data loaded: {len(df)} rows")
    return df


def load_occupations() -> pd.DataFrame:
    """Load O*NET occupation master data."""
    logger.info("Loading occupation_data.csv …")
    df = pd.read_csv(OCCUPATION_CSV)
    df.columns = [c.strip() for c in df.columns]
    logger.success(f"Occupations loaded: {len(df)} rows")
    return df


def load_essential_skills() -> pd.DataFrame:
    """Load O*NET essential skills with importance scores."""
    logger.info("Loading essential_skills.csv …")
    df = pd.read_csv(ESSENTIAL_SKILLS_CSV)
    df.columns = [c.strip() for c in df.columns]

    # Keep only Importance scale rows for clarity
    if "Scale ID" in df.columns:
        df = df[df["Scale ID"] == "IM"].copy()

    # Clean up
    for col in ["Data Value", "N", "Standard Error"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.success(f"Essential skills loaded: {len(df)} rows")
    return df


def load_software_skills() -> pd.DataFrame:
    """Load O*NET software/technology skills."""
    logger.info("Loading software_skills.csv …")
    df = pd.read_csv(SOFTWARE_SKILLS_CSV)
    df.columns = [c.strip() for c in df.columns]
    logger.success(f"Software skills loaded: {len(df)} rows")
    return df


def load_performance_pro() -> pd.DataFrame:
    """Load extended employee performance dataset."""
    logger.info("Loading employee_performance_pro.csv …")
    df = pd.read_csv(PERFORMANCE_PRO_CSV)
    df.columns = [c.strip() for c in df.columns]

    if "JoiningDate" in df.columns:
        df["JoiningDate"] = pd.to_datetime(df["JoiningDate"], errors="coerce")

    # Derive tenure in years
    if "JoiningDate" in df.columns:
        df["TenureYears"] = (pd.Timestamp.now() - df["JoiningDate"]).dt.days / 365.25

    logger.success(f"Performance Pro loaded: {len(df)} rows")
    return df


def load_performance_dataset() -> pd.DataFrame:
    """Load KPI / attendance / peer-rating dataset."""
    logger.info("Loading Employee_Performance_Dataset.csv …")
    df = pd.read_csv(PERFORMANCE_DATASET_CSV)
    df.columns = [c.strip() for c in df.columns]
    logger.success(f"Performance Dataset loaded: {len(df)} rows")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#   PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════════

def save_to_db(dfs: dict[str, pd.DataFrame], engine) -> None:
    """Write all DataFrames to SQLite tables.
    Uses a COPY so original DataFrames keep their original column names.
    """
    for table_name, df in dfs.items():
        df_copy = df.copy()
        df_copy.columns = [re.sub(r"[^a-zA-Z0-9_]", "_", c) for c in df_copy.columns]
        df_copy.to_sql(table_name, con=engine, if_exists="replace", index=False)
        logger.info(f"  ✔ Saved table '{table_name}' ({len(df_copy)} rows)")


def save_processed_csvs(dfs: dict[str, pd.DataFrame]) -> None:
    """Save processed DataFrames to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in dfs.items():
        out = PROCESSED_DIR / f"{name}.csv"
        df.to_csv(out, index=False)
        logger.info(f"  ✔ Saved CSV '{out.name}'")


# ═══════════════════════════════════════════════════════════════════════════════
#   PIPELINE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline() -> dict[str, pd.DataFrame]:
    """
    Full ingestion pipeline.
    Returns a dict of table_name → DataFrame for downstream use.
    """
    logger.info("═" * 60)
    logger.info("  Enterprise HR AI  —  Data Pipeline Starting")
    logger.info("═" * 60)

    dfs = {
        "attrition": load_attrition(),
        "hr_performance": load_hr_performance(),
        "occupations": load_occupations(),
        "essential_skills": load_essential_skills(),
        "software_skills": load_software_skills(),
        "performance_pro": load_performance_pro(),
        "performance_kpi": load_performance_dataset(),
    }

    # ── persist ──────────────────────────────────────────────────────────────
    engine = create_engine(DATABASE_URL, echo=False)
    logger.info("Saving to SQLite …")
    save_to_db(dfs, engine)

    logger.info("Saving processed CSVs …")
    save_processed_csvs(dfs)

    logger.success("Pipeline complete ✔")
    logger.info("═" * 60)
    return dfs


# ═══════════════════════════════════════════════════════════════════════════════
#   SINGLETON CACHE  (used by routers to avoid re-loading)
# ═══════════════════════════════════════════════════════════════════════════════

_CACHE: dict[str, pd.DataFrame] | None = None


def get_data() -> dict[str, pd.DataFrame]:
    """Return cached DataFrames, running pipeline on first call."""
    global _CACHE
    if _CACHE is None:
        _CACHE = run_pipeline()
    return _CACHE


if __name__ == "__main__":
    run_pipeline()
