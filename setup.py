"""
Enterprise HR AI — One-shot Setup Script
Run this FIRST before starting FastAPI or Streamlit.
  python setup.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")


def main():
    logger.info("=" * 65)
    logger.info("   Enterprise HR AI  —  Project Setup")
    logger.info("=" * 65)

    # 1. Data pipeline
    logger.info("\n[1/4] Running data pipeline …")
    from app.backend.services.data_pipeline import run_pipeline
    data = run_pipeline()

    # 2. Train ML models
    logger.info("\n[2/4] Training ML models …")
    from app.backend.services.ml_engine import train_all_models
    train_all_models(data)

    # 3. NLP / RAG init
    logger.info("\n[3/4] Initialising NLP engine (policy RAG + synthetic data) …")
    from app.backend.services.nlp_engine import initialize_nlp
    initialize_nlp()

    # 4. Summary
    logger.info("\n[4/4] Setup complete!")
    logger.info("=" * 65)
    logger.success("✔  Database        → data/hr_platform.db")
    logger.success("✔  Processed CSVs  → data/processed/")
    logger.success("✔  ML Models       → models/")
    logger.success("✔  Policy RAG      → models/policy_faiss.index")
    logger.success("✔  HR Policies     → data/hr_policies/")
    logger.success("✔  Resumes         → data/resumes/")
    logger.success("✔  Job Descriptions→ data/job_descriptions/")
    logger.info("=" * 65)
    logger.info("\nNext steps:")
    logger.info("  Start API:       uvicorn app.backend.main:app --reload --port 8000")
    logger.info("  Start Dashboard: streamlit run app/frontend/streamlit_app.py")
    logger.info("  API Docs:        http://localhost:8000/docs")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
