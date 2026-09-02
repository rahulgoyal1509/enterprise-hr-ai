# Enterprise HR AI Workforce Intelligence Platform

<div align="center">

![Platform](https://img.shields.io/badge/AI%20HR%20Platform-v1.0-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11+-green?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-teal?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=for-the-badge)

**A production-grade AI-powered HR Analytics platform with attrition prediction, skill gap analysis, succession planning, RAG policy Q&A, and resume screening.**

</div>

---

## Architecture

```
HR Data Sources (7 CSVs)
        │
        ▼
Data Ingestion Layer (pandas + SQLite)
        │
        ▼
Central HR Data Platform (SQLAlchemy)
        │
   ┌────┴────┐
   │         │
ML Engine   NLP Engine       Skill Engine
(RF + GB)  (FAISS RAG)      (O*NET Gap)
   │         │                   │
   └────┬────┘───────────────────┘
        │
    FastAPI (9 endpoints)
        │
    Streamlit (8 pages)
```

## Features

| Module | Capability |
|---|---|
| **ML Engine** | Attrition Risk (RandomForest), Performance Prediction (GradientBoosting), Engagement Clustering (KMeans) |
| **Skill Gap Engine** | O*NET role mapping, essential + software skill gaps, personalised upskilling plans |
| **NLP / RAG Engine** | Policy Q&A (FAISS vector search), sentiment analysis, TF-IDF resume matching |
| **FastAPI** | 9 REST endpoints with Pydantic validation and auto docs |
| **Streamlit Dashboard** | 8 interactive pages with Plotly visualisations |

## Data Sources

| File | Description |
|---|---|
| `employee_attrition.csv` | IBM HR Analytics — 1,470 employees × 35 features |
| `Cleaned_HR_Data_Analysis.csv` | Performance, engagement, training data |
| `occupation_data.csv` | O*NET occupation titles and descriptions |
| `essential_skills.csv` | O*NET skill importance scores by role |
| `software_skills.csv` | O*NET software/technology requirements |
| `employee_performance_pro.csv` | Extended performance + attrition risk |
| `Employee_Performance_Dataset.csv` | KPI, attendance, peer ratings |

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run setup (pipeline + model training)
```bash
python setup.py
```

### 3. Start FastAPI backend
```bash
uvicorn app.backend.main:app --reload --port 8000
```

### 4. Start Streamlit dashboard
```bash
streamlit run app/frontend/streamlit_app.py
```

### Access
- **Dashboard**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Project Structure

```
enterprise_hr_ai/
├── data/
│   ├── raw/                    ← Source CSV files (7 datasets)
│   ├── processed/              ← Cleaned CSV outputs
│   ├── hr_policies/            ← HR policy text files (for RAG)
│   ├── resumes/                ← Candidate resume files
│   └── job_descriptions/       ← Job description files
├── app/
│   ├── backend/
│   │   ├── main.py             ← FastAPI application
│   │   ├── routers/            ← 5 API routers
│   │   ├── models/             ← Pydantic schemas
│   │   └── services/           ← 4 engine services
│   └── frontend/
│       └── streamlit_app.py    ← 8-page Streamlit dashboard
├── models/                     ← Trained ML model files (.pkl)
├── notebooks/                  ← 4 EDA + ML notebooks (.py)
├── config/
│   └── settings.py             ← Central configuration
├── setup.py                    ← One-shot setup runner
└── requirements.txt
```

## Dashboard Pages

1. **🏠 Executive Dashboard** — KPI cards, attrition by dept, skill gaps, upskilling
2. **⚠️ Attrition Risk** — Risk table, feature importance, income vs risk scatter
3. **📈 Performance Analytics** — Leaderboard, KPI vs performance, training impact
4. **🎯 Skill Gap Analysis** — O*NET gaps, hot technologies, role skill explorer, upskilling plan generator
5. **🔁 Succession Planning** — Readiness matrix, high-potential identification
6. **🤖 AI Policy Assistant** — Chat-based RAG Q&A over HR policies
7. **📋 Resume Screener** — Batch ranking + custom resume vs JD matching
8. **📊 EDA Explorer** — Interactive custom chart builder + pre-built analyses

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/attrition/predict` | Predict attrition risk for one employee |
| GET | `/api/v1/attrition/risk-summary` | Dept-level risk breakdown |
| GET | `/api/v1/attrition/feature-importance` | Model feature importances |
| POST | `/api/v1/performance/predict` | Predict performance score |
| GET | `/api/v1/performance/leaderboard` | Top performers |
| GET | `/api/v1/skills/org-gap-analysis` | Org-wide skill gaps |
| POST | `/api/v1/skills/upskilling-plan` | Individual upskilling plan |
| GET | `/api/v1/succession/candidates` | Succession pipeline |
| POST | `/api/v1/ai/policy/query` | RAG policy Q&A |
| POST | `/api/v1/ai/resume/match` | Resume vs JD matching |

## Tech Stack

- **ML**: scikit-learn (RandomForest, GradientBoosting, KMeans)
- **NLP**: sentence-transformers, FAISS, TF-IDF, TextBlob
- **API**: FastAPI + Uvicorn + Pydantic
- **Frontend**: Streamlit + Plotly
- **DB**: SQLite + SQLAlchemy
- **Data**: pandas, numpy
