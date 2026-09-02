"""
NLP / AI Engine
────────────────
• HR Policy RAG Q&A  — FAISS + sentence-transformers
• Sentiment Analysis — TextBlob on feedback text
• Resume Matching    — TF-IDF cosine similarity vs JDs
• Synthetic data gen — HR policies + job descriptions + resumes
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    FAISS_INDEX_PATH, FAISS_CHUNKS_PATH,
    POLICIES_DIR, RESUMES_DIR, JD_DIR
)

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")


# ═══════════════════════════════════════════════════════════════════════════════
#   SYNTHETIC HR POLICY DOCUMENTS
# ═══════════════════════════════════════════════════════════════════════════════

HR_POLICIES = {
    "leave_policy.txt": """
ENTERPRISE HR LEAVE POLICY
Version 2.1 | Effective: January 2024

1. ANNUAL LEAVE
   Employees are entitled to 21 working days of paid annual leave per calendar year.
   Annual leave accrues at 1.75 days per month of service.
   Unused leave up to 10 days may be carried forward to the next year.
   Leave must be applied for at least 5 working days in advance.

2. SICK LEAVE
   Employees are entitled to 15 days of paid sick leave per year.
   A medical certificate is required for absences exceeding 3 consecutive days.
   Sick leave does not accrue and cannot be carried forward.
   Sick leave during probation period is unpaid.

3. MATERNITY LEAVE
   Female employees are entitled to 26 weeks of paid maternity leave.
   Maternity leave can begin 8 weeks before the expected date of delivery.
   Employees on maternity leave retain all benefits and seniority.

4. PATERNITY LEAVE
   Male employees are entitled to 15 days of paid paternity leave.
   Paternity leave must be taken within 3 months of the child's birth.

5. EMERGENCY LEAVE
   3 days of paid emergency leave per year for unforeseen family emergencies.
   Requires documentation within 5 working days of returning to work.

6. LEAVE WITHOUT PAY (LWP)
   Employees who exhaust their leave entitlement may apply for LWP.
   LWP requires manager approval and HR sign-off.
   Maximum continuous LWP is 90 days.

7. PUBLIC HOLIDAYS
   Employees are entitled to all 10 national public holidays.
   Working on a public holiday entitles employees to a day off in lieu.

8. LEAVE APPLICATION PROCESS
   All leave applications must be submitted through the HR portal.
   Manager must approve/reject within 2 business days.
   Unapproved leave will be treated as absence without permission.
""",

    "remote_work_policy.txt": """
ENTERPRISE HR REMOTE WORK POLICY
Version 1.5 | Effective: March 2024

1. ELIGIBILITY
   Remote work is available to all permanent employees who have completed 6 months of service.
   Probationary employees are not eligible for remote work arrangements.
   Roles requiring physical presence (Lab Tech, Production) are excluded.

2. HYBRID WORK MODEL
   The standard model is 3 days in office, 2 days remote per week.
   Core office days are Tuesday, Wednesday, and Thursday.
   Monday and Friday are flexible remote days by default.

3. EQUIPMENT AND INFRASTRUCTURE
   Company provides a laptop, monitor, and ergonomic accessories for remote work.
   Employees are responsible for a stable internet connection (minimum 25 Mbps).
   Company will reimburse internet costs up to INR 1500 / USD 20 per month.
   VPN usage is mandatory when accessing company systems remotely.

4. WORK HOURS AND AVAILABILITY
   Employees must maintain core hours: 10 AM – 4 PM in their local time zone.
   Total work hours remain unchanged (8 hours per day).
   Employees must respond to messages within 2 hours during core hours.

5. SECURITY AND COMPLIANCE
   Employees must not use public Wi-Fi without VPN.
   Company data must not be stored on personal devices.
   Employees must lock screens when stepping away from devices.
   Any security incidents must be reported to IT within 1 hour.

6. PERFORMANCE AND MONITORING
   Performance expectations are identical for remote and in-office employees.
   Managers must conduct weekly 1:1 check-ins with remote team members.
   Remote work privileges may be revoked for performance or policy violations.

7. TRAVEL AND EXPENSE
   Occasional travel to office or client site is expected even in remote roles.
   Travel expenses are reimbursed per company travel policy.
""",

    "payroll_policy.txt": """
ENTERPRISE HR PAYROLL POLICY
Version 3.0 | Effective: April 2024

1. PAYROLL CYCLE
   Salaries are processed on the 28th of each month.
   If the 28th falls on a weekend/holiday, payment is made the preceding Friday.
   Payslips are available on the HR portal by the 25th.

2. SALARY COMPONENTS
   Basic Salary: 50% of CTC
   House Rent Allowance (HRA): 40% of basic (metro cities) / 30% (non-metro)
   Special Allowance: Balance of CTC after other components
   Performance Bonus: Paid quarterly based on performance rating
   Annual Increment: Effective April 1st each year

3. DEDUCTIONS
   Provident Fund (PF): 12% of basic salary (employee + employer)
   Professional Tax: As per state regulations
   TDS: As per Income Tax Act provisions
   Group Medical Insurance: INR 500 per month

4. OVERTIME AND ADDITIONAL COMPENSATION
   Overtime is compensated at 1.5x the hourly rate for non-exempt employees.
   Working on weekends is compensated with a comp-off within 30 days.
   On-call allowance: INR 5000 / USD 60 per month for eligible roles.

5. SALARY INCREMENTS
   Annual performance-based increments range from 5% to 25% of CTC.
   Exceptional performers (rating 5) are eligible for off-cycle increments.
   Promotion increments are processed within 30 days of promotion date.

6. TAX AND COMPLIANCE
   Company deducts TDS monthly based on the employee's tax declaration.
   Employees must submit investment proof by January 31st each year.
   Form 16 is issued by June 15th each year.
""",

    "learning_policy.txt": """
ENTERPRISE HR LEARNING AND DEVELOPMENT POLICY
Version 2.0 | Effective: January 2024

1. LEARNING PHILOSOPHY
   The company is committed to continuous learning and employee development.
   Each employee receives an annual Learning Budget of INR 50,000 / USD 600.
   Learning time of 5% (approx. 2 hours/week) is protected during work hours.

2. MANDATORY TRAINING
   All employees must complete:
   - Annual Information Security Awareness (4 hours)
   - Code of Conduct and Ethics (2 hours)
   - Diversity, Equity & Inclusion (3 hours)
   - Role-specific compliance training as applicable

3. OPTIONAL TRAINING AND CERTIFICATION
   Employees may pursue external certifications with manager approval.
   Pre-approved certifications include: AWS, GCP, Azure, PMP, CFA, SHRM, etc.
   The company sponsors certification exam fees for first attempts.
   Successful certification earns a one-time bonus of INR 10,000 / USD 120.

4. INTERNAL LEARNING PROGRAMS
   Monthly tech talks and knowledge sharing sessions.
   Annual Hackathon with prizes for top teams.
   Mentorship program: 6-month structured mentoring cycles.
   Leadership development tracks for high-potential employees.

5. TUITION ASSISTANCE
   Employees pursuing job-related higher education may apply for tuition assistance.
   Maximum reimbursement: INR 200,000 / USD 2500 per academic year.
   Bond period: 2 years of service post-completion required.

6. LEARNING MANAGEMENT SYSTEM
   All trainings are tracked on the company LMS (learning.company.com).
   Training completion is factored into the annual performance review.
   Managers can assign mandatory courses to their team members.
"""
}

JOB_DESCRIPTIONS = {
    "ml_engineer.txt": """
JOB TITLE: Machine Learning Engineer
DEPARTMENT: Technology / AI
LEVEL: Mid-Senior (Level 3-4)

ABOUT THE ROLE:
We are looking for a Machine Learning Engineer to join our AI team and build 
production-grade ML systems that power our HR intelligence platform.

KEY RESPONSIBILITIES:
• Design, develop, and deploy machine learning models (classification, regression, clustering)
• Build ML pipelines for data preprocessing, feature engineering, and model training
• Implement MLOps practices: versioning, monitoring, A/B testing, model retraining
• Collaborate with data engineers to build scalable data pipelines
• Work with product teams to translate business requirements into ML solutions
• Optimize model performance and inference latency
• Write clean, well-documented, production-ready Python code

REQUIRED SKILLS:
• 4+ years of ML/DS experience
• Strong Python (scikit-learn, pandas, numpy, PyTorch or TensorFlow)
• Experience with MLflow, DVC, or similar MLOps tools
• Cloud ML platforms: AWS SageMaker, GCP Vertex AI, or Azure ML
• Feature stores, model registries, and serving infrastructure
• SQL and data warehouse experience (Snowflake, BigQuery, Redshift)
• Statistical modeling and experimental design

NICE TO HAVE:
• Experience with LLMs, fine-tuning, RAG systems
• Knowledge of Spark, Kafka for streaming ML
• Familiarity with Kubernetes and Docker
• Publications or open-source contributions

COMPENSATION: ₹18–28 LPA / $90,000–$130,000 USD
""",

    "data_analyst.txt": """
JOB TITLE: Data Analyst
DEPARTMENT: Business Intelligence
LEVEL: Mid (Level 2-3)

ABOUT THE ROLE:
We are seeking a skilled Data Analyst to transform complex HR and business data 
into actionable insights that drive strategic decisions.

KEY RESPONSIBILITIES:
• Analyze large datasets to identify trends, patterns, and insights
• Build and maintain dashboards and reports using BI tools
• Partner with HR, Finance, and Operations teams to understand data needs
• Perform statistical analysis and hypothesis testing
• Create data visualizations that communicate insights effectively
• Write SQL queries to extract and transform data from various sources
• Automate recurring reports and data workflows

REQUIRED SKILLS:
• 2+ years of data analysis experience
• Advanced SQL (window functions, CTEs, subqueries)
• Python or R for data analysis (pandas, matplotlib, seaborn)
• Experience with BI tools: Tableau, Power BI, or Looker
• Strong Excel/Google Sheets skills
• Statistical knowledge: regression, correlation, significance testing
• Excellent communication and presentation skills

NICE TO HAVE:
• Experience with dbt for data transformation
• Knowledge of Google Analytics or Adobe Analytics
• Familiarity with A/B testing and experimentation
• People analytics experience

COMPENSATION: ₹8–15 LPA / $55,000–$85,000 USD
""",

    "backend_engineer.txt": """
JOB TITLE: Backend Software Engineer
DEPARTMENT: Technology / Platform
LEVEL: Mid-Senior (Level 3-4)

ABOUT THE ROLE:
We are looking for a Backend Engineer to design and build the APIs and services 
that power our enterprise HR AI platform.

KEY RESPONSIBILITIES:
• Design and develop RESTful APIs and microservices using Python/FastAPI
• Build scalable, high-performance backend systems
• Design database schemas and optimize SQL queries (PostgreSQL)
• Implement authentication, authorization, and security best practices
• Write unit and integration tests with >80% code coverage
• Participate in code reviews and architectural discussions
• Collaborate with frontend and ML teams for seamless integration
• Monitor and troubleshoot production systems

REQUIRED SKILLS:
• 4+ years of backend development experience
• Strong Python (FastAPI, Django, or Flask)
• PostgreSQL or similar relational databases
• REST API design principles and OpenAPI/Swagger
• Redis for caching and message queuing
• Docker and container orchestration (Kubernetes)
• Git, CI/CD (GitHub Actions, Jenkins)
• Microservices architecture patterns

NICE TO HAVE:
• Experience with message queues (Kafka, RabbitMQ)
• Knowledge of GraphQL
• Cloud deployment (AWS, GCP, Azure)
• Familiarity with ML model serving

COMPENSATION: ₹15–25 LPA / $80,000–$120,000 USD
"""
}

CANDIDATE_RESUMES = {
    "candidate_001.txt": """
RESUME - CANDIDATE 001

NAME: Arjun Sharma
EMAIL: arjun.sharma@email.com | PHONE: +91-9876543210
LOCATION: Bangalore, India | LINKEDIN: linkedin.com/in/arjunsharma-ml

SUMMARY:
Experienced Machine Learning Engineer with 5 years of building and deploying production ML systems.
Passionate about applying AI to solve real-world business problems. Strong background in Python, 
scikit-learn, TensorFlow, and cloud ML platforms.

WORK EXPERIENCE:

Senior ML Engineer | TechCorp India | 2022–Present
• Built attrition prediction model (RandomForest) achieving 87% accuracy for 50,000 employees
• Deployed 12 ML models to AWS SageMaker with auto-scaling serving
• Implemented MLflow for model versioning and experiment tracking
• Reduced model inference latency by 40% through quantization and caching
• Led a team of 3 junior data scientists

Data Scientist | Analytics Co | 2020–2022
• Developed customer churn model using XGBoost, reducing churn by 15%
• Built recommendation engine serving 2M users daily
• Created automated feature engineering pipeline with Feast feature store
• Presented insights to C-suite executives monthly

ML Engineer Intern | StartupXYZ | 2019–2020
• Implemented NLP sentiment analysis for 10K+ customer reviews
• Built data pipelines with Apache Airflow

SKILLS:
Python, scikit-learn, TensorFlow, PyTorch, pandas, numpy, SQL, PostgreSQL,
AWS SageMaker, GCP Vertex AI, MLflow, DVC, Docker, Kubernetes, Spark, FAISS,
Transformers (HuggingFace), FastAPI, Git, CI/CD

EDUCATION:
M.Tech in Computer Science (AI Specialization) | IIT Bangalore | 2019
B.Tech in Computer Science | NIT Trichy | 2017

CERTIFICATIONS:
• AWS Certified Machine Learning – Specialty (2023)
• Google Professional Data Engineer (2022)
• Deep Learning Specialization – Coursera (2020)

PROJECTS:
• HR Analytics Dashboard: Built full-stack ML pipeline for workforce analytics
• NLP Resume Screener: TF-IDF + BERT-based resume ranking system
• Time Series Forecasting: LSTM-based demand forecasting for supply chain
""",

    "candidate_002.txt": """
RESUME - CANDIDATE 002

NAME: Priya Reddy
EMAIL: priya.reddy@email.com | PHONE: +91-9845123456
LOCATION: Hyderabad, India

SUMMARY:
Data Analyst with 3 years of experience transforming complex datasets into business insights.
Expertise in SQL, Python, Power BI, and statistical analysis. Strong communication skills 
with a track record of influencing data-driven decisions.

WORK EXPERIENCE:

Senior Data Analyst | MegaCorp | 2023–Present
• Built executive HR dashboard in Power BI tracking 20+ KPIs for 5,000 employees
• Performed attrition analysis identifying 3 key drivers, reducing turnover by 12%
• Automated monthly reporting pipeline using Python, saving 20 hours/month
• Designed A/B tests for training programs, measuring ROI of learning investments
• Collaborated with HRBP team to translate analysis into actionable HR policies

Data Analyst | RetailCo | 2021–2023
• Analyzed sales data for 500 stores across India, identifying regional trends
• Built profitability dashboards showing product/category/region breakdowns
• Wrote complex SQL queries joining 15+ tables for customer 360 view
• Created employee engagement survey analysis using statistical techniques

Junior Analyst | ConsultingFirm | 2020–2021
• Supported data collection and cleaning for client analytics projects
• Created Excel models for financial forecasting

SKILLS:
SQL (Advanced), Python (pandas, matplotlib, seaborn, plotly), Power BI, Tableau,
Google Analytics, Excel (Advanced), R, Statistical Analysis, A/B Testing,
Google BigQuery, Snowflake, dbt, Git

EDUCATION:
MBA (Business Analytics) | ISB Hyderabad | 2020
B.Com (Statistics) | Osmania University | 2018

CERTIFICATIONS:
• Microsoft Power BI Data Analyst Associate (2023)
• Google Data Analytics Professional Certificate (2021)
• Tableau Desktop Specialist (2022)

KEY ACHIEVEMENTS:
• Reduced reporting time by 60% through automation
• Identified ₹2 Cr cost-saving opportunity through spend analysis
• Speaker at HR Analytics Conference 2023
""",

    "candidate_003.txt": """
RESUME - CANDIDATE 003

NAME: Rahul Kumar
EMAIL: rahul.kumar@email.com | PHONE: +91-9123456789
LOCATION: Pune, India | GITHUB: github.com/rahulkumar-dev

SUMMARY:
Backend Software Engineer with 4 years of experience building scalable APIs and microservices.
Expert in Python, FastAPI, PostgreSQL, and cloud infrastructure. Passionate about clean code,
testing, and developer experience.

WORK EXPERIENCE:

Backend Engineer | FinTech Startup | 2022–Present
• Designed and built 25+ REST APIs using FastAPI serving 500K+ requests/day
• Migrated monolith to microservices architecture, improving deployment frequency 5x
• Optimized PostgreSQL queries reducing average response time from 800ms to 80ms
• Implemented JWT authentication and RBAC for enterprise security compliance
• Set up CI/CD pipeline with GitHub Actions, reducing deployment time by 70%
• Containerized all services with Docker, deployed to AWS EKS (Kubernetes)

Python Developer | Enterprise IT | 2020–2022
• Built internal automation tools saving 15 hours/week across teams
• Developed ETL pipelines processing 10GB daily with Apache Airflow
• Created REST APIs using Flask integrated with Oracle and PostgreSQL databases
• Wrote comprehensive unit and integration tests (92% coverage)

Software Engineer Intern | WebAgency | 2019–2020
• Built web scraping tools and data collection pipelines
• Developed REST endpoints for client management platform

SKILLS:
Python, FastAPI, Flask, PostgreSQL, Redis, Docker, Kubernetes, AWS (EC2, RDS, S3, EKS),
GitHub Actions, Apache Airflow, Kafka, SQLAlchemy, Pydantic, pytest, REST APIs,
GraphQL, Git, Linux, Nginx, Celery

EDUCATION:
B.Tech in Information Technology | PICT Pune | 2019

CERTIFICATIONS:
• AWS Solutions Architect Associate (2023)
• Docker Certified Associate (2022)
• Python Institute PCPP (2021)

OPEN SOURCE:
• Contributed to FastAPI (15 PRs merged)
• Built "fastapi-auth-kit" library with 500+ GitHub stars
• Active contributor to SQLAlchemy documentation
"""
}


def generate_synthetic_files():
    """Write all synthetic policy, JD, and resume files to disk."""
    POLICIES_DIR.mkdir(parents=True, exist_ok=True)
    JD_DIR.mkdir(parents=True, exist_ok=True)
    RESUMES_DIR.mkdir(parents=True, exist_ok=True)

    for fname, content in HR_POLICIES.items():
        (POLICIES_DIR / fname).write_text(content.strip(), encoding="utf-8")
    logger.info(f"HR policies written: {list(HR_POLICIES.keys())}")

    for fname, content in JOB_DESCRIPTIONS.items():
        (JD_DIR / fname).write_text(content.strip(), encoding="utf-8")
    logger.info(f"Job descriptions written: {list(JOB_DESCRIPTIONS.keys())}")

    for fname, content in CANDIDATE_RESUMES.items():
        (RESUMES_DIR / fname).write_text(content.strip(), encoding="utf-8")
    logger.info(f"Resumes written: {list(CANDIDATE_RESUMES.keys())}")


# ═══════════════════════════════════════════════════════════════════════════════
#   POLICY RAG ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def _chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Split text into overlapping word chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i: i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def build_policy_index() -> None:
    """Build FAISS vector index from HR policy documents (using TF-IDF as fallback)."""
    logger.info("Building policy RAG index …")

    # Read all policy files
    all_chunks = []
    all_metadata = []

    policy_files = list(POLICIES_DIR.glob("*.txt"))
    if not policy_files:
        generate_synthetic_files()
        policy_files = list(POLICIES_DIR.glob("*.txt"))

    for fpath in policy_files:
        text = fpath.read_text(encoding="utf-8")
        chunks = _chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({"source": fpath.name, "chunk_id": i})

    logger.info(f"Total chunks: {len(all_chunks)}")

    # Try to use sentence-transformers, fall back to TF-IDF
    try:
        from sentence_transformers import SentenceTransformer
        import faiss

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(all_chunks, show_progress_bar=True, batch_size=32)
        embeddings = np.array(embeddings, dtype="float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        faiss.write_index(index, str(FAISS_INDEX_PATH))
        with open(FAISS_CHUNKS_PATH, "wb") as f:
            pickle.dump({"chunks": all_chunks, "metadata": all_metadata,
                         "engine": "faiss", "model": "all-MiniLM-L6-v2"}, f)
        logger.success("FAISS index built with sentence-transformers")

    except Exception as e:
        logger.warning(f"FAISS/sentence-transformers unavailable ({e}), using TF-IDF fallback")
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(all_chunks)

        with open(FAISS_CHUNKS_PATH, "wb") as f:
            pickle.dump({
                "chunks": all_chunks,
                "metadata": all_metadata,
                "engine": "tfidf",
                "vectorizer": vectorizer,
                "tfidf_matrix": tfidf_matrix
            }, f)
        logger.success("TF-IDF policy index built (fallback)")


def query_policy(question: str, top_k: int = 5) -> dict:
    """Query the policy RAG index and return relevant excerpts."""
    if not FAISS_CHUNKS_PATH.exists():
        build_policy_index()

    with open(FAISS_CHUNKS_PATH, "rb") as f:
        store = pickle.load(f)

    chunks = store["chunks"]
    metadata = store["metadata"]
    engine = store.get("engine", "tfidf")

    if engine == "faiss":
        try:
            from sentence_transformers import SentenceTransformer
            import faiss

            model = SentenceTransformer(store.get("model", "all-MiniLM-L6-v2"))
            index = faiss.read_index(str(FAISS_INDEX_PATH))

            q_emb = model.encode([question], convert_to_numpy=True).astype("float32")
            faiss.normalize_L2(q_emb)
            scores, ids = index.search(q_emb, top_k)

            results = []
            for score, idx in zip(scores[0], ids[0]):
                if idx >= 0:
                    results.append({
                        "text": chunks[idx],
                        "source": metadata[idx]["source"],
                        "relevance_score": round(float(score), 4)
                    })
        except Exception:
            engine = "tfidf"

    if engine == "tfidf":
        vectorizer = store["vectorizer"]
        tfidf_matrix = store["tfidf_matrix"]
        q_vec = vectorizer.transform([question])
        sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
        top_ids = sims.argsort()[::-1][:top_k]
        results = [
            {
                "text": chunks[i],
                "source": metadata[i]["source"],
                "relevance_score": round(float(sims[i]), 4)
            }
            for i in top_ids if sims[i] > 0
        ]

    # Compose answer
    if not results:
        answer = "I could not find relevant information in the HR policy documents for your query."
    else:
        top_context = results[0]["text"]
        answer = (
            f"Based on the **{results[0]['source'].replace('_', ' ').replace('.txt', '').title()}**:\n\n"
            f"{top_context}\n\n"
            f"*(Source: {results[0]['source']} — relevance: {results[0]['relevance_score']:.2%})*"
        )

    return {
        "question": question,
        "answer": answer,
        "sources": results,
        "num_sources": len(results)
    }


# ═══════════════════════════════════════════════════════════════════════════════
#   SENTIMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_sentiment(text: str) -> dict:
    """Analyse sentiment of manager feedback or review text."""
    try:
        from textblob import TextBlob
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
    except Exception:
        polarity = 0.0
        subjectivity = 0.5

    if polarity > 0.2:
        label = "Positive"
    elif polarity < -0.2:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "polarity": round(polarity, 4),
        "subjectivity": round(subjectivity, 4),
        "sentiment": label,
        "confidence": round(abs(polarity), 4)
    }


def batch_sentiment(texts: list[str]) -> pd.DataFrame:
    """Run sentiment on a list of texts."""
    results = [analyze_sentiment(t) for t in texts]
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
#   RESUME MATCHING
# ═══════════════════════════════════════════════════════════════════════════════

def match_resume_to_jd(resume_text: str, jd_name: str) -> dict:
    """
    Match a resume against a specific job description.
    Returns similarity score and recommendation.
    """
    jd_path = JD_DIR / jd_name
    if not jd_path.exists():
        generate_synthetic_files()

    if not jd_path.exists():
        return {"error": f"Job description '{jd_name}' not found"}

    jd_text = jd_path.read_text(encoding="utf-8")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=3000)
    tfidf = vectorizer.fit_transform([jd_text, resume_text])
    sim_score = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])

    # Extract matched keywords
    jd_words = set(jd_text.lower().split())
    resume_words = set(resume_text.lower().split())
    common = jd_words & resume_words
    important_matches = [
        w for w in common
        if len(w) > 4 and w not in {"years", "experience", "skills", "work", "team", "will", "using"}
    ][:15]

    return {
        "jd_file": jd_name,
        "match_score": round(sim_score * 100, 2),
        "match_percentage": f"{sim_score * 100:.1f}%",
        "recommendation": (
            "Strong Match — Shortlist for Interview" if sim_score >= 0.3 else
            "Moderate Match — Consider for Screening" if sim_score >= 0.15 else
            "Weak Match — Review Manually"
        ),
        "matched_keywords": important_matches,
        "shortlist": sim_score >= 0.3
    }


def rank_all_resumes(jd_name: str) -> list[dict]:
    """Rank all available resumes against a JD."""
    resume_files = list(RESUMES_DIR.glob("*.txt")) + list(RESUMES_DIR.glob("*.pdf"))
    results = []

    for rf in resume_files:
        text = rf.read_text(encoding="utf-8", errors="ignore")
        match = match_resume_to_jd(text, jd_name)
        match["resume_file"] = rf.name
        results.append(match)

    return sorted(results, key=lambda x: x.get("match_score", 0), reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
#   INITIALISER
# ═══════════════════════════════════════════════════════════════════════════════

def initialize_nlp():
    """Generate synthetic files and build policy index."""
    logger.info("Initialising NLP engine …")
    generate_synthetic_files()
    build_policy_index()
    logger.success("NLP engine ready ✔")


if __name__ == "__main__":
    initialize_nlp()
    result = query_policy("How many days of annual leave are employees entitled to?")
    print(result["answer"])
