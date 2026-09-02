"""
Enterprise HR AI — Streamlit Dashboard
8-page premium workforce intelligence dashboard.
Run: streamlit run app/frontend/streamlit_app.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Workforce Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 40%, #0a1628 100%);
    color: #e2e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1923 0%, #0a1628 100%);
    border-right: 1px solid rgba(99,179,237,0.15);
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(15,25,50,0.9), rgba(20,35,70,0.85));
    border: 1px solid rgba(99,179,237,0.25);
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s, box-shadow 0.2s;
    margin-bottom: 12px;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 35px rgba(99,179,237,0.2);
}
.metric-value {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #63b3ed, #90cdf4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.78rem;
    color: #94a3b8;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 6px;
}
.metric-delta {
    font-size: 0.75rem;
    color: #68d391;
    margin-top: 4px;
}

/* Section headers */
.section-header {
    background: linear-gradient(90deg, rgba(99,179,237,0.15), transparent);
    border-left: 4px solid #63b3ed;
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    margin: 24px 0 16px 0;
    font-weight: 700;
    font-size: 1.05rem;
    color: #e2e8f0;
    letter-spacing: 0.02em;
}

/* Risk badges */
.badge-critical { background:#fc4e4e22; border:1px solid #fc4e4e; color:#fc4e4e; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.badge-high     { background:#f6ad5522; border:1px solid #f6ad55; color:#f6ad55; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.badge-medium   { background:#ecc94b22; border:1px solid #ecc94b; color:#ecc94b; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.badge-low      { background:#68d39122; border:1px solid #68d391; color:#68d391; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }

/* Chat bubbles */
.chat-user     { background:linear-gradient(135deg,#2b4a7a,#1a3560); border-radius:12px 12px 4px 12px; padding:12px 16px; margin:8px 0; color:#e2e8f0; }
.chat-bot      { background:linear-gradient(135deg,#1a2e1a,#0f2010); border:1px solid rgba(104,211,145,0.3); border-radius:12px 12px 12px 4px; padding:12px 16px; margin:8px 0; color:#e2e8f0; }

/* Plotly chart bg */
.js-plotly-plot .plotly .svg-container { border-radius: 12px; }

/* Stmetric overrides */
[data-testid="stMetric"] {
    background: rgba(15,25,50,0.7);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 12px;
    padding: 16px;
}
[data-testid="stMetricValue"] { color: #90cdf4 !important; font-weight: 700; }

/* Sidebar Nav Buttons Styling */
section[data-testid="stSidebar"] button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    text-align: left !important;
    padding: 10px 14px !important;
    margin-bottom: 6px !important;
    font-size: 0.95rem !important;
    transition: all 0.2s ease !important;
}

/* Inactive Nav Buttons */
section[data-testid="stSidebar"] button[kind="secondary"],
section[data-testid="stSidebar"] button[data-testid="baseButton-secondary"] {
    background: rgba(15, 23, 42, 0.75) !important;
    border: 1px solid rgba(148, 163, 184, 0.25) !important;
}
section[data-testid="stSidebar"] button[kind="secondary"] p,
section[data-testid="stSidebar"] button[data-testid="baseButton-secondary"] p,
section[data-testid="stSidebar"] button[kind="secondary"] span,
section[data-testid="stSidebar"] button[data-testid="baseButton-secondary"] span {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
}

/* Inactive Nav Buttons Hover */
section[data-testid="stSidebar"] button[kind="secondary"]:hover,
section[data-testid="stSidebar"] button[data-testid="baseButton-secondary"]:hover {
    background: rgba(30, 41, 59, 0.95) !important;
    border-color: #60a5fa !important;
}
section[data-testid="stSidebar"] button[kind="secondary"]:hover p,
section[data-testid="stSidebar"] button[data-testid="baseButton-secondary"]:hover p,
section[data-testid="stSidebar"] button[kind="secondary"]:hover span,
section[data-testid="stSidebar"] button[data-testid="baseButton-secondary"]:hover span {
    color: #ffffff !important;
}

/* Active Nav Button */
section[data-testid="stSidebar"] button[kind="primary"],
section[data-testid="stSidebar"] button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
    border: 1px solid #60a5fa !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.45) !important;
}
section[data-testid="stSidebar"] button[kind="primary"] p,
section[data-testid="stSidebar"] button[data-testid="baseButton-primary"] p,
section[data-testid="stSidebar"] button[kind="primary"] span,
section[data-testid="stSidebar"] button[data-testid="baseButton-primary"] span {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Plotly dark template ──────────────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"
CHART_BG = "rgba(10,14,26,0)"
COLOR_SEQ = ["#63b3ed", "#68d391", "#f6ad55", "#fc4e4e", "#b794f4",
             "#76e4f7", "#f687b3", "#fbd38d", "#9ae6b4", "#bee3f8"]


# ═══════════════════════════════════════════════════════════════════════════════
#   DATA CACHE
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner="Loading HR data …")
def load_all_data():
    from app.backend.services.data_pipeline import get_data
    return get_data()


@st.cache_data(ttl=3600, show_spinner="Scoring attrition risk …")
def get_attrition_scored():
    from app.backend.services.ml_engine import batch_predict_attrition
    data = load_all_data()
    return batch_predict_attrition(data["attrition"])


@st.cache_data(ttl=3600, show_spinner="Computing skill gaps …")
def get_skill_gaps():
    from app.backend.services.skill_engine import compute_org_skill_gaps
    data = load_all_data()
    return compute_org_skill_gaps(
        data["essential_skills"], data["software_skills"], data["occupations"]
    )


# ── helpers ───────────────────────────────────────────────────────────────────
def metric_card(value, label, delta=None, color="#63b3ed"):
    delta_html = f'<div class="metric-delta">▲ {delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="background:linear-gradient(135deg,{color},#90cdf4);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="section-header">📊 {title}</div>', unsafe_allow_html=True)


def styled_chart(fig, height=380):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=dict(family="Inter", color="#cbd5e0"),
        margin=dict(l=10, r=10, t=40, b=10),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(99,179,237,0.2)"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#   SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:20px 0 10px 0;">
        <div style="font-size:2.4rem;">🧠</div>
        <div style="font-size:1.05rem; font-weight:700; color:#63b3ed; letter-spacing:0.03em;">
            AI Workforce<br>Intelligence
        </div>
        <div style="font-size:0.7rem; color:#4a5568; margin-top:4px;">Enterprise HR Platform v1.0</div>
    </div>
    <hr style="border-color:rgba(99,179,237,0.15); margin:10px 0;">
    """, unsafe_allow_html=True)

    pages = {
        "🏠  Executive Dashboard":    "executive",
        "⚠️  Attrition Risk":         "attrition",
        "📈  Performance Analytics":  "performance",
        "🎯  Skill Gap Analysis":      "skills",
        "🔁  Succession Planning":    "succession",
        "🤖  AI Policy Assistant":     "policy",
        "📋  Resume Screener":         "resume",
        "📊  EDA Explorer":            "eda",
    }
    page_list = list(pages.keys())

    # Clear legacy st.radio key if present to prevent Streamlit 1.63 session state crash
    if "Navigation" in st.session_state:
        del st.session_state["Navigation"]

    if "current_page" not in st.session_state or st.session_state["current_page"] not in pages:
        st.session_state["current_page"] = page_list[0]

    selected_label = st.session_state["current_page"]
    for label in page_list:
        is_active = (label == selected_label)
        btn_type = "primary" if is_active else "secondary"
        if st.button(
            label,
            key=f"nav_btn_{pages[label]}",
            type=btn_type,
            use_container_width=True,
        ):
            st.session_state["current_page"] = label
            st.rerun()

    current = pages[st.session_state["current_page"]]

    st.markdown("<hr style='border-color:rgba(99,179,237,0.1);'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.7rem; color:#4a5568; text-align:center; padding:8px 0;">
        Powered by RandomForest · GradientBoosting<br>
        FAISS RAG · O*NET Skills · FastAPI
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#   PAGE 1 — EXECUTIVE DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

if current == "executive":
    st.markdown("""
    <div style="margin-bottom:24px;">
        <h1 style="font-size:1.9rem;font-weight:800;color:#e2e8f0;margin:0;">
            🧠 AI Workforce Intelligence Platform
        </h1>
        <p style="color:#718096; font-size:0.9rem; margin:4px 0 0 0;">
            Real-time HR analytics · Attrition prediction · Skill intelligence · Succession planning
        </p>
    </div>
    """, unsafe_allow_html=True)

    try:
        data   = load_all_data()
        df_att = get_attrition_scored()
        gaps   = get_skill_gaps()

        # ── KPI Cards ────────────────────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            metric_card(f"{len(df_att):,}", "Total Employees", color="#63b3ed")
        with c2:
            hr = int((df_att["RiskTier"].isin(["High","Critical"])).sum())
            metric_card(str(hr), "High Risk", color="#fc4e4e")
        with c3:
            cr = int((df_att["RiskTier"] == "Critical").sum())
            metric_card(str(cr), "Critical Risk", color="#fc4e4e")
        with c4:
            eng_col = next((c for c in ["Engagement Score","Satisfaction Score"]
                            if c in data["hr_performance"].columns), None)
            avg_eng = round(data["hr_performance"][eng_col].mean(), 1) if eng_col else 3.2
            metric_card(f"{avg_eng:.1f}/5", "Avg Engagement", color="#68d391")
        with c5:
            attr_rate = round(df_att["Attrition_enc"].mean()*100
                              if "Attrition_enc" in df_att.columns else 16.1, 1)
            metric_card(f"{attr_rate}%", "Attrition Rate", color="#f6ad55")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 1: Attrition by Dept + Risk Tier pie ──────────────────────
        col_a, col_b = st.columns([3, 2])

        with col_a:
            section("Attrition Risk by Department")
            dept = df_att.groupby("Department").agg(
                total=("EmployeeNumber","count"),
                high_risk=("RiskTier", lambda x: x.isin(["High","Critical"]).sum()),
                avg_risk=("AttritionProbability","mean")
            ).reset_index()
            dept["risk_pct"] = (dept["high_risk"]/dept["total"]*100).round(1)
            fig = px.bar(dept.sort_values("avg_risk", ascending=True),
                         x="avg_risk", y="Department", orientation="h",
                         color="avg_risk", color_continuous_scale="RdYlGn_r",
                         text=dept.sort_values("avg_risk")["risk_pct"].astype(str)+"% high risk",
                         labels={"avg_risk":"Avg Attrition Probability","Department":"Department"})
            fig.update_traces(textposition="outside")
            styled_chart(fig, 320)

        with col_b:
            section("Risk Tier Distribution")
            tier_counts = df_att["RiskTier"].value_counts().reset_index()
            tier_counts.columns = ["tier","count"]
            colors_map = {"Critical":"#fc4e4e","High":"#f6ad55","Medium":"#ecc94b","Low":"#68d391"}
            fig2 = px.pie(tier_counts, names="tier", values="count",
                          color="tier", color_discrete_map=colors_map, hole=0.55)
            fig2.update_traces(textinfo="percent+label", textfont_size=12)
            styled_chart(fig2, 320)

        # ── Row 2: Skill Gaps + Upskilling ───────────────────────────────
        col_c, col_d = st.columns([3, 2])

        with col_c:
            section("Critical Organisational Skill Gaps")
            top_gaps = gaps[gaps["gap_severity"].isin(["High","Medium"])].head(12) if not gaps.empty else pd.DataFrame()
            if not top_gaps.empty:
                fig3 = px.bar(top_gaps, x="gap_score", y="Element Name",
                              orientation="h", color="gap_severity",
                              color_discrete_map={"High":"#fc4e4e","Medium":"#f6ad55","Low":"#68d391"},
                              labels={"gap_score":"Gap Score","Element Name":"Skill"})
                styled_chart(fig3, 350)
            else:
                st.info("Run setup.py first to compute skill gaps.")

        with col_d:
            section("Top Upskilling Recommendations")
            from app.backend.services.skill_engine import COURSE_RECOMMENDATIONS
            recs = [
                {"Employee": f"Emp #{101+i}", "Skill": s, "Priority": "High" if i < 3 else "Medium"}
                for i, s in enumerate(list(COURSE_RECOMMENDATIONS.keys())[:6])
            ]
            for r in recs:
                badge = f'<span class="badge-{"high" if r["Priority"]=="High" else "medium"}">{r["Priority"]}</span>'
                st.markdown(
                    f'<div style="padding:8px 12px;margin:4px 0;background:rgba(99,179,237,0.07);'
                    f'border-radius:8px;border-left:3px solid #63b3ed;">'
                    f'<b style="color:#90cdf4">{r["Employee"]}</b> → '
                    f'<span style="color:#e2e8f0">{r["Skill"]}</span> {badge}</div>',
                    unsafe_allow_html=True
                )

        # ── Row 3: Performance dist + Engagement heatmap ─────────────────
        col_e, col_f = st.columns(2)
        with col_e:
            section("Performance Score Distribution")
            perf_df = data["performance_kpi"]
            if "Performance Score" in perf_df.columns:
                fig4 = px.histogram(perf_df, x="Performance Score", nbins=20,
                                    color_discrete_sequence=["#63b3ed"],
                                    labels={"Performance Score":"Score","count":"Employees"})
                styled_chart(fig4, 300)

        with col_f:
            section("Department Headcount")
            dept_count = df_att["Department"].value_counts().reset_index()
            dept_count.columns = ["Department","Count"]
            fig5 = px.bar(dept_count, x="Department", y="Count",
                          color="Count", color_continuous_scale="Blues",
                          labels={"Count":"Employees"})
            styled_chart(fig5, 300)

    except Exception as e:
        st.error(f"Dashboard error: {e}")
        st.info("Run `python setup.py` first to initialise the platform.")


# ═══════════════════════════════════════════════════════════════════════════════
#   PAGE 2 — ATTRITION RISK
# ═══════════════════════════════════════════════════════════════════════════════

elif current == "attrition":
    st.markdown('<h1 style="color:#fc4e4e;font-size:1.8rem;font-weight:800;">⚠️ Attrition Risk Intelligence</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#718096;">ML-powered attrition risk scoring for every employee using IBM HR Analytics data.</p>', unsafe_allow_html=True)

    try:
        df = get_attrition_scored()
        data = load_all_data()

        # KPIs
        c1,c2,c3,c4 = st.columns(4)
        with c1: metric_card(f"{len(df):,}", "Total Scored", color="#63b3ed")
        with c2: metric_card(str(int((df["RiskTier"]=="Critical").sum())), "Critical", color="#fc4e4e")
        with c3: metric_card(str(int((df["RiskTier"]=="High").sum())), "High Risk", color="#f6ad55")
        with c4:
            acc = round(df["Attrition_enc"].mean()*100 if "Attrition_enc" in df.columns else 16.1,1)
            metric_card(f"{acc}%", "Actual Attrition", color="#ecc94b")

        st.markdown("<br>", unsafe_allow_html=True)

        # Filters
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            dept_filter = st.multiselect("Filter by Department",
                                          df["Department"].unique().tolist() if "Department" in df.columns else [],
                                          default=[])
        with col_f2:
            tier_filter = st.multiselect("Filter by Risk Tier",
                                          ["Critical","High","Medium","Low"], default=["Critical","High"])
        with col_f3:
            role_options = df["JobRole"].unique().tolist() if "JobRole" in df.columns else []
            role_filter = st.multiselect("Filter by Job Role", role_options, default=[])

        filtered = df.copy()
        if dept_filter:
            filtered = filtered[filtered["Department"].isin(dept_filter)]
        if tier_filter:
            filtered = filtered[filtered["RiskTier"].isin(tier_filter)]
        if role_filter:
            filtered = filtered[filtered["JobRole"].isin(role_filter)]

        col_a, col_b = st.columns([3,2])
        with col_a:
            section(f"Risk Employee Table ({len(filtered):,} employees)")
            display_cols = [c for c in ["EmployeeNumber","Age","Department","JobRole",
                                         "MonthlyIncome","YearsAtCompany","JobSatisfaction",
                                         "OverTime","AttritionProbability","RiskTier"] if c in filtered.columns]
            st.dataframe(
                filtered[display_cols].sort_values("AttritionProbability", ascending=False)
                    .reset_index(drop=True),
                use_container_width=True, height=380
            )

        with col_b:
            section("Attrition by Job Role")
            if "JobRole" in df.columns:
                role_risk = df.groupby("JobRole")["AttritionProbability"].mean().sort_values(ascending=False).head(10)
                fig = px.bar(x=role_risk.values, y=role_risk.index, orientation="h",
                             color=role_risk.values, color_continuous_scale="Reds",
                             labels={"x":"Avg Risk","y":"Job Role"})
                styled_chart(fig, 380)

        # Feature importance + scatter
        col_c, col_d = st.columns(2)
        with col_c:
            section("Attrition Risk Drivers (Feature Importance)")
            try:
                from app.backend.services.ml_engine import get_attrition_feature_importance
                fi = get_attrition_feature_importance().head(15)
                fig2 = px.bar(fi, x="importance", y="feature", orientation="h",
                              color="importance", color_continuous_scale="Blues",
                              labels={"importance":"Importance","feature":"Feature"})
                styled_chart(fig2, 380)
            except Exception:
                st.info("Feature importance available after model training.")

        with col_d:
            section("Monthly Income vs Attrition Probability")
            if "MonthlyIncome" in df.columns:
                sample = df.sample(min(500, len(df)))
                fig3 = px.scatter(sample, x="MonthlyIncome", y="AttritionProbability",
                                  color="RiskTier", size_max=8,
                                  color_discrete_map={"Critical":"#fc4e4e","High":"#f6ad55",
                                                       "Medium":"#ecc94b","Low":"#68d391"},
                                  opacity=0.7,
                                  labels={"MonthlyIncome":"Monthly Income (USD)","AttritionProbability":"Risk Score"})
                styled_chart(fig3, 380)

        # Overtime analysis
        col_e, col_f_chart = st.columns(2)
        with col_e:
            section("Overtime vs Attrition Risk")
            if "OverTime" in df.columns:
                ot = df.groupby("OverTime")["AttritionProbability"].mean().reset_index()
                fig4 = px.bar(ot, x="OverTime", y="AttritionProbability",
                              color="OverTime", color_discrete_sequence=["#68d391","#fc4e4e"],
                              labels={"AttritionProbability":"Avg Risk Score","OverTime":"Overtime"})
                styled_chart(fig4, 300)

        with col_f_chart:
            section("Age Band vs Attrition Rate")
            if "AgeBand" in df.columns and "Attrition_enc" in df.columns:
                age_att = df.groupby("AgeBand", observed=True)["Attrition_enc"].mean().reset_index()
                age_att["Attrition_enc"] = (age_att["Attrition_enc"]*100).round(1)
                fig5 = px.line(age_att, x="AgeBand", y="Attrition_enc", markers=True,
                               color_discrete_sequence=["#63b3ed"],
                               labels={"Attrition_enc":"Attrition Rate (%)","AgeBand":"Age Band"})
                styled_chart(fig5, 300)

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Run `python setup.py` first.")


# ═══════════════════════════════════════════════════════════════════════════════
#   PAGE 3 — PERFORMANCE ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

elif current == "performance":
    st.markdown('<h1 style="color:#68d391;font-size:1.8rem;font-weight:800;">📈 Performance Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#718096;">KPI tracking, performance prediction, promotion eligibility and training impact.</p>', unsafe_allow_html=True)

    try:
        data = load_all_data()
        perf = data["performance_kpi"].copy()
        hr   = data["hr_performance"].copy()

        # KPIs
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            metric_card(f"{len(perf):,}", "Employees Tracked", color="#68d391")
        with c2:
            ps = perf["Performance Score"].mean() if "Performance Score" in perf.columns else 0
            metric_card(f"{ps:.1f}", "Avg Performance Score", color="#63b3ed")
        with c3:
            kpi = perf["KPI Score"].mean() if "KPI Score" in perf.columns else 0
            metric_card(f"{kpi:.1f}", "Avg KPI Score", color="#b794f4")
        with c4:
            promo = (perf["Promotion Eligibility"]=="Yes").sum() if "Promotion Eligibility" in perf.columns else 0
            metric_card(str(promo), "Promotion Eligible", color="#f6ad55")

        st.markdown("<br>", unsafe_allow_html=True)

        # Dept filter
        dept_opts = perf["Department"].unique().tolist() if "Department" in perf.columns else []
        selected_dept = st.selectbox("Filter by Department", ["All"] + dept_opts)
        pdf = perf if selected_dept == "All" else perf[perf["Department"] == selected_dept]

        col_a, col_b = st.columns(2)
        with col_a:
            section("Performance Score Distribution by Department")
            if "Department" in pdf.columns and "Performance Score" in pdf.columns:
                fig = px.box(pdf, x="Department", y="Performance Score",
                             color="Department", color_discrete_sequence=COLOR_SEQ,
                             points="outliers")
                styled_chart(fig, 380)

        with col_b:
            section("KPI vs Performance Score")
            if "KPI Score" in pdf.columns and "Performance Score" in pdf.columns:
                fig2 = px.scatter(pdf.sample(min(300,len(pdf))),
                                  x="KPI Score", y="Performance Score",
                                  color="Department" if "Department" in pdf.columns else None,
                                  color_discrete_sequence=COLOR_SEQ,
                                  trendline="ols", opacity=0.7,
                                  labels={"KPI Score":"KPI Score","Performance Score":"Performance Score"})
                styled_chart(fig2, 380)

        col_c, col_d = st.columns(2)
        with col_c:
            section("Training Hours vs Performance")
            if "Training Hours" in pdf.columns and "Performance Score" in pdf.columns:
                fig3 = px.scatter(pdf, x="Training Hours", y="Performance Score",
                                  color="Promotion Eligibility" if "Promotion Eligibility" in pdf.columns else None,
                                  color_discrete_map={"Yes":"#68d391","No":"#fc4e4e"},
                                  opacity=0.65,
                                  labels={"Training Hours":"Training Hours","Performance Score":"Perf Score"})
                styled_chart(fig3, 320)

        with col_d:
            section("Attendance vs Task Completion")
            if "Attendance (%)" in pdf.columns and "Task Completion (%)" in pdf.columns:
                fig4 = px.scatter(pdf.sample(min(300,len(pdf))),
                                  x="Attendance (%)", y="Task Completion (%)",
                                  color="Performance Score" if "Performance Score" in pdf.columns else None,
                                  color_continuous_scale="Viridis", opacity=0.7)
                styled_chart(fig4, 320)

        # Leaderboard
        section("🏆 Performance Leaderboard — Top 15")
        lb_cols = [c for c in ["Employee ID","Name","Department","Job Role",
                               "Performance Score","KPI Score","Attendance (%)",
                               "Peer Rating","Promotion Eligibility"] if c in perf.columns]
        top15 = perf[lb_cols].sort_values("Performance Score", ascending=False).head(15)
        st.dataframe(top15.reset_index(drop=True), use_container_width=True)

        # Engagement / satisfaction from HR performance
        section("Engagement & Satisfaction Analysis")
        eng_cols = [c for c in ["Engagement Score","Satisfaction Score","Work-Life Balance Score"]
                    if c in hr.columns]
        if eng_cols and "DepartmentType" in hr.columns:
            col_e, col_f2 = st.columns(2)
            with col_e:
                dept_eng = hr.groupby("DepartmentType")[eng_cols].mean().reset_index()
                fig5 = px.bar(dept_eng, x="DepartmentType", y=eng_cols,
                              barmode="group", color_discrete_sequence=COLOR_SEQ,
                              labels={"value":"Score","DepartmentType":"Department","variable":"Metric"})
                styled_chart(fig5, 320)
            with col_f2:
                if "Performance Score" in hr.columns:
                    perf_dist = hr["Performance Score"].value_counts().reset_index()
                    perf_dist.columns = ["Score","Count"]
                    fig6 = px.pie(perf_dist, names="Score", values="Count",
                                  color_discrete_sequence=COLOR_SEQ, hole=0.45)
                    styled_chart(fig6, 320)

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Run `python setup.py` first.")


# ═══════════════════════════════════════════════════════════════════════════════
#   PAGE 4 — SKILL GAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

elif current == "skills":
    st.markdown('<h1 style="color:#b794f4;font-size:1.8rem;font-weight:800;">🎯 Skill Gap Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#718096;">O*NET-powered skill gap mapping for all roles, hot technologies, and upskilling intelligence.</p>', unsafe_allow_html=True)

    try:
        data  = load_all_data()
        gaps  = get_skill_gaps()
        att   = get_attrition_scored()

        # KPIs
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            metric_card(str(len(gaps)), "Skills Analysed", color="#b794f4")
        with c2:
            hg = int((gaps["gap_severity"]=="High").sum()) if not gaps.empty else 0
            metric_card(str(hg), "High Severity Gaps", color="#fc4e4e")
        with c3:
            occ = data["occupations"]
            metric_card(str(len(occ)), "O*NET Occupations", color="#63b3ed")
        with c4:
            sw = data["software_skills"]
            hot = int((sw["Hot Technology"]=="Y").sum()) if "Hot Technology" in sw.columns else 0
            metric_card(str(hot), "Hot Technologies", color="#68d391")

        st.markdown("<br>", unsafe_allow_html=True)
        sev_filter = st.multiselect("Filter by Gap Severity", ["High","Medium","Low"],
                                     default=["High","Medium"])
        gap_filtered = gaps[gaps["gap_severity"].isin(sev_filter)] if not gaps.empty else gaps

        col_a, col_b = st.columns([3,2])
        with col_a:
            section("Organisation-Wide Skill Gaps")
            top = gap_filtered.head(20)
            if not top.empty:
                fig = px.bar(top, x="gap_score", y="Element Name", orientation="h",
                             color="gap_severity",
                             color_discrete_map={"High":"#fc4e4e","Medium":"#f6ad55","Low":"#68d391"},
                             labels={"gap_score":"Gap Score (0-100)","Element Name":"Skill"})
                styled_chart(fig, 500)

        with col_b:
            section("Severity Breakdown")
            if not gaps.empty:
                sev_cnt = gaps["gap_severity"].value_counts().reset_index()
                sev_cnt.columns = ["Severity","Count"]
                fig2 = px.pie(sev_cnt, names="Severity", values="Count",
                              color="Severity",
                              color_discrete_map={"High":"#fc4e4e","Medium":"#f6ad55","Low":"#68d391"},
                              hole=0.5)
                styled_chart(fig2, 280)

            section("Hot Technologies (O*NET)")
            if "Hot Technology" in data["software_skills"].columns:
                hot_tech = (data["software_skills"][data["software_skills"]["Hot Technology"]=="Y"]
                            ["Element Name"].value_counts().head(10).reset_index())
                hot_tech.columns = ["Technology","Count"]
                for _, r in hot_tech.iterrows():
                    pct = int(r["Count"] / hot_tech["Count"].max() * 100)
                    st.markdown(
                        f'<div style="margin:4px 0;">'
                        f'<div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#cbd5e0;">'
                        f'<span>{r["Technology"]}</span><span style="color:#63b3ed">{r["Count"]}</span></div>'
                        f'<div style="background:rgba(99,179,237,0.15);border-radius:4px;height:6px;">'
                        f'<div style="background:linear-gradient(90deg,#63b3ed,#b794f4);width:{pct}%;height:6px;border-radius:4px;"></div></div>'
                        f'</div>', unsafe_allow_html=True
                    )

        # Role skill explorer
        section("Role Skill Requirements Explorer")
        from app.backend.services.skill_engine import ROLE_ONET_MAP, get_role_required_skills
        role_sel = st.selectbox("Select Job Role", list(ROLE_ONET_MAP.keys()))
        role_skills = get_role_required_skills(role_sel, data["essential_skills"], data["software_skills"])

        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown(f"**Essential Skills for {role_sel}**")
            if role_skills["essential_skills"]:
                es_df = pd.DataFrame(role_skills["essential_skills"])
                fig3 = px.bar(es_df.head(12), x="importance", y="Element Name",
                              orientation="h", color="importance",
                              color_continuous_scale="Purples",
                              labels={"importance":"Importance","Element Name":"Skill"})
                styled_chart(fig3, 350)
            else:
                st.info("No O*NET data found for this role mapping.")

        with col_d:
            st.markdown(f"**Software & Technology Skills for {role_sel}**")
            sw_skills = role_skills.get("software_skills",[])
            if sw_skills:
                for i, s in enumerate(sw_skills[:15]):
                    st.markdown(
                        f'<div style="padding:5px 12px;margin:3px 0;background:rgba(183,148,244,0.1);'
                        f'border-left:3px solid #b794f4;border-radius:4px;font-size:0.82rem;color:#e2e8f0;">'
                        f'🔧 {s}</div>', unsafe_allow_html=True
                    )
            else:
                st.info("No software skills mapped for this role.")

        # Individual upskilling plan
        section("Individual Upskilling Plan Generator")
        col_e, col_ff = st.columns(2)
        with col_e:
            emp_role = st.selectbox("Employee Job Role", list(ROLE_ONET_MAP.keys()), key="up_role")
            emp_train = st.slider("Training Times Last Year", 0, 10, 2)
            emp_perf  = st.slider("Performance Rating", 1, 5, 3)
        with col_ff:
            if st.button("Generate Upskilling Plan 🚀", type="primary"):
                from app.backend.services.skill_engine import generate_upskilling_plan
                plan = generate_upskilling_plan(
                    {"JobRole": emp_role, "TrainingTimesLastYear": emp_train,
                     "PerformanceRating": emp_perf},
                    data["essential_skills"], data["software_skills"]
                )
                st.markdown(f"**Urgency:** `{plan['training_urgency']}`  |  **O*NET Code:** `{plan['onet_code']}`")
                for rec in plan["skill_recommendations"]:
                    with st.expander(f"📚 {rec['skill']} — {rec['domain']} ({rec['priority']} priority)"):
                        for c in rec["recommended_courses"]:
                            st.markdown(f"• {c}")
                        st.caption(f"Estimated: {rec['estimated_hours']} hours")

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Run `python setup.py` first.")


# ═══════════════════════════════════════════════════════════════════════════════
#   PAGE 5 — SUCCESSION PLANNING
# ═══════════════════════════════════════════════════════════════════════════════

elif current == "succession":
    st.markdown('<h1 style="color:#76e4f7;font-size:1.8rem;font-weight:800;">🔁 Succession Planning</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#718096;">Leadership pipeline, readiness tiers, and high-potential employee identification.</p>', unsafe_allow_html=True)

    try:
        data = load_all_data()
        df   = get_attrition_scored()
        from app.backend.services.skill_engine import compute_succession_scores, get_succession_pipeline

        scored = compute_succession_scores(df)

        # KPIs
        if "ReadinessTier" in scored.columns:
            c1,c2,c3,c4 = st.columns(4)
            with c1: metric_card(str(int((scored["ReadinessTier"]=="Ready Now").sum())), "Ready Now", color="#68d391")
            with c2: metric_card(str(int((scored["ReadinessTier"]=="Ready (1-2yr)").sum())), "Ready 1-2yr", color="#63b3ed")
            with c3: metric_card(str(int((scored["ReadinessTier"]=="Developing").sum())), "Developing", color="#f6ad55")
            with c4: metric_card(str(int((scored["ReadinessTier"]=="Not Ready").sum())), "Not Ready", color="#fc4e4e")

        st.markdown("<br>", unsafe_allow_html=True)
        dept_opts = scored["Department"].unique().tolist() if "Department" in scored.columns else []
        dept_sel  = st.selectbox("Filter by Department", ["All"] + dept_opts)

        col_a, col_b = st.columns([3,2])
        with col_a:
            section("Succession Candidates Table")
            cands = get_succession_pipeline(df, department=None if dept_sel=="All" else dept_sel, top_n=30)
            if cands:
                cdf = pd.DataFrame(cands)
                st.dataframe(cdf.reset_index(drop=True), use_container_width=True, height=400)

        with col_b:
            section("Readiness Tier Distribution")
            if "ReadinessTier" in scored.columns:
                tier_cnt = scored["ReadinessTier"].value_counts().reset_index()
                tier_cnt.columns = ["Tier","Count"]
                colors_r = {"Ready Now":"#68d391","Ready (1-2yr)":"#63b3ed","Developing":"#f6ad55","Not Ready":"#fc4e4e"}
                fig = px.pie(tier_cnt, names="Tier", values="Count",
                             color="Tier", color_discrete_map=colors_r, hole=0.5)
                styled_chart(fig, 280)

        col_c, col_d = st.columns(2)
        with col_c:
            section("Succession Score by Department")
            if "Department" in scored.columns:
                dept_scores = scored.groupby("Department")["SuccessionScore"].mean().sort_values(ascending=False).reset_index()
                fig2 = px.bar(dept_scores, x="Department", y="SuccessionScore",
                              color="SuccessionScore", color_continuous_scale="Greens",
                              labels={"SuccessionScore":"Avg Succession Score"})
                styled_chart(fig2, 320)

        with col_d:
            section("Succession Score vs Performance Rating")
            if "PerformanceRating" in scored.columns:
                fig3 = px.scatter(scored.sample(min(300,len(scored))),
                                  x="PerformanceRating", y="SuccessionScore",
                                  color="ReadinessTier" if "ReadinessTier" in scored.columns else None,
                                  color_discrete_map={"Ready Now":"#68d391","Ready (1-2yr)":"#63b3ed",
                                                       "Developing":"#f6ad55","Not Ready":"#fc4e4e"},
                                  opacity=0.7,
                                  labels={"PerformanceRating":"Performance Rating","SuccessionScore":"Succession Score"})
                styled_chart(fig3, 320)

        section("🌟 High-Potential Employees (Low Risk + High Score)")
        hp = scored[scored["SuccessionScore"] >= scored["SuccessionScore"].quantile(0.8)]
        if "AttritionProbability" in hp.columns:
            hp = hp[hp["AttritionProbability"] < 0.35]
        hp_cols = [c for c in ["EmployeeNumber","Department","JobRole","JobLevel",
                               "TotalWorkingYears","PerformanceRating","SuccessionScore","ReadinessTier"] if c in hp.columns]
        st.dataframe(hp[hp_cols].reset_index(drop=True), use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Run `python setup.py` first.")


# ═══════════════════════════════════════════════════════════════════════════════
#   PAGE 6 — AI POLICY ASSISTANT (RAG)
# ═══════════════════════════════════════════════════════════════════════════════

elif current == "policy":
    st.markdown('<h1 style="color:#fbd38d;font-size:1.8rem;font-weight:800;">🤖 AI HR Policy Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#718096;">Ask any question about HR policies. Powered by RAG (Retrieval-Augmented Generation) over 4 policy documents.</p>', unsafe_allow_html=True)

    # Policy doc badges
    cols = st.columns(4)
    policies = [("📅","Leave Policy","Annual, sick, maternity leave"),
                ("🏠","Remote Work Policy","Hybrid model & security"),
                ("💰","Payroll Policy","Salary, deductions, increments"),
                ("📚","Learning Policy","Training budget & certifications")]
    for col, (icon, title, desc) in zip(cols, policies):
        col.markdown(
            f'<div class="metric-card"><div style="font-size:1.8rem">{icon}</div>'
            f'<div style="font-weight:600;color:#90cdf4;margin:4px 0;">{title}</div>'
            f'<div style="font-size:0.75rem;color:#718096;">{desc}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Suggested questions
    st.markdown("**💡 Suggested Questions:**")
    q_cols = st.columns(3)
    suggestions = [
        "How many days of annual leave am I entitled to?",
        "What is the company's hybrid work policy?",
        "How is my salary calculated and when is it paid?",
        "What certifications does the company sponsor?",
        "Can I carry forward unused leave to next year?",
        "What are the core office hours for remote work?",
    ]
    for i, (col, q) in enumerate(zip(q_cols * 2, suggestions)):
        if col.button(q, key=f"sugg_{i}", use_container_width=True):
            st.session_state["policy_q"] = q

    st.markdown("<br>", unsafe_allow_html=True)

    # Chat interface
    if "policy_history" not in st.session_state:
        st.session_state["policy_history"] = []

    with st.form("policy_form", clear_on_submit=True):
        user_q = st.text_input(
            "Ask the HR Policy Assistant …",
            value=st.session_state.get("policy_q",""),
            placeholder="e.g. How many sick leave days do I get per year?",
            label_visibility="collapsed"
        )
        submitted = st.form_submit_button("Ask 🔍", type="primary", use_container_width=True)

    if submitted and user_q.strip():
        st.session_state["policy_q"] = ""
        with st.spinner("Searching policy documents …"):
            from app.backend.services.nlp_engine import query_policy
            result = query_policy(user_q.strip())
        st.session_state["policy_history"].insert(0, {
            "q": user_q.strip(), "a": result["answer"],
            "sources": result["sources"], "n": result["num_sources"]
        })

    for item in st.session_state["policy_history"]:
        st.markdown(f'<div class="chat-user">🙋 {item["q"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-bot">🤖 {item["a"]}</div>', unsafe_allow_html=True)
        with st.expander(f"📎 {item['n']} source chunks"):
            for src in item["sources"]:
                st.markdown(f"**{src['source']}** (relevance: {src['relevance_score']:.2%})")
                st.text(src["text"][:300] + "…")


# ═══════════════════════════════════════════════════════════════════════════════
#   PAGE 7 — RESUME SCREENER
# ═══════════════════════════════════════════════════════════════════════════════

elif current == "resume":
    st.markdown('<h1 style="color:#f687b3;font-size:1.8rem;font-weight:800;">📋 AI Resume Screener</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#718096;">TF-IDF cosine similarity matching between candidate resumes and job descriptions.</p>', unsafe_allow_html=True)

    from app.backend.services.nlp_engine import rank_all_resumes, match_resume_to_jd
    from config.settings import JD_DIR, RESUMES_DIR

    tab1, tab2 = st.tabs(["📂 Batch Rank Resumes", "✏️ Custom Resume Match"])

    with tab1:
        jd_files = [f.name for f in JD_DIR.glob("*.txt")]
        if not jd_files:
            st.info("Job descriptions will be generated on first run of setup.py")
        else:
            selected_jd = st.selectbox("Select Job Description", jd_files)
            if st.button("🔍 Rank All Candidates", type="primary"):
                with st.spinner("Analysing resumes …"):
                    results = rank_all_resumes(selected_jd)

                if results:
                    st.markdown(f"### Results for: `{selected_jd}`")
                    for i, r in enumerate(results):
                        rank_color = "#68d391" if r["match_score"] >= 30 else "#f6ad55" if r["match_score"] >= 15 else "#fc4e4e"
                        shortlist_badge = '✅ Shortlisted' if r["shortlist"] else '⏳ Review'
                        st.markdown(
                            f'<div style="background:rgba(15,25,50,0.8);border:1px solid rgba(99,179,237,0.2);'
                            f'border-left:4px solid {rank_color};border-radius:12px;padding:16px;margin:8px 0;">'
                            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                            f'<span style="font-weight:700;color:#e2e8f0;font-size:1rem;">#{i+1} {r["resume_file"]}</span>'
                            f'<span style="font-size:1.4rem;font-weight:800;color:{rank_color};">{r["match_score"]:.1f}%</span>'
                            f'</div>'
                            f'<div style="color:#a0aec0;font-size:0.82rem;margin:6px 0;">{r["recommendation"]}</div>'
                            f'<div style="color:#68d391;font-size:0.8rem;">{shortlist_badge}</div>'
                            f'<div style="margin-top:8px;font-size:0.75rem;color:#63b3ed;">'
                            f'🔑 Matched: {", ".join(r["matched_keywords"][:8])}</div>'
                            f'</div>', unsafe_allow_html=True
                        )
                else:
                    st.info("No resume files found. Run `python setup.py` first.")

        # Show available resumes
        resume_files = list(RESUMES_DIR.glob("*.txt"))
        if resume_files:
            section("Candidate Resumes Preview")
            sel_resume = st.selectbox("View Resume", [f.name for f in resume_files])
            rf = RESUMES_DIR / sel_resume
            st.text_area("Resume Content", rf.read_text(encoding="utf-8"), height=350)

    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            jd_files_custom = [f.name for f in JD_DIR.glob("*.txt")]
            selected_jd_c = st.selectbox("Job Description", jd_files_custom if jd_files_custom else ["ml_engineer.txt"])
            resume_text = st.text_area("Paste Resume Text", height=300,
                                        placeholder="Paste the candidate's resume here …")
        with col_b:
            if st.button("Analyse Match 🚀", type="primary") and resume_text.strip():
                with st.spinner("Computing match score …"):
                    result = match_resume_to_jd(resume_text, selected_jd_c)

                score = result["match_score"]
                color = "#68d391" if score >= 30 else "#f6ad55" if score >= 15 else "#fc4e4e"
                st.markdown(f"""
                <div style="background:rgba(15,25,50,0.9);border:1px solid {color}44;
                     border-radius:16px;padding:24px;text-align:center;margin:16px 0;">
                    <div style="font-size:3rem;font-weight:800;color:{color};">{score:.1f}%</div>
                    <div style="color:#a0aec0;margin:8px 0;">Match Score</div>
                    <div style="font-size:1rem;color:{color};font-weight:600;">{result['recommendation']}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"**Matched Keywords:** {', '.join(result['matched_keywords'][:10])}")
                badge = "✅ **SHORTLISTED**" if result["shortlist"] else "⏳ **MANUAL REVIEW REQUIRED**"
                st.markdown(badge)


# ═══════════════════════════════════════════════════════════════════════════════
#   PAGE 8 — EDA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════

elif current == "eda":
    st.markdown('<h1 style="color:#90cdf4;font-size:1.8rem;font-weight:800;">📊 EDA Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#718096;">Interactive exploratory data analysis across all HR datasets.</p>', unsafe_allow_html=True)

    try:
        data = load_all_data()
        att  = get_attrition_scored()
        perf = data["performance_kpi"]
        hr   = data["hr_performance"]

        dataset_choice = st.selectbox(
            "Select Dataset",
            ["IBM HR Attrition", "Performance KPI", "HR Performance & Engagement"]
        )

        if dataset_choice == "IBM HR Attrition":
            df_eda = att.copy()
        elif dataset_choice == "Performance KPI":
            df_eda = perf.copy()
        else:
            df_eda = hr.copy()

        # Dataset overview
        section("Dataset Overview")
        c1,c2,c3,c4 = st.columns(4)
        with c1: metric_card(f"{len(df_eda):,}", "Rows", color="#63b3ed")
        with c2: metric_card(str(len(df_eda.columns)), "Columns", color="#68d391")
        with c3: metric_card(str(df_eda.isnull().sum().sum()), "Missing Values", color="#f6ad55")
        with c4: metric_card(str(df_eda.select_dtypes("number").shape[1]), "Numeric Cols", color="#b794f4")

        with st.expander("📋 Show Dataset Sample (top 50 rows)"):
            st.dataframe(df_eda.head(50), use_container_width=True)

        with st.expander("📈 Descriptive Statistics"):
            st.dataframe(df_eda.describe().T.style.background_gradient(cmap="Blues"), use_container_width=True)

        # ── IBM HR Attrition EDA ─────────────────────────────────────────────
        if dataset_choice == "IBM HR Attrition":
            section("Attrition Analysis (IBM HR — 1,470 Employees)")

            col_a, col_b = st.columns(2)
            with col_a:
                # Attrition by Department
                if "Department" in df_eda.columns and "Attrition" in df_eda.columns:
                    dept_att = df_eda.groupby(["Department","Attrition"]).size().reset_index(name="Count")
                    fig = px.bar(dept_att, x="Department", y="Count", color="Attrition",
                                 barmode="group", color_discrete_map={"Yes":"#fc4e4e","No":"#68d391"},
                                 title="Attrition by Department")
                    styled_chart(fig, 320)

            with col_b:
                # Attrition by Gender
                if "Gender" in df_eda.columns and "Attrition" in df_eda.columns:
                    gen = df_eda.groupby(["Gender","Attrition"]).size().reset_index(name="Count")
                    fig2 = px.bar(gen, x="Gender", y="Count", color="Attrition",
                                  barmode="group", color_discrete_map={"Yes":"#fc4e4e","No":"#68d391"},
                                  title="Attrition by Gender")
                    styled_chart(fig2, 320)

            col_c, col_d = st.columns(2)
            with col_c:
                # Age distribution
                if "Age" in df_eda.columns:
                    fig3 = px.histogram(df_eda, x="Age", color="Attrition" if "Attrition" in df_eda.columns else None,
                                        nbins=20, color_discrete_map={"Yes":"#fc4e4e","No":"#63b3ed"},
                                        title="Age Distribution by Attrition", barmode="overlay", opacity=0.75)
                    styled_chart(fig3, 320)

            with col_d:
                # Monthly income by job role
                if "JobRole" in df_eda.columns and "MonthlyIncome" in df_eda.columns:
                    fig4 = px.box(df_eda, x="JobRole", y="MonthlyIncome", color="Attrition" if "Attrition" in df_eda.columns else None,
                                  color_discrete_map={"Yes":"#fc4e4e","No":"#63b3ed"},
                                  title="Monthly Income by Job Role")
                    fig4.update_xaxes(tickangle=45)
                    styled_chart(fig4, 360)

            col_e, col_f2 = st.columns(2)
            with col_e:
                # Years at company
                if "YearsAtCompany" in df_eda.columns:
                    fig5 = px.histogram(df_eda, x="YearsAtCompany", color="Attrition" if "Attrition" in df_eda.columns else None,
                                        nbins=20, title="Years at Company vs Attrition",
                                        color_discrete_map={"Yes":"#fc4e4e","No":"#63b3ed"},
                                        barmode="overlay", opacity=0.75)
                    styled_chart(fig5, 300)

            with col_f2:
                # Marital status
                if "MaritalStatus" in df_eda.columns and "Attrition" in df_eda.columns:
                    ms = df_eda.groupby(["MaritalStatus","Attrition"]).size().reset_index(name="Count")
                    fig6 = px.bar(ms, x="MaritalStatus", y="Count", color="Attrition",
                                  barmode="group", color_discrete_map={"Yes":"#fc4e4e","No":"#68d391"},
                                  title="Attrition by Marital Status")
                    styled_chart(fig6, 300)

            # Correlation heatmap
            section("Correlation Heatmap (Numeric Features)")
            num_cols = df_eda.select_dtypes(include="number").columns.tolist()[:18]
            corr = df_eda[num_cols].corr()
            fig7 = px.imshow(corr, color_continuous_scale="RdBu_r", aspect="auto",
                             title="Feature Correlation Matrix", text_auto=".2f",
                             zmin=-1, zmax=1)
            fig7.update_traces(textfont_size=8)
            styled_chart(fig7, 550)

        # ── Performance KPI EDA ──────────────────────────────────────────────
        elif dataset_choice == "Performance KPI":
            section("Performance KPI Analysis")

            col_a, col_b = st.columns(2)
            with col_a:
                if "Performance Score" in df_eda.columns and "Department" in df_eda.columns:
                    fig = px.violin(df_eda, x="Department", y="Performance Score",
                                   color="Department", color_discrete_sequence=COLOR_SEQ,
                                   title="Performance Score Distribution by Dept", box=True)
                    fig.update_xaxes(tickangle=30)
                    styled_chart(fig, 360)

            with col_b:
                if "KPI Score" in df_eda.columns and "Promotion Eligibility" in df_eda.columns:
                    fig2 = px.histogram(df_eda, x="KPI Score", color="Promotion Eligibility",
                                        nbins=25, barmode="overlay", opacity=0.75,
                                        color_discrete_map={"Yes":"#68d391","No":"#fc4e4e"},
                                        title="KPI Score by Promotion Eligibility")
                    styled_chart(fig2, 360)

            col_c, col_d = st.columns(2)
            with col_c:
                if "Work Hours Logged" in df_eda.columns and "KPI Score" in df_eda.columns:
                    fig3 = px.scatter(df_eda.sample(min(400,len(df_eda))),
                                      x="Work Hours Logged", y="KPI Score",
                                      color="Department" if "Department" in df_eda.columns else None,
                                      color_discrete_sequence=COLOR_SEQ, opacity=0.7,
                                      title="Work Hours vs KPI Score")
                    styled_chart(fig3, 320)
            with col_d:
                if "Peer Rating" in df_eda.columns and "Performance Score" in df_eda.columns:
                    fig4 = px.scatter(df_eda, x="Peer Rating", y="Performance Score",
                                      trendline="ols", opacity=0.6,
                                      color_discrete_sequence=["#b794f4"],
                                      title="Peer Rating vs Performance Score")
                    styled_chart(fig4, 320)

        # ── HR Performance & Engagement EDA ──────────────────────────────────
        else:
            section("HR Performance & Engagement Analysis")

            col_a, col_b = st.columns(2)
            with col_a:
                eng_cols = [c for c in ["Engagement Score","Satisfaction Score","Work-Life Balance Score"]
                            if c in df_eda.columns]
                if eng_cols:
                    melted = df_eda[eng_cols].melt(var_name="Metric", value_name="Score").dropna()
                    fig = px.histogram(melted, x="Score", color="Metric",
                                       nbins=15, barmode="overlay", opacity=0.7,
                                       color_discrete_sequence=COLOR_SEQ,
                                       title="Engagement / Satisfaction Score Distributions")
                    styled_chart(fig, 340)

            with col_b:
                if "Training Outcome" in df_eda.columns:
                    outcome_cnt = df_eda["Training Outcome"].value_counts().reset_index()
                    outcome_cnt.columns = ["Outcome","Count"]
                    fig2 = px.pie(outcome_cnt, names="Outcome", values="Count",
                                  color_discrete_sequence=COLOR_SEQ, hole=0.45,
                                  title="Training Outcome Distribution")
                    styled_chart(fig2, 340)

            col_c, col_d = st.columns(2)
            with col_c:
                if "Training Type" in df_eda.columns and "Training Duration(Days)" in df_eda.columns:
                    fig3 = px.box(df_eda, x="Training Type", y="Training Duration(Days)",
                                  color="Training Type", color_discrete_sequence=COLOR_SEQ,
                                  title="Training Duration by Type")
                    styled_chart(fig3, 320)
            with col_d:
                if "DepartmentType" in df_eda.columns and "Training Cost" in df_eda.columns:
                    tc = df_eda.groupby("DepartmentType")["Training Cost"].sum().reset_index()
                    fig4 = px.bar(tc.sort_values("Training Cost",ascending=False),
                                  x="DepartmentType", y="Training Cost",
                                  color="Training Cost", color_continuous_scale="Blues",
                                  title="Training Cost by Department")
                    styled_chart(fig4, 320)

        # ── Universal: custom axis chart ──────────────────────────────────────
        section("🔧 Custom Chart Builder")
        num_cols_all = df_eda.select_dtypes(include="number").columns.tolist()
        cat_cols_all = df_eda.select_dtypes(include="object").columns.tolist()
        col_x, col_y, col_c2, col_type = st.columns(4)
        with col_x:    x_ax = st.selectbox("X Axis", num_cols_all + cat_cols_all, key="cx")
        with col_y:    y_ax = st.selectbox("Y Axis", num_cols_all, key="cy",
                                            index=min(1,len(num_cols_all)-1))
        with col_c2:   c_ax = st.selectbox("Color By", ["None"] + cat_cols_all, key="cc")
        with col_type: chart_t = st.selectbox("Chart Type", ["Scatter","Bar","Box","Histogram","Line"], key="ct")

        color_param = None if c_ax == "None" else c_ax
        try:
            if chart_t == "Scatter":
                fig_c = px.scatter(df_eda.sample(min(500,len(df_eda))), x=x_ax, y=y_ax,
                                   color=color_param, color_discrete_sequence=COLOR_SEQ, opacity=0.7)
            elif chart_t == "Bar":
                if color_param:
                    gb = df_eda.groupby([x_ax, color_param])[y_ax].mean().reset_index()
                    fig_c = px.bar(gb, x=x_ax, y=y_ax, color=color_param,
                                   barmode="group", color_discrete_sequence=COLOR_SEQ)
                else:
                    gb = df_eda.groupby(x_ax)[y_ax].mean().reset_index()
                    fig_c = px.bar(gb, x=x_ax, y=y_ax, color_discrete_sequence=["#63b3ed"])
            elif chart_t == "Box":
                fig_c = px.box(df_eda, x=x_ax, y=y_ax, color=color_param,
                               color_discrete_sequence=COLOR_SEQ)
            elif chart_t == "Histogram":
                fig_c = px.histogram(df_eda, x=x_ax, color=color_param, nbins=25,
                                     barmode="overlay", opacity=0.75,
                                     color_discrete_sequence=COLOR_SEQ)
            else:
                gb2 = df_eda.groupby(x_ax)[y_ax].mean().reset_index()
                fig_c = px.line(gb2, x=x_ax, y=y_ax, markers=True,
                                color_discrete_sequence=["#63b3ed"])
            styled_chart(fig_c, 380)
        except Exception as ex:
            st.warning(f"Chart error: {ex}")

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Run `python setup.py` first.")
