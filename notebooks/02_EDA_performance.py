"""
Notebook 02 — EDA: HR Performance & Engagement Dataset
"""

# %% [markdown]
# # 📈 EDA — HR Performance & Engagement Dataset
# **Datasets**: Cleaned_HR_Data_Analysis + Employee_Performance_Dataset
# Covers: Performance scores, KPIs, engagement, satisfaction, training impact.

# %% Imports
import sys
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
warnings.filterwarnings("ignore")
ROOT = Path("..").resolve()
sys.path.insert(0, str(ROOT))
plt.style.use("dark_background")
FIGSIZE = (14, 6)

# %% Load datasets
from config.settings import HR_PERFORMANCE_CSV, PERFORMANCE_DATASET_CSV
hr   = pd.read_csv(HR_PERFORMANCE_CSV)
kpi  = pd.read_csv(PERFORMANCE_DATASET_CSV)
hr.columns  = [c.strip() for c in hr.columns]
kpi.columns = [c.strip() for c in kpi.columns]
print("HR Performance shape:", hr.shape)
print("KPI Dataset shape:   ", kpi.shape)
print("\nHR cols:", list(hr.columns))
print("\nKPI cols:", list(kpi.columns))

# %% ── Performance Score Distribution ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
if "Performance Score" in kpi.columns:
    axes[0].hist(kpi["Performance Score"].dropna(), bins=25, color="#63b3ed", edgecolor="none", alpha=0.85)
    axes[0].axvline(kpi["Performance Score"].mean(), color="#fc4e4e", linestyle="--", linewidth=2, label=f"Mean: {kpi['Performance Score'].mean():.1f}")
    axes[0].set_title("Performance Score Distribution", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Performance Score"); axes[0].legend()

if "KPI Score" in kpi.columns:
    axes[1].hist(kpi["KPI Score"].dropna(), bins=25, color="#68d391", edgecolor="none", alpha=0.85)
    axes[1].axvline(kpi["KPI Score"].mean(), color="#fc4e4e", linestyle="--", linewidth=2, label=f"Mean: {kpi['KPI Score'].mean():.1f}")
    axes[1].set_title("KPI Score Distribution", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("KPI Score"); axes[1].legend()
plt.tight_layout(); plt.show()

# %% ── Performance by Department ──────────────────────────────────────────────
if "Department" in kpi.columns and "Performance Score" in kpi.columns:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    dept_groups = [kpi[kpi["Department"]==d]["Performance Score"].dropna()
                   for d in kpi["Department"].unique()]
    dept_labels = kpi["Department"].unique()
    bp = ax.boxplot(dept_groups, labels=dept_labels, patch_artist=True,
                    medianprops=dict(color="#fc4e4e", linewidth=2))
    colors = ["#63b3ed","#68d391","#f6ad55","#b794f4","#76e4f7","#f687b3"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_title("Performance Score Distribution by Department", fontsize=13, fontweight="bold")
    ax.set_ylabel("Performance Score"); ax.tick_params(axis="x", rotation=20)
    plt.tight_layout(); plt.show()

# %% ── Training Hours Impact ───────────────────────────────────────────────────
if "Training Hours" in kpi.columns and "Performance Score" in kpi.columns:
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

    kpi_clean = kpi[["Training Hours","Performance Score","KPI Score","Promotion Eligibility"]].dropna()
    bins = [0,10,20,30,50,200]; labels = ["0-10h","10-20h","20-30h","30-50h","50h+"]
    kpi_clean = kpi_clean.copy()
    kpi_clean["TrainingBand"] = pd.cut(kpi_clean["Training Hours"], bins=bins, labels=labels)

    band_perf = kpi_clean.groupby("TrainingBand", observed=True)["Performance Score"].mean()
    axes[0].bar(band_perf.index.astype(str), band_perf.values,
                color=["#fc4e4e","#f6ad55","#ecc94b","#68d391","#63b3ed"], alpha=0.85)
    axes[0].set_title("Avg Performance Score by Training Band", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Avg Performance Score"); axes[0].set_xlabel("Training Hours Band")

    axes[1].scatter(kpi_clean["Training Hours"], kpi_clean["Performance Score"],
                    c=kpi_clean["Promotion Eligibility"].map({"Yes":1,"No":0}),
                    cmap="RdYlGn", alpha=0.4, s=12)
    axes[1].set_title("Training Hours vs Performance (Green=Promoted)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Training Hours"); axes[1].set_ylabel("Performance Score")
    plt.tight_layout(); plt.show()

# %% ── Engagement & Satisfaction ──────────────────────────────────────────────
eng_cols = [c for c in ["Engagement Score","Satisfaction Score","Work-Life Balance Score"] if c in hr.columns]
if eng_cols:
    fig, axes = plt.subplots(1, len(eng_cols), figsize=(16,5))
    for ax, col in zip(axes if len(eng_cols)>1 else [axes], eng_cols):
        data = hr[col].dropna()
        ax.hist(data, bins=20, color="#b794f4", edgecolor="none", alpha=0.85)
        ax.axvline(data.mean(), color="#fc4e4e", linestyle="--", linewidth=2, label=f"Mean: {data.mean():.2f}")
        ax.set_title(col, fontsize=12, fontweight="bold"); ax.legend()
    plt.suptitle("HR Engagement & Satisfaction Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.show()

# %% ── Training by Department ─────────────────────────────────────────────────
if "DepartmentType" in hr.columns and "Training Cost" in hr.columns:
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    tc = hr.groupby("DepartmentType")["Training Cost"].sum().sort_values(ascending=False)
    axes[0].bar(tc.index, tc.values, color=plt.cm.cool(np.linspace(0.2,0.9,len(tc))), alpha=0.85)
    axes[0].set_title("Total Training Cost by Department", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Total Cost ($)"); axes[0].tick_params(axis="x", rotation=30)

    if "Training Outcome" in hr.columns:
        outcome = hr["Training Outcome"].value_counts()
        axes[1].pie(outcome.values, labels=outcome.index,
                    colors=["#68d391","#fc4e4e","#f6ad55"], autopct="%1.1f%%",
                    startangle=90, pctdistance=0.85)
        axes[1].set_title("Training Outcome Distribution", fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.show()

# %% ── Promotion Eligibility ───────────────────────────────────────────────────
if "Promotion Eligibility" in kpi.columns and "Department" in kpi.columns:
    promo = kpi.groupby(["Department","Promotion Eligibility"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12,5))
    promo.plot(kind="bar", ax=ax, color={"No":"#fc4e4e","Yes":"#68d391"}, edgecolor="none")
    ax.set_title("Promotion Eligibility by Department", fontsize=13, fontweight="bold")
    ax.set_xlabel("Department"); ax.set_ylabel("Count"); ax.tick_params(axis="x", rotation=30)
    plt.tight_layout(); plt.show()

# %% ── Key Insights ────────────────────────────────────────────────────────────
print("=" * 60)
print("KEY INSIGHTS — PERFORMANCE & ENGAGEMENT")
print("=" * 60)
if "Performance Score" in kpi.columns:
    print(f"Avg Performance Score: {kpi['Performance Score'].mean():.1f}")
    print(f"Std Dev Performance:   {kpi['Performance Score'].std():.1f}")
if "Promotion Eligibility" in kpi.columns:
    print(f"Promotion Eligible:    {(kpi['Promotion Eligibility']=='Yes').mean()*100:.1f}%")
if eng_cols and "Engagement Score" in hr.columns:
    print(f"Avg Engagement Score:  {hr['Engagement Score'].mean():.2f}")
print("=" * 60)
