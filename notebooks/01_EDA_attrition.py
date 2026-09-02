"""
Notebook 01 — EDA: IBM HR Attrition Dataset
Run: jupyter notebook notebooks/01_EDA_attrition.py
Or open as .py script in VS Code / Jupyter
"""

# %% [markdown]
# # 📊 EDA — IBM HR Attrition Dataset
# **Objective**: Understand the IBM HR Attrition dataset, identify key attrition drivers, and build intuition for the ML model.
#
# **Dataset**: 1,470 employees × 35 features (IBM HR Analytics)

# %% Imports
import sys
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
ROOT = Path("..").resolve()
sys.path.insert(0, str(ROOT))

plt.style.use("dark_background")
sns.set_palette("husl")
FIGSIZE = (14, 6)

# %% Load data
from config.settings import ATTRITION_CSV
df = pd.read_csv(ATTRITION_CSV)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
df.head()

# %% Basic info
print(df.info())
print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])
print("\nAttrition distribution:\n", df["Attrition"].value_counts(normalize=True).round(3))

# %% ── Attrition by Department ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

dept_att = df.groupby(["Department", "Attrition"]).size().unstack(fill_value=0)
dept_att.plot(kind="bar", ax=axes[0], color=["#68d391","#fc4e4e"], edgecolor="none")
axes[0].set_title("Attrition Count by Department", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Department"); axes[0].set_ylabel("Count")
axes[0].legend(title="Attrition"); axes[0].tick_params(axis="x", rotation=15)

dept_rate = df.groupby("Department")["Attrition"].apply(lambda x: (x=="Yes").mean()*100)
axes[1].bar(dept_rate.index, dept_rate.values, color=["#63b3ed","#f6ad55","#b794f4"])
axes[1].set_title("Attrition Rate (%) by Department", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Attrition Rate (%)"); axes[1].tick_params(axis="x", rotation=15)
for i, v in enumerate(dept_rate.values):
    axes[1].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=10)
plt.tight_layout(); plt.show()

# %% ── Age Distribution ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

axes[0].hist(df[df["Attrition"]=="No"]["Age"], bins=20, alpha=0.7, color="#63b3ed", label="Stay", edgecolor="none")
axes[0].hist(df[df["Attrition"]=="Yes"]["Age"], bins=20, alpha=0.7, color="#fc4e4e", label="Leave", edgecolor="none")
axes[0].set_title("Age Distribution by Attrition", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Age"); axes[0].set_ylabel("Count"); axes[0].legend()

df["AgeBand"] = pd.cut(df["Age"], bins=[18,25,35,45,60,100], labels=["18-25","26-35","36-45","46-60","60+"])
age_rate = df.groupby("AgeBand", observed=True)["Attrition"].apply(lambda x: (x=="Yes").mean()*100)
axes[1].plot(age_rate.index.astype(str), age_rate.values, marker="o", color="#f6ad55", linewidth=2.5, markersize=8)
axes[1].fill_between(range(len(age_rate)), age_rate.values, alpha=0.2, color="#f6ad55")
axes[1].set_title("Attrition Rate by Age Band", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Attrition Rate (%)"); axes[1].set_xlabel("Age Band")
plt.tight_layout(); plt.show()

# %% ── Income Analysis ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

df.boxplot(column="MonthlyIncome", by="Attrition", ax=axes[0],
           boxprops=dict(color="#63b3ed"), medianprops=dict(color="#fc4e4e"),
           whiskerprops=dict(color="#63b3ed"), capprops=dict(color="#63b3ed"),
           flierprops=dict(markerfacecolor="#f6ad55", alpha=0.5))
axes[0].set_title("Monthly Income by Attrition", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Attrition"); plt.sca(axes[0]); plt.title("")

role_income = df.groupby("JobRole")["MonthlyIncome"].median().sort_values()
axes[1].barh(role_income.index, role_income.values,
             color=plt.cm.cool(np.linspace(0.2, 0.8, len(role_income))))
axes[1].set_title("Median Income by Job Role", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Median Monthly Income (USD)")
plt.tight_layout(); plt.show()

# %% ── Job Satisfaction ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, col, title in zip(axes,
    ["JobSatisfaction","EnvironmentSatisfaction","WorkLifeBalance"],
    ["Job Satisfaction","Environment Satisfaction","Work-Life Balance"]):
    rate = df.groupby(col)["Attrition"].apply(lambda x: (x=="Yes").mean()*100)
    ax.bar(rate.index, rate.values, color=["#fc4e4e","#f6ad55","#ecc94b","#68d391"])
    ax.set_title(f"Attrition Rate by {title}", fontsize=12, fontweight="bold")
    ax.set_ylabel("Attrition Rate (%)"); ax.set_xlabel(title)
    for i, v in enumerate(rate.values):
        ax.text(i+1, v+0.2, f"{v:.1f}%", ha="center", fontsize=9)
plt.tight_layout(); plt.show()

# %% ── Overtime Impact ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

ot_rate = df.groupby("OverTime")["Attrition"].apply(lambda x: (x=="Yes").mean()*100)
axes[0].bar(["No OT","OT"], ot_rate.values, color=["#68d391","#fc4e4e"], width=0.5)
axes[0].set_title("Attrition Rate: Overtime vs No Overtime", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Attrition Rate (%)")
for i, v in enumerate(ot_rate.values):
    axes[0].text(i, v+0.3, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")

travel_rate = df.groupby("BusinessTravel")["Attrition"].apply(lambda x: (x=="Yes").mean()*100)
axes[1].bar(travel_rate.index, travel_rate.values, color=["#63b3ed","#f6ad55","#fc4e4e"])
axes[1].set_title("Attrition Rate by Business Travel", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Attrition Rate (%)")
plt.tight_layout(); plt.show()

# %% ── Tenure analysis ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

axes[0].hist(df[df["Attrition"]=="No"]["YearsAtCompany"], bins=20, alpha=0.7, color="#63b3ed", label="Stay")
axes[0].hist(df[df["Attrition"]=="Yes"]["YearsAtCompany"], bins=20, alpha=0.7, color="#fc4e4e", label="Leave")
axes[0].set_title("Years at Company by Attrition", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Years at Company"); axes[0].legend()

axes[1].scatter(df["TotalWorkingYears"], df["MonthlyIncome"],
                c=df["Attrition"].map({"Yes":1,"No":0}), cmap="RdYlGn_r",
                alpha=0.5, s=15)
axes[1].set_title("Work Experience vs Income (Red=Left)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Total Working Years"); axes[1].set_ylabel("Monthly Income")
plt.tight_layout(); plt.show()

# %% ── Correlation Heatmap ────────────────────────────────────────────────────
num_df = df.select_dtypes(include="number")
num_df["Attrition_enc"] = (df["Attrition"]=="Yes").astype(int)
corr = num_df.corr()["Attrition_enc"].drop("Attrition_enc").sort_values()

fig, ax = plt.subplots(figsize=(12, 7))
colors = ["#68d391" if v > 0 else "#fc4e4e" for v in corr.values]
ax.barh(corr.index, corr.values, color=colors, alpha=0.85)
ax.axvline(0, color="white", linewidth=0.8)
ax.set_title("Correlation with Attrition (Pearson)", fontsize=14, fontweight="bold")
ax.set_xlabel("Correlation Coefficient")
ax.grid(axis="x", alpha=0.2)
plt.tight_layout(); plt.show()

# %% ── Key Insights ────────────────────────────────────────────────────────────
print("=" * 60)
print("KEY INSIGHTS — IBM HR ATTRITION")
print("=" * 60)
print(f"Overall attrition rate: {(df['Attrition']=='Yes').mean()*100:.1f}%")
print(f"Overtime attrition rate: {ot_rate['Yes']:.1f}% vs {ot_rate['No']:.1f}% (no OT)")
print(f"Avg income (stay): ${df[df['Attrition']=='No']['MonthlyIncome'].mean():.0f}")
print(f"Avg income (leave): ${df[df['Attrition']=='Yes']['MonthlyIncome'].mean():.0f}")
print(f"Most at-risk job role: {df.groupby('JobRole')['Attrition'].apply(lambda x:(x=='Yes').mean()).idxmax()}")
print(f"Most at-risk dept: {df.groupby('Department')['Attrition'].apply(lambda x:(x=='Yes').mean()).idxmax()}")
print("=" * 60)
