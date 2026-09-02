"""
Notebook 03 — EDA: O*NET Skills Dataset
"""

# %% [markdown]
# # 🎯 EDA — O*NET Skills & Occupations
# **Datasets**: occupation_data.csv, essential_skills.csv, software_skills.csv
# Covers: Skill demand, hot technologies, occupation landscape, role-skill mapping.

# %% Imports
import sys, warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path("..").resolve()
sys.path.insert(0, str(ROOT))
plt.style.use("dark_background")
FIGSIZE = (14, 6)

# %% Load
from config.settings import OCCUPATION_CSV, ESSENTIAL_SKILLS_CSV, SOFTWARE_SKILLS_CSV
occ  = pd.read_csv(OCCUPATION_CSV)
es   = pd.read_csv(ESSENTIAL_SKILLS_CSV)
sw   = pd.read_csv(SOFTWARE_SKILLS_CSV)
for df in [occ, es, sw]: df.columns = [c.strip() for c in df.columns]
print(f"Occupations: {len(occ)} | Essential Skills: {len(es)} | Software Skills: {len(sw)}")

# %% ── Occupation Count by SOC Major Group ────────────────────────────────────
occ["MajorGroup"] = occ["O*NET-SOC Code"].str[:2]
major_counts = occ["MajorGroup"].value_counts().head(15)
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.bar(major_counts.index, major_counts.values,
       color=plt.cm.viridis(np.linspace(0,0.9,len(major_counts))), alpha=0.85)
ax.set_title("O*NET Occupations by Major SOC Group (Top 15)", fontsize=13, fontweight="bold")
ax.set_xlabel("Major SOC Group Code"); ax.set_ylabel("Number of Occupations")
plt.tight_layout(); plt.show()

# %% ── Top Essential Skills by Importance ─────────────────────────────────────
if "Scale ID" in es.columns and "Data Value" in es.columns and "Element Name" in es.columns:
    es_im = es[es["Scale ID"]=="IM"].copy()
    es_im["Data Value"] = pd.to_numeric(es_im["Data Value"], errors="coerce")
    top_skills = es_im.groupby("Element Name")["Data Value"].mean().sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(14,8))
    bars = ax.barh(top_skills.index[::-1], top_skills.values[::-1],
                   color=plt.cm.Blues(np.linspace(0.4,0.9,20)), alpha=0.9)
    ax.set_title("Top 20 Essential Skills by Average Importance (All Occupations)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Average Importance Score (0-5)")
    ax.axvline(top_skills.mean(), color="#fc4e4e", linestyle="--", linewidth=1.5, label=f"Mean: {top_skills.mean():.2f}")
    ax.legend()
    plt.tight_layout(); plt.show()

# %% ── Hot Technologies ────────────────────────────────────────────────────────
if "Hot Technology" in sw.columns and "Element Name" in sw.columns:
    hot = sw[sw["Hot Technology"]=="Y"]
    hot_counts = hot["Element Name"].value_counts().head(20)
    fig, ax = plt.subplots(figsize=(14,8))
    ax.barh(hot_counts.index[::-1], hot_counts.values[::-1],
            color=plt.cm.hot(np.linspace(0.3,0.8,20)), alpha=0.85)
    ax.set_title("Top 20 Hot Technologies Across All Roles (O*NET)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Roles Requiring This Technology")
    plt.tight_layout(); plt.show()

# %% ── In-Demand Technologies ─────────────────────────────────────────────────
if "In Demand" in sw.columns:
    indem = sw[sw["In Demand"]=="Y"]["Element Name"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(range(len(indem)), indem.values,
           color=plt.cm.cool(np.linspace(0.2,0.9,len(indem))), alpha=0.85)
    ax.set_xticks(range(len(indem)))
    ax.set_xticklabels(indem.index, rotation=45, ha="right", fontsize=9)
    ax.set_title("Top 15 In-Demand Technologies (O*NET)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Roles")
    plt.tight_layout(); plt.show()

# %% ── Skills for Target HR Roles ─────────────────────────────────────────────
TARGET_ROLES = {
    "Data Scientist": "15-2051.00",
    "Software Engineer": "15-1252.00",
    "HR Manager": "11-3121.00",
    "Research Scientist": "19-1042.00",
}
if "O*NET-SOC Code" in es.columns and "Element Name" in es.columns and "Data Value" in es.columns:
    es_im2 = es[es["Scale ID"]=="IM"].copy() if "Scale ID" in es.columns else es.copy()
    es_im2["Data Value"] = pd.to_numeric(es_im2["Data Value"], errors="coerce")

    fig, axes = plt.subplots(2, 2, figsize=(16,12))
    for ax, (role, code) in zip(axes.flat, TARGET_ROLES.items()):
        role_skills = es_im2[es_im2["O*NET-SOC Code"]==code]
        if not role_skills.empty:
            top = role_skills.nlargest(10, "Data Value")[["Element Name","Data Value"]]
            ax.barh(top["Element Name"].values[::-1], top["Data Value"].values[::-1],
                    color="#63b3ed", alpha=0.85)
            ax.set_title(f"Top Skills: {role}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Importance Score")
        else:
            ax.text(0.5, 0.5, f"No data for\n{role}\n({code})",
                    ha="center", va="center", transform=ax.transAxes, color="#718096")
            ax.set_title(role, fontsize=11)
    plt.suptitle("Essential Skill Requirements by Target HR Role (O*NET)", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.show()

# %% ── Summary ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("KEY INSIGHTS — O*NET SKILLS LANDSCAPE")
print("=" * 60)
print(f"Total unique occupations: {len(occ)}")
print(f"Total essential skills records: {len(es)}")
print(f"Total software skill records: {len(sw)}")
if "Hot Technology" in sw.columns:
    print(f"Hot Technology entries: {(sw['Hot Technology']=='Y').sum()}")
if "Element Name" in sw.columns and "Hot Technology" in sw.columns:
    top_hot = sw[sw["Hot Technology"]=="Y"]["Element Name"].value_counts().head(3)
    print(f"Top 3 hot technologies: {list(top_hot.index)}")
print("=" * 60)
