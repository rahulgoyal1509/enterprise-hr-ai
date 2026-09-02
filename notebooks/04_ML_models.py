"""
Notebook 04 — ML Model Training & Evaluation
Full end-to-end: feature engineering → training → evaluation → SHAP explainability
"""

# %% [markdown]
# # 🤖 ML Model Training & Evaluation
# **Models**: RandomForest Attrition Risk + GradientBoosting Performance + KMeans Engagement
# **Goal**: Train production models, evaluate performance, analyse feature importance.

# %% Imports
import sys, warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve,
                             mean_absolute_error, r2_score)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
warnings.filterwarnings("ignore")
ROOT = Path("..").resolve()
sys.path.insert(0, str(ROOT))
plt.style.use("dark_background")
FIGSIZE = (14, 6)

# %% Load processed data
from app.backend.services.data_pipeline import run_pipeline
data = run_pipeline()
att_df = data["attrition"]
kpi_df = data["performance_kpi"]
hr_df  = data["hr_performance"]
print("Datasets loaded ✔")
print(f"Attrition: {att_df.shape}, KPI: {kpi_df.shape}, HR: {hr_df.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — ATTRITION RISK MODEL
# ═══════════════════════════════════════════════════════════════════════════════

# %% Feature engineering
from config.settings import ATTRITION_FEATURES
feature_cols = [c for c in ATTRITION_FEATURES if c in att_df.columns]
print(f"Using {len(feature_cols)} features: {feature_cols}")

X = att_df[feature_cols].fillna(0)
y = att_df["Attrition_enc"]
print(f"\nClass distribution:\n{y.value_counts()}")
print(f"Attrition rate: {y.mean()*100:.1f}%")

# %% Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# %% Train RandomForest
rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5,
                             class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:,1]

# %% Evaluation
print("\n" + "="*50)
print("ATTRITION MODEL EVALUATION")
print("="*50)
print(f"Accuracy:  {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Stay","Leave"]))

# Cross-validation
cv_scores = cross_val_score(rf, X, y, cv=StratifiedKFold(5), scoring="roc_auc")
print(f"5-Fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# %% ROC Curve + Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
axes[0].plot(fpr, tpr, color="#63b3ed", linewidth=2.5, label=f"RF (AUC = {auc:.3f})")
axes[0].plot([0,1],[0,1],"--", color="#718096", linewidth=1)
axes[0].fill_between(fpr, tpr, alpha=0.1, color="#63b3ed")
axes[0].set_title("ROC Curve — Attrition Model", fontsize=13, fontweight="bold")
axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
axes[0].legend(); axes[0].grid(alpha=0.2)

cm = confusion_matrix(y_test, y_pred)
im = axes[1].imshow(cm, cmap="Blues")
axes[1].set_xticks([0,1]); axes[1].set_yticks([0,1])
axes[1].set_xticklabels(["Predicted Stay","Predicted Leave"])
axes[1].set_yticklabels(["Actual Stay","Actual Leave"])
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, str(cm[i,j]), ha="center", va="center",
                     color="white" if cm[i,j] < cm.max()/2 else "black", fontsize=14, fontweight="bold")
axes[1].set_title("Confusion Matrix", fontsize=13, fontweight="bold")
plt.colorbar(im, ax=axes[1])
plt.tight_layout(); plt.show()

# %% Feature Importance
fi = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
top_fi = fi.head(15)
fig, ax = plt.subplots(figsize=(12,7))
ax.barh(top_fi.index[::-1], top_fi.values[::-1],
        color=plt.cm.Blues(np.linspace(0.4,0.9,15)), alpha=0.9)
ax.set_title("Top 15 Feature Importances — Attrition Model", fontsize=13, fontweight="bold")
ax.set_xlabel("Feature Importance Score")
plt.tight_layout(); plt.show()
print("Top 5 attrition drivers:", list(top_fi.head(5).index))

# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — PERFORMANCE PREDICTION MODEL
# ═══════════════════════════════════════════════════════════════════════════════

# %% Features for performance model
perf_features = ["KPI Score","Attendance (%)","Peer Rating","Task Completion (%)","Work Hours Logged","Training Hours"]
perf_features = [c for c in perf_features if c in kpi_df.columns]
target = "Performance Score"

if target in kpi_df.columns and perf_features:
    kpi_clean = kpi_df[perf_features + [target]].dropna()
    Xp = kpi_clean[perf_features]; yp = kpi_clean[target]

    scaler = StandardScaler()
    Xp_scaled = scaler.fit_transform(Xp)
    Xp_train, Xp_test, yp_train, yp_test = train_test_split(Xp_scaled, yp, test_size=0.2, random_state=42)

    gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    gb.fit(Xp_train, yp_train)
    yp_pred = gb.predict(Xp_test)

    print("\n" + "="*50)
    print("PERFORMANCE MODEL EVALUATION")
    print("="*50)
    print(f"MAE:  {mean_absolute_error(yp_test, yp_pred):.3f}")
    print(f"R²:   {r2_score(yp_test, yp_pred):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    axes[0].scatter(yp_test, yp_pred, alpha=0.4, color="#68d391", s=10)
    lims = [min(yp_test.min(), yp_pred.min()), max(yp_test.max(), yp_pred.max())]
    axes[0].plot(lims, lims, "--", color="#fc4e4e", linewidth=1.5, label="Perfect Fit")
    axes[0].set_title("Actual vs Predicted Performance Score", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted"); axes[0].legend()

    fi_gb = pd.Series(gb.feature_importances_, index=perf_features).sort_values(ascending=False)
    axes[1].bar(fi_gb.index, fi_gb.values, color=plt.cm.Greens(np.linspace(0.4,0.9,len(fi_gb))), alpha=0.85)
    axes[1].set_title("Feature Importance — Performance Model", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Importance"); axes[1].tick_params(axis="x", rotation=35)
    plt.tight_layout(); plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — ENGAGEMENT CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════

# %% KMeans clustering
cluster_features = [c for c in ["Engagement Score","Satisfaction Score","Work-Life Balance Score"] if c in hr_df.columns]
if len(cluster_features) >= 2:
    hr_clean = hr_df[cluster_features].dropna()
    scaler2  = StandardScaler()
    Xc = scaler2.fit_transform(hr_clean)

    # Elbow method
    inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xc).inertia_ for k in range(2,9)]
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    axes[0].plot(range(2,9), inertias, marker="o", color="#63b3ed", linewidth=2)
    axes[0].set_title("Elbow Method — Optimal K", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Number of Clusters"); axes[0].set_ylabel("Inertia"); axes[0].grid(alpha=0.2)

    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = km.fit_predict(Xc)
    hr_clean = hr_clean.copy(); hr_clean["Cluster"] = labels
    segment_map = {0:"Disengaged",1:"At Risk",2:"Engaged",3:"Highly Engaged"}
    hr_clean["Segment"] = hr_clean["Cluster"].map(segment_map)

    seg_counts = hr_clean["Segment"].value_counts()
    colors_seg = {"Highly Engaged":"#68d391","Engaged":"#63b3ed","At Risk":"#f6ad55","Disengaged":"#fc4e4e"}
    axes[1].pie(seg_counts.values, labels=seg_counts.index,
                colors=[colors_seg.get(s,"#718096") for s in seg_counts.index],
                autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Engagement Segment Distribution", fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.show()

    print("\nEngagement Segment Sizes:")
    print(seg_counts)
    print("\nCluster Centres (normalised):")
    print(pd.DataFrame(km.cluster_centers_, columns=cluster_features).round(3))

# %% Final Summary
print("\n" + "="*60)
print("MODEL TRAINING COMPLETE — SUMMARY")
print("="*60)
print(f"✔ Attrition Model:     RandomForest  | AUC: {auc:.3f}")
if target in kpi_df.columns and perf_features:
    print(f"✔ Performance Model:   GradientBoost | R²: {r2_score(yp_test, yp_pred):.3f}")
print(f"✔ Engagement Clusters: KMeans k=4")
print("="*60)
print("Run `python setup.py` to save models to disk for production use.")
