"""
ML Engine Service
──────────────────
Trains and serves:
  • Attrition Risk Model  (RandomForest Classifier)
  • Performance Prediction Model (GradientBoosting Regressor)
  • Engagement Cluster Model (KMeans)
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             mean_absolute_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    ATTRITION_FEATURES, ATTRITION_MODEL_PATH, ATTRITION_RISK_TIERS,
    ENGAGEMENT_MODEL_PATH, PERFORMANCE_DATASET_CSV,
    PERFORMANCE_FEATURES, PERFORMANCE_MODEL_PATH, SCALER_PATH
)

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")


# ═══════════════════════════════════════════════════════════════════════════════
#   ATTRITION RISK MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def train_attrition_model(df: pd.DataFrame) -> RandomForestClassifier:
    """Train RandomForest attrition risk classifier on IBM HR data."""
    logger.info("Training Attrition Risk Model (RandomForest) …")

    feature_cols = [c for c in ATTRITION_FEATURES if c in df.columns]
    missing = set(ATTRITION_FEATURES) - set(feature_cols)
    if missing:
        logger.warning(f"Missing features: {missing}")

    X = df[feature_cols].fillna(0)
    y = df["Attrition_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.success(f"Attrition model accuracy: {acc:.3f}")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=["Stay", "Leave"]))

    # Save feature importances
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    joblib.dump((model, feature_cols, fi), ATTRITION_MODEL_PATH)
    logger.success(f"Model saved → {ATTRITION_MODEL_PATH}")
    return model


def predict_attrition(employee_row: dict) -> dict:
    """Return risk score + tier for a single employee dict."""
    model, feature_cols, _ = joblib.load(ATTRITION_MODEL_PATH)
    row = pd.DataFrame([employee_row])
    X = row.reindex(columns=feature_cols, fill_value=0)
    prob = model.predict_proba(X)[0][1]

    tier = "Low"
    for (lo, hi), label in ATTRITION_RISK_TIERS.items():
        if lo <= prob < hi:
            tier = label
            break

    return {
        "attrition_probability": round(float(prob), 4),
        "risk_tier": tier,
        "will_leave": bool(prob >= 0.5)
    }


def batch_predict_attrition(df: pd.DataFrame) -> pd.DataFrame:
    """Score every employee in the DataFrame."""
    if not ATTRITION_MODEL_PATH.exists():
        logger.warning(f"{ATTRITION_MODEL_PATH} not found — training model on the fly …")
        train_attrition_model(df)

    model, feature_cols, _ = joblib.load(ATTRITION_MODEL_PATH)
    X = df.reindex(columns=feature_cols, fill_value=0)
    probs = model.predict_proba(X)[:, 1]
    df = df.copy()
    df["AttritionProbability"] = probs

    def _tier(p):
        for (lo, hi), label in ATTRITION_RISK_TIERS.items():
            if lo <= p < hi:
                return label
        return "Low"

    df["RiskTier"] = df["AttritionProbability"].apply(_tier)
    return df


def get_attrition_feature_importance() -> pd.DataFrame:
    """Return feature importance DataFrame."""
    if not ATTRITION_MODEL_PATH.exists():
        from app.backend.services.data_pipeline import get_data
        data = get_data()
        train_attrition_model(data["attrition"])

    _, _, fi = joblib.load(ATTRITION_MODEL_PATH)
    return fi


# ═══════════════════════════════════════════════════════════════════════════════
#   PERFORMANCE PREDICTION MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def train_performance_model(df: pd.DataFrame) -> GradientBoostingRegressor:
    """Train GradientBoosting regressor to predict Performance Score."""
    logger.info("Training Performance Prediction Model (GradientBoosting) …")

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    target = "Performance Score"
    if target not in df.columns:
        logger.error(f"'{target}' not found. Columns: {list(df.columns[:8])}")
        return None

    # Exclude target from features to avoid leakage / 2D y error
    feature_cols = [c for c in PERFORMANCE_FEATURES if c in df.columns and c != target]
    if not feature_cols:
        logger.error("No performance feature columns found.")
        return None

    df_clean = df[feature_cols + [target]].dropna()
    X = df_clean[feature_cols]
    y = df_clean[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump((scaler, feature_cols), SCALER_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.success(f"Performance model — MAE: {mae:.2f}, R²: {r2:.3f}")

    joblib.dump(model, PERFORMANCE_MODEL_PATH)
    logger.success(f"Model saved → {PERFORMANCE_MODEL_PATH}")
    return model


def predict_performance(employee_features: dict) -> dict:
    """Predict performance score for one employee."""
    model = joblib.load(PERFORMANCE_MODEL_PATH)
    scaler, feature_cols = joblib.load(SCALER_PATH)
    row = pd.DataFrame([employee_features])
    X = row.reindex(columns=feature_cols, fill_value=0)
    X_scaled = scaler.transform(X)
    score = float(model.predict(X_scaled)[0])
    score = max(0.0, min(100.0, score))

    return {
        "predicted_performance_score": round(score, 2),
        "performance_tier": (
            "Excellent" if score >= 85 else
            "Good" if score >= 70 else
            "Average" if score >= 55 else "Needs Improvement"
        ),
        "promotion_eligible": score >= 75
    }


# ═══════════════════════════════════════════════════════════════════════════════
#   ENGAGEMENT CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════

def train_engagement_model(df: pd.DataFrame) -> KMeans:
    """KMeans clustering on Engagement + Satisfaction + Work-Life Balance."""
    logger.info("Training Engagement Cluster Model (KMeans) …")

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    cluster_features = [
        c for c in ["Engagement Score", "Satisfaction Score", "Work-Life Balance Score"]
        if c in df.columns
    ]
    df_clean = df[cluster_features].dropna()
    if df_clean.empty or len(df_clean) < 4:
        logger.warning(f"Not enough data for engagement clustering ({len(df_clean)} rows, features: {cluster_features}) — skipping")
        return None

    scaler = StandardScaler()
    X = scaler.fit_transform(df_clean)

    model = KMeans(n_clusters=4, random_state=42, n_init=10)
    model.fit(X)

    labels = model.labels_
    segment_map = {}
    centers = model.cluster_centers_
    avg_scores = centers.mean(axis=1)
    sorted_clusters = np.argsort(avg_scores)
    segment_names = ["Disengaged", "At Risk", "Engaged", "Highly Engaged"]
    for rank, cluster_id in enumerate(sorted_clusters):
        segment_map[int(cluster_id)] = segment_names[rank]

    joblib.dump((model, scaler, cluster_features, segment_map), ENGAGEMENT_MODEL_PATH)
    logger.success(f"Engagement model saved → {ENGAGEMENT_MODEL_PATH}")
    return model


def predict_engagement_segment(scores: dict) -> dict:
    """Predict engagement segment for one employee."""
    model, scaler, features, segment_map = joblib.load(ENGAGEMENT_MODEL_PATH)
    row = pd.DataFrame([[scores.get(f, 3.0) for f in features]], columns=features)
    X_scaled = scaler.transform(row)
    cluster = int(model.predict(X_scaled)[0])
    return {
        "cluster_id": cluster,
        "engagement_segment": segment_map.get(cluster, "Unknown")
    }


# ═══════════════════════════════════════════════════════════════════════════════
#   TRAINING ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def train_all_models(data: dict) -> None:
    """Train all ML models from the pipeline data cache."""
    logger.info("═" * 60)
    logger.info("  Enterprise HR AI  —  ML Training Starting")
    logger.info("═" * 60)

    # Attrition model
    if "attrition" in data and data["attrition"] is not None:
        train_attrition_model(data["attrition"])

    # Performance model — use the KPI dataset (has Performance Score column)
    kpi_df = data.get("performance_kpi")
    if kpi_df is not None and not kpi_df.empty:
        train_performance_model(kpi_df)
    else:
        logger.warning("performance_kpi not found — skipping performance model")

    # Engagement clustering — use hr_performance (has Engagement Score column)
    hr_perf_df = data.get("hr_performance")
    if hr_perf_df is not None and not hr_perf_df.empty:
        train_engagement_model(hr_perf_df)
    else:
        logger.warning("hr_performance not found — skipping engagement clustering")

    logger.success("All models trained ✔")
    logger.info("═" * 60)


if __name__ == "__main__":
    from app.backend.services.data_pipeline import get_data
    data = get_data()
    train_all_models(data)
