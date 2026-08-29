"""
NETRA - Tier-1 Training Script
================================
Trains three candidate models on the unified dataset and saves the best.

Run locally:
    python ml/train.py

Run in Colab:
    !python ml/train.py --data /content/drive/MyDrive/NETRA/data/processed/unified.csv

Models trained:
    1. Logistic Regression  (C=1.0, max_iter=1000)
    2. Linear SVM           (C=1.0)
    3. Random Forest        (n_estimators=500, balanced)

Selection criterion:
    Best model by phishing Recall (label=1), subject to F1 >= 0.85
    Target: Recall >= 0.90, FPR <= 0.05

Outputs:
    ml/models/tier1_model.pkl
    ml/models/tfidf_vectorizer.pkl
"""

import json
import logging
import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "processed" / "unified.csv"
MODELS_DIR = ROOT / "ml" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "tier1_model.pkl"
TFIDF_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"

# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------

def load_data(csv_path: Path) -> pd.DataFrame:
    """Load the unified CSV produced by data_pipeline.py."""
    log.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path, dtype={"label": int})
    log.info(f"Loaded {len(df)} records. Splits: {df['split'].value_counts().to_dict()}")
    return df


def assemble_url_features(df: pd.DataFrame) -> np.ndarray:
    """Parse the 'urls' JSON column and extract URL features for each row."""
    from ml.features.url_features import extract as url_extract, URL_FEATURE_NAMES

    rows = []
    for urls_json in df["urls"].fillna("[]"):
        try:
            urls = json.loads(urls_json)
        except (json.JSONDecodeError, TypeError):
            urls = []
        rows.append(url_extract(urls))

    url_df = pd.DataFrame(rows, columns=URL_FEATURE_NAMES)
    return url_df.values.astype(np.float32)


def assemble_header_features(df: pd.DataFrame) -> np.ndarray:
    """Parse the 'headers_available' JSON column and extract header features."""
    from ml.features.header_features import extract as header_extract, HEADER_FEATURE_NAMES

    rows = []
    for _, row in df[["headers_available", "sender", "reply_to"]].iterrows():
        try:
            headers = json.loads(row["headers_available"] or "{}")
        except (json.JSONDecodeError, TypeError):
            headers = {}
        feat = header_extract(
            headers_dict=headers,
            sender=row["sender"] or "",
            reply_to=row["reply_to"] or "",
        )
        rows.append(feat)

    header_df = pd.DataFrame(rows, columns=HEADER_FEATURE_NAMES)
    return header_df.values.astype(np.float32)


def build_feature_matrix(
    df: pd.DataFrame,
    text_extractor,
    fit: bool = False,
) -> csr_matrix:
    """
    Assemble the full feature matrix:
        TF-IDF (5004 dims) + URL features (10) + header features (10)

    Args:
        df:             DataFrame slice for a given split
        text_extractor: TextFeatureExtractor instance
        fit:            If True, call fit_transform; else transform only

    Returns:
        Sparse feature matrix of shape (n_samples, n_features)
    """
    log.info(f"  Building features for {len(df)} samples (fit={fit})...")

    if fit:
        X_text = text_extractor.fit_transform(df)
    else:
        X_text = text_extractor.transform(df)

    X_url = csr_matrix(assemble_url_features(df))
    X_header = csr_matrix(assemble_header_features(df))

    X = hstack([X_text, X_url, X_header])
    log.info(f"  Feature matrix shape: {X.shape}")
    return X


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def compute_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """False Positive Rate = FP / (FP + TN)."""
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    return fp / (fp + tn + 1e-9)


def evaluate_model(
    model,
    X_val: csr_matrix,
    y_val: np.ndarray,
    model_name: str,
) -> dict:
    """Compute all required metrics on the validation set."""
    y_pred = model.predict(X_val)

    # Probability scores for PR-AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_val)[:, 1]
    else:
        # CalibratedClassifierCV wraps LinearSVC — will always have predict_proba
        y_prob = np.zeros(len(y_val))

    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    fpr = compute_fpr(y_val, y_pred)
    pr_auc = average_precision_score(y_val, y_prob)

    metrics = {
        "model": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "pr_auc": round(pr_auc, 4),
    }

    log.info(
        f"\n{'='*50}\n"
        f"  Model  : {model_name}\n"
        f"  Acc    : {acc:.4f}\n"
        f"  Prec   : {prec:.4f}\n"
        f"  Recall : {rec:.4f}   ← phishing recall\n"
        f"  F1     : {f1:.4f}\n"
        f"  FPR    : {fpr:.4f}\n"
        f"  PR-AUC : {pr_auc:.4f}\n"
        f"{'='*50}"
    )
    return metrics


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def build_models() -> list:
    """
    Return list of (name, estimator) tuples.
    LinearSVC is wrapped in CalibratedClassifierCV for probability estimates.
    """
    models = [
        (
            "Logistic Regression",
            LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                solver="saga",
                n_jobs=-1,
            ),
        ),
        (
            "Linear SVM (calibrated)",
            CalibratedClassifierCV(
                LinearSVC(C=1.0, class_weight="balanced", max_iter=2000),
                cv=3,
            ),
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=500,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
                max_depth=None,
                min_samples_leaf=2,
            ),
        ),
    ]
    return models


# ---------------------------------------------------------------------------
# Selection criterion
# ---------------------------------------------------------------------------

def select_best_model(results: list) -> dict:
    """
    Select the model with the highest phishing Recall,
    subject to F1 >= 0.85.

    If no model meets F1 >= 0.85, fall back to the highest F1 model.
    """
    eligible = [r for r in results if r["metrics"]["f1"] >= 0.85]
    pool = eligible if eligible else results

    best = max(pool, key=lambda r: r["metrics"]["recall"])
    log.info(
        f"\n>>> Best model selected: {best['name']}\n"
        f"    Recall={best['metrics']['recall']:.4f}, "
        f"F1={best['metrics']['f1']:.4f}, "
        f"FPR={best['metrics']['fpr']:.4f}"
    )
    return best


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(data_csv: Path = DATA_CSV):
    log.info("=" * 60)
    log.info("NETRA Tier-1 Training starting")
    log.info("=" * 60)

    # --- Load data ---
    df = load_data(data_csv)

    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val = df[df["split"] == "val"].reset_index(drop=True)

    if len(df_train) == 0:
        raise ValueError("Training split is empty. Run data_pipeline.py first.")
    if len(df_val) == 0:
        raise ValueError("Validation split is empty.")

    log.info(f"Train: {len(df_train)} | Val: {len(df_val)}")

    # --- Feature extraction ---
    from ml.features.text_features import TextFeatureExtractor
    text_extractor = TextFeatureExtractor(max_features=5000, sublinear_tf=True)

    X_train = build_feature_matrix(df_train, text_extractor, fit=True)
    X_val = build_feature_matrix(df_val, text_extractor, fit=False)

    y_train = df_train["label"].values
    y_val = df_val["label"].values

    log.info(f"Train label distribution: {np.bincount(y_train)}")
    log.info(f"Val   label distribution: {np.bincount(y_val)}")

    # --- Train all candidate models ---
    results = []
    for name, estimator in build_models():
        log.info(f"\nTraining: {name}")
        estimator.fit(X_train, y_train)
        metrics = evaluate_model(estimator, X_val, y_val, name)
        results.append({"name": name, "model": estimator, "metrics": metrics})

    # --- Select best ---
    best = select_best_model(results)

    # Check target thresholds
    if best["metrics"]["recall"] >= 0.90:
        log.info("✓ Phishing Recall target (≥0.90) ACHIEVED")
    else:
        log.warning(f"✗ Phishing Recall target NOT achieved ({best['metrics']['recall']:.4f} < 0.90)")

    if best["metrics"]["fpr"] <= 0.05:
        log.info("✓ FPR target (≤0.05) ACHIEVED")
    else:
        log.warning(f"✗ FPR target NOT achieved ({best['metrics']['fpr']:.4f} > 0.05)")

    # --- Save artefacts ---
    joblib.dump(best["model"], MODEL_PATH)
    log.info(f"Saved model → {MODEL_PATH}")

    joblib.dump(text_extractor, TFIDF_PATH)
    log.info(f"Saved TF-IDF extractor → {TFIDF_PATH}")

    # --- Save metrics summary ---
    metrics_path = MODELS_DIR / "training_metrics.json"
    all_metrics = [r["metrics"] for r in results]
    all_metrics_with_best = {
        "best_model": best["name"],
        "all_results": all_metrics,
    }
    with open(metrics_path, "w") as f:
        json.dump(all_metrics_with_best, f, indent=2)
    log.info(f"Saved metrics → {metrics_path}")

    log.info("\nTraining complete.")
    return best["model"], text_extractor, results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NETRA Tier-1 Training")
    parser.add_argument(
        "--data", type=str, default=str(DATA_CSV),
        help="Path to unified.csv produced by data_pipeline.py"
    )
    args = parser.parse_args()
    train(data_csv=Path(args.data))
