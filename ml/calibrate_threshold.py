# -*- coding: utf-8 -*-
"""
NETRA - Dynamic Threshold Finder
Loads model + val split, sweeps PR curve, finds optimal threshold.
Run: python ml/calibrate_threshold.py
Outputs: ml/models/threshold_config.json
"""

import sys
import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import precision_recall_curve, confusion_matrix

from ml.features.url_features import extract as url_extract, URL_FEATURE_NAMES
from ml.features.header_features import extract as header_extract, HEADER_FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR  = ROOT / "ml" / "models"
MODEL_PATH  = MODELS_DIR / "tier1_model.pkl"
TFIDF_PATH  = MODELS_DIR / "tfidf_vectorizer.pkl"
DATA_PATH   = ROOT / "data" / "processed" / "unified.csv"
OUTPUT_PATH = MODELS_DIR / "threshold_config.json"


def build_feature_matrix(df, text_extractor):
    log.info(f"  Building text features for {len(df):,} records...")
    X_text = text_extractor.transform(df)

    log.info("  Building URL features...")
    url_rows = []
    for urls_json in df["urls"].fillna("[]"):
        try:
            urls = json.loads(urls_json)
        except Exception:
            urls = []
        url_rows.append([url_extract(urls)[k] for k in URL_FEATURE_NAMES])
    X_url = csr_matrix(np.array(url_rows, dtype=np.float32))

    log.info("  Building header features...")
    header_rows = []
    for _, row in df[["headers_available", "sender", "reply_to"]].iterrows():
        try:
            hdrs = json.loads(row["headers_available"] or "{}")
        except Exception:
            hdrs = {}
        feat = header_extract(
            headers_dict=hdrs,
            sender=str(row["sender"] or ""),
            reply_to=str(row["reply_to"] or ""),
        )
        header_rows.append([feat[k] for k in HEADER_FEATURE_NAMES])
    X_header = csr_matrix(np.array(header_rows, dtype=np.float32))

    return hstack([X_text, X_url, X_header])


def sweep_thresholds(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    rows = []
    for i, thresh in enumerate(thresholds):
        p = precision[i]
        r = recall[i]
        f1 = 2 * p * r / (p + r + 1e-9)
        y_pred = (y_prob >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn + 1e-9)
        rows.append({
            "threshold": round(float(thresh), 4),
            "precision": round(float(p), 4),
            "recall":    round(float(r), 4),
            "f1":        round(float(f1), 4),
            "fpr":       round(float(fpr), 4),
        })
    return pd.DataFrame(rows)


def run():
    log.info("=" * 60)
    log.info("NETRA Threshold Calibration")
    log.info("=" * 60)

    model          = joblib.load(MODEL_PATH)
    text_extractor = joblib.load(TFIDF_PATH)

    log.info(f"Loading val split from: {DATA_PATH}")
    df     = pd.read_csv(DATA_PATH, dtype=str)
    df_val = df[df["split"] == "val"].reset_index(drop=True)
    df_val["label"] = df_val["label"].astype(int)
    log.info(f"Val split: {len(df_val):,} | label dist: {df_val['label'].value_counts().to_dict()}")

    X_val  = build_feature_matrix(df_val, text_extractor)
    y_val  = df_val["label"].values
    y_prob = model.predict_proba(X_val)[:, 1]

    log.info(f"Score stats -- min:{y_prob.min():.4f} max:{y_prob.max():.4f} mean:{y_prob.mean():.4f} median:{np.median(y_prob):.4f}")

    df_sweep = sweep_thresholds(y_val, y_prob)

    best_f1_idx = df_sweep["f1"].idxmax()
    best_f1_row = df_sweep.loc[best_f1_idx]

    recall_90_rows = df_sweep[df_sweep["recall"] >= 0.90]
    recall_90_row  = recall_90_rows.loc[recall_90_rows["f1"].idxmax()] if not recall_90_rows.empty else None

    print("\n" + "=" * 70)
    print(f"{'threshold':>10} | {'precision':>10} | {'recall':>8} | {'f1':>8} | {'fpr':>8}")
    print("-" * 70)
    step = max(1, len(df_sweep) // 30)
    for _, row in df_sweep.iloc[::step].iterrows():
        marker = ""
        if abs(row["threshold"] - best_f1_row["threshold"]) < 0.001:
            marker = " <<< BEST F1"
        if recall_90_row is not None and abs(row["threshold"] - recall_90_row["threshold"]) < 0.001:
            marker = " <<< RECALL>=0.90"
        print(f"{row['threshold']:>10.4f} | {row['precision']:>10.4f} | {row['recall']:>8.4f} | {row['f1']:>8.4f} | {row['fpr']:>8.4f}{marker}")
    print("=" * 70)

    print(f"\nBest F1 Threshold: {best_f1_row['threshold']:.4f}")
    print(f"  Precision={best_f1_row['precision']:.4f} Recall={best_f1_row['recall']:.4f} F1={best_f1_row['f1']:.4f} FPR={best_f1_row['fpr']:.4f}")

    if recall_90_row is not None:
        print(f"\nRecall>=0.90 Threshold: {recall_90_row['threshold']:.4f}")
        print(f"  Precision={recall_90_row['precision']:.4f} Recall={recall_90_row['recall']:.4f} F1={recall_90_row['f1']:.4f} FPR={recall_90_row['fpr']:.4f}")

    phishing_threshold = float(best_f1_row["threshold"])
    config = {
        "phishing_threshold":  phishing_threshold,
        "suspicious_lower":    round(phishing_threshold * 0.6, 4),
        "suspicious_upper":    phishing_threshold,
        "method":              "pr_curve_f1",
        "val_precision":       float(best_f1_row["precision"]),
        "val_recall":          float(best_f1_row["recall"]),
        "val_f1":              float(best_f1_row["f1"]),
        "val_fpr":             float(best_f1_row["fpr"]),
        "recall_90_threshold": float(recall_90_row["threshold"]) if recall_90_row is not None else None,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(config, f, indent=2)

    log.info(f"Saved: {OUTPUT_PATH}")
    print(f"\nConfig:\n{json.dumps(config, indent=2)}")
    return config


if __name__ == "__main__":
    run()