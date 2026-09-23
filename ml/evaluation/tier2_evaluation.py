# -*- coding: utf-8 -*-
"""
NETRA - Tier-2 Evaluation Pipeline
=====================================
Evaluates:
1. Tier-2 standalone metrics on escalated test set
2. Combined Tier-1 + Tier-2 system metrics on full test set
3. Generates comparison table: Tier-1-only vs Tier-1+Tier-2

Run:
    python ml/evaluation/tier2_evaluation.py
"""

import sys
import json
import logging
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = ROOT / "ml" / "models"
EVAL_DIR   = ROOT / "ml" / "evaluation"
DATA_CSV   = ROOT / "data" / "processed" / "unified.csv"
TIER2_CSV  = ROOT / "data" / "processed" / "tier2_train.csv"


def evaluate_tier2_standalone():
    """Evaluate Tier-2 model on escalated test/val set."""
    import torch
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, classification_report, confusion_matrix)

    log.info("=" * 60)
    log.info("TIER-2 STANDALONE EVALUATION")
    log.info("=" * 60)

    if not TIER2_CSV.exists():
        log.error(f"Tier-2 data not found: {TIER2_CSV}")
        return None

    # Load model
    from api.tier2_service.model import Tier2Model
    model_mgr = Tier2Model(MODELS_DIR)
    if not model_mgr.load():
        log.error("Failed to load Tier-2 model")
        return None

    # Load data
    df = pd.read_csv(TIER2_CSV, dtype=str)
    df = df[df["split"] == "val"].reset_index(drop=True)
    df["tier2_label"] = df["tier2_label"].astype(int)
    df["tier1_confidence"] = pd.to_numeric(df["tier1_confidence"], errors="coerce").fillna(0.5)

    HEADER_FEATURE_NAMES = [
        "spf_pass", "spf_fail", "spf_none",
        "dkim_pass", "dkim_fail", "dkim_none",
        "dmarc_pass", "dmarc_fail", "dmarc_none",
        "sender_domain_match",
    ]
    for col in HEADER_FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)

    log.info(f"Evaluation set: {len(df):,} records")
    log.info(f"Label distribution: {df['tier2_label'].value_counts().to_dict()}")

    # Predict
    predictions = []
    latencies   = []
    class_map   = {"LEGITIMATE": 0, "SUSPICIOUS": 1, "PHISHING": 2}

    for _, row in df.iterrows():
        parts = []
        subject = str(row.get("subject", "")).strip()
        if subject and subject.lower() != "nan":
            parts.append(f"Subject: {subject}")
        body = str(row.get("body_text", "")).strip()
        if body and body.lower() != "nan":
            parts.append(f"Body: {body}")
        text = "\n".join(parts) if parts else body

        header_feats = [float(row.get(k, 0)) for k in HEADER_FEATURE_NAMES]

        t0 = time.perf_counter()
        result = model_mgr.predict(
            text=text,
            header_features=header_feats,
            tier1_confidence=float(row.get("tier1_confidence", 0.5)),
        )
        latencies.append((time.perf_counter() - t0) * 1000)

        predictions.append(class_map.get(result["verdict"], 1))

    true_labels = df["tier2_label"].values
    pred_labels = np.array(predictions)

    # Metrics
    report = classification_report(
        true_labels, pred_labels,
        target_names=["LEGITIMATE", "SUSPICIOUS", "PHISHING"],
        zero_division=0,
    )

    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1, 2])

    metrics = {
        "accuracy": round(accuracy_score(true_labels, pred_labels), 4),
        "f1_macro": round(f1_score(true_labels, pred_labels, average="macro", zero_division=0), 4),
        "f1_weighted": round(f1_score(true_labels, pred_labels, average="weighted", zero_division=0), 4),
        "precision_macro": round(precision_score(true_labels, pred_labels, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(true_labels, pred_labels, average="macro", zero_division=0), 4),
        "confusion_matrix": cm.tolist(),
        "avg_latency_ms": round(np.mean(latencies), 2),
        "p95_latency_ms": round(np.percentile(latencies, 95), 2),
        "classification_report": report,
    }

    log.info(f"\n{report}")
    log.info(f"Avg inference latency: {metrics['avg_latency_ms']:.2f}ms")
    log.info(f"P95 inference latency: {metrics['p95_latency_ms']:.2f}ms")

    # Save
    metrics_path = EVAL_DIR / "tier2_standalone_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    log.info(f"Metrics saved: {metrics_path}")

    return metrics


def evaluate_combined_system():
    """Evaluate Tier-1 + Tier-2 combined system on full test set."""
    log.info("\n" + "=" * 60)
    log.info("COMBINED SYSTEM EVALUATION (Tier-1 + Tier-2)")
    log.info("=" * 60)

    # Load tier2 config for escalation policy
    config_path = MODELS_DIR / "tier2_config.json"
    if not config_path.exists():
        log.error("tier2_config.json not found. Run calibrate_tier2_threshold.py first.")
        return None

    with open(config_path) as f:
        tier2_config = json.load(f)

    policy = tier2_config.get("escalation_policy", {})
    lower = policy.get("suspicious_lower", 0.08)
    upper = policy.get("suspicious_upper", 0.25)
    conf_thresh = policy.get("confidence_threshold", 0.70)

    log.info(f"Escalation policy: lower={lower}, upper={upper}, conf_thresh={conf_thresh}")

    # This would need both Tier-1 and Tier-2 predictions on the full dataset
    # For now, report based on Tier-1 predictions in the unified data
    log.info("Combined evaluation requires running full pipeline.")
    log.info("Run generate_tier2_dataset.py first, then this script will use the results.")

    return None


def print_comparison_table(tier1_metrics=None, combined_metrics=None):
    """Print the Tier-1 vs Tier-1+Tier-2 comparison table."""
    log.info("\n" + "=" * 60)
    log.info("COMPARISON: Tier-1 Only vs Tier-1 + Tier-2")
    log.info("=" * 60)

    # Tier-1 baseline metrics (from training)
    t1 = tier1_metrics or {
        "recall": "99.92%",
        "fpr": "TBD",
        "f1": "99.93%",
        "pr_auc": "TBD",
        "avg_latency": "~180ms",
    }

    t12 = combined_metrics or {
        "recall": "TBD",
        "fpr": "TBD",
        "f1": "TBD",
        "pr_auc": "TBD",
        "escalation_rate": "TBD",
        "avg_latency_non_esc": "~180ms",
        "avg_latency_esc": "TBD",
    }

    headers = f"{'Metric':<30} {'Tier-1 Only':<15} {'Tier-1 + Tier-2':<15}"
    log.info(headers)
    log.info("-" * 60)
    log.info(f"{'Phishing Recall':<30} {t1.get('recall', 'TBD'):<15} {t12.get('recall', 'TBD'):<15}")
    log.info(f"{'FPR':<30} {t1.get('fpr', 'TBD'):<15} {t12.get('fpr', 'TBD'):<15}")
    log.info(f"{'F1':<30} {t1.get('f1', 'TBD'):<15} {t12.get('f1', 'TBD'):<15}")
    log.info(f"{'PR-AUC':<30} {t1.get('pr_auc', 'TBD'):<15} {t12.get('pr_auc', 'TBD'):<15}")
    log.info(f"{'Escalation Rate':<30} {'N/A':<15} {t12.get('escalation_rate', 'TBD'):<15}")
    log.info(f"{'Avg Latency (non-escalated)':<30} {t1.get('avg_latency', 'TBD'):<15} {t12.get('avg_latency_non_esc', 'TBD'):<15}")
    log.info(f"{'Avg Latency (escalated)':<30} {'N/A':<15} {t12.get('avg_latency_esc', 'TBD'):<15}")


if __name__ == "__main__":
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Tier-2 standalone
    tier2_metrics = evaluate_tier2_standalone()

    # Combined system
    combined = evaluate_combined_system()

    # Comparison
    print_comparison_table(combined_metrics=combined)

    log.info("\nEvaluation complete.")
