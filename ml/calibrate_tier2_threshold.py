# -*- coding: utf-8 -*-
"""
NETRA - Tier-2 Escalation Threshold Calibrator
================================================
Sweeps escalation boundary parameters to find the optimal
tradeoff between escalation rate and recall improvement.

Run:
    python ml/calibrate_tier2_threshold.py

Output:
    ml/models/tier2_config.json
"""

import sys
import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = ROOT / "ml" / "models"
DATA_CSV   = ROOT / "data" / "processed" / "unified.csv"


def calibrate():
    """
    Sweep escalation thresholds and report the tradeoff between
    escalation rate and the number of phishing emails caught.
    """
    if not DATA_CSV.exists():
        log.error(f"unified.csv not found at {DATA_CSV}")
        return

    df = pd.read_csv(DATA_CSV, dtype=str)
    df["label"] = df["label"].astype(int)

    # If tier1 predictions exist in the tier2_train.csv, use those
    tier2_csv = ROOT / "data" / "processed" / "tier2_train.csv"
    if tier2_csv.exists():
        t2 = pd.read_csv(tier2_csv, dtype=str)
        log.info(f"Using existing Tier-1 predictions from {tier2_csv}")
    else:
        log.error("Run generate_tier2_dataset.py first to get Tier-1 predictions")
        return

    # Sweep lower and upper thresholds
    results = []
    lower_vals = np.arange(0.02, 0.15, 0.01)
    upper_vals = np.arange(0.15, 0.60, 0.05)

    # For now, compute based on the Tier-1 risk_score distribution
    t2["tier1_risk_score"] = pd.to_numeric(t2["tier1_risk_score"], errors="coerce")
    t2["label"] = t2["label"].astype(int)
    total = len(df)

    log.info(f"Total records in unified: {total:,}")
    log.info(f"Escalated records: {len(t2):,}")
    log.info(f"\nSweeping escalation thresholds...")

    best_config = None
    best_score  = -1

    for lower in lower_vals:
        for upper in upper_vals:
            if lower >= upper:
                continue

            # Count how many would be escalated with these thresholds
            mask = (t2["tier1_risk_score"] >= lower) & (t2["tier1_risk_score"] <= upper)
            escalated = mask.sum()
            esc_rate  = escalated / total if total > 0 else 0

            # Count phishing caught in escalated set
            phishing_in_esc = ((t2["label"] == 1) & mask).sum()
            total_phishing  = (t2["label"] == 1).sum()
            phishing_recall = phishing_in_esc / total_phishing if total_phishing > 0 else 0

            # Score: maximize phishing recall while keeping escalation rate reasonable
            # Penalize escalation rates > 30%
            penalty = max(0, esc_rate - 0.30) * 2
            score   = phishing_recall - penalty

            results.append({
                "lower":           round(float(lower), 3),
                "upper":           round(float(upper), 3),
                "escalated":       int(escalated),
                "escalation_rate": round(float(esc_rate), 4),
                "phishing_recall": round(float(phishing_recall), 4),
                "score":           round(float(score), 4),
            })

            if score > best_score:
                best_score  = score
                best_config = results[-1]

    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)

    log.info("\nTop 10 threshold configurations:")
    log.info(f"{'Lower':>8} {'Upper':>8} {'Escalated':>10} {'Esc Rate':>10} {'Phish Recall':>13} {'Score':>8}")
    log.info("-" * 65)
    for r in results[:10]:
        log.info(
            f"{r['lower']:>8.3f} {r['upper']:>8.3f} "
            f"{r['escalated']:>10,} {r['escalation_rate']:>10.4f} "
            f"{r['phishing_recall']:>13.4f} {r['score']:>8.4f}"
        )

    # Save best config
    config = {
        "escalation_policy": {
            "suspicious_lower": best_config["lower"],
            "suspicious_upper": best_config["upper"],
            "confidence_threshold": 0.70,
            "escalate_suspicious_always": True,
            "escalate_low_confidence_phishing": True,
            "escalate_conflicting_signals": True,
        },
        "calibration_results": {
            "best_escalation_rate": best_config["escalation_rate"],
            "best_phishing_recall": best_config["phishing_recall"],
            "best_score":           best_config["score"],
        },
        "tier2_model": {
            "model_name": "roberta-base",
            "model_path": "ml/models/roberta_tier2.pt",
            "tokenizer_path": "ml/models/roberta_tier2_tokenizer",
            "max_length": 512,
            "num_classes": 3,
        },
    }

    config_path = MODELS_DIR / "tier2_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    log.info(f"\nBest config saved to: {config_path}")
    log.info(f"Best: lower={best_config['lower']}, upper={best_config['upper']}, "
             f"esc_rate={best_config['escalation_rate']:.4f}, "
             f"phish_recall={best_config['phishing_recall']:.4f}")

    return config


if __name__ == "__main__":
    calibrate()
