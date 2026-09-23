# -*- coding: utf-8 -*-
"""
NETRA - Tier-2 Training Dataset Generator
==========================================
Runs all training records through the Tier-1 DistilBERT model,
applies the escalation policy, and produces a filtered CSV for
Tier-2 RoBERTa training.

Run:
    python ml/generate_tier2_dataset.py [--data path/to/unified.csv]

Output:
    data/processed/tier2_train.csv
"""

import sys
import json
import re
import logging
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODELS_DIR       = ROOT / "ml" / "models"
DATA_CSV         = ROOT / "data" / "processed" / "unified.csv"
OUTPUT_CSV       = ROOT / "data" / "processed" / "tier2_train.csv"

DISTILBERT_PATH        = MODELS_DIR / "distilbert_tier1.pt"
DISTILBERT_CONFIG_PATH = MODELS_DIR / "distilbert_config.json"
DISTILBERT_TOK_PATH    = MODELS_DIR / "distilbert_tokenizer"

# ---------------------------------------------------------------------------
# Header feature names - must match train_distilbert.py
# ---------------------------------------------------------------------------
HEADER_FEATURE_NAMES = [
    "spf_pass", "spf_fail", "spf_none",
    "dkim_pass", "dkim_fail", "dkim_none",
    "dmarc_pass", "dmarc_fail", "dmarc_none",
    "sender_domain_match",
]

# ---------------------------------------------------------------------------
# Escalation parameters (widened for training data generation)
# ---------------------------------------------------------------------------
TRAIN_ESCALATION_LOWER = 0.03   # widen from production 0.08
TRAIN_ESCALATION_UPPER = 0.50   # widen from production 0.25
CONFIDENCE_THRESHOLD   = 0.70


# ---------------------------------------------------------------------------
# Tier-1 model loader
# ---------------------------------------------------------------------------
def load_tier1_model(device):
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, DistilBertModel

    log.info("Loading Tier-1 DistilBERT model...")
    with open(DISTILBERT_CONFIG_PATH) as f:
        config = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(str(DISTILBERT_TOK_PATH))

    class PhishingClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
            self.header_projection = nn.Linear(10, 32)
            self.classifier = nn.Sequential(
                nn.Linear(768 + 32, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 2),
            )

        def forward(self, input_ids, attention_mask, header_features):
            db_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = db_output.last_hidden_state[:, 0, :]
            header_proj = torch.relu(self.header_projection(header_features))
            combined = torch.cat([cls_output, header_proj], dim=1)
            return self.classifier(combined)

    model = PhishingClassifier()
    model.load_state_dict(torch.load(DISTILBERT_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    log.info(f"Tier-1 model loaded on {device}")
    return model, tokenizer, config


# ---------------------------------------------------------------------------
# Signal extraction (mirrors api/main.py _extract_signals)
# ---------------------------------------------------------------------------
def extract_signals(row):
    from ml.features.url_features import extract as url_extract

    headers = {}
    if pd.notna(row.get("headers_available")):
        try:
            headers = json.loads(row["headers_available"])
        except (json.JSONDecodeError, TypeError):
            pass

    spf   = str(headers.get("spf",  "none")).lower()
    dkim  = str(headers.get("dkim", "none")).lower()
    dmarc = str(headers.get("dmarc","none")).lower()
    header_auth_failed = any(v in ("fail", "softfail") for v in [spf, dkim, dmarc])

    body = str(row.get("body_text", "")).lower()
    URGENCY_KEYWORDS = [
        "urgent", "immediately", "verify your account", "suspended",
        "unauthorized", "click here", "act now", "expire", "confirm your",
        "security alert", "unusual activity", "limited time",
    ]
    urgency_count = sum(1 for kw in URGENCY_KEYWORDS if kw in body)
    urgency_detected = urgency_count >= 2

    all_urls = []
    if pd.notna(row.get("urls")):
        try:
            all_urls = json.loads(row["urls"])
        except (json.JSONDecodeError, TypeError):
            pass
    if not all_urls and body:
        all_urls = re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', body)

    url_feats = url_extract(all_urls)
    suspicious_urls_count = int(
        url_feats.get("any_http_url", 0) +
        url_feats.get("any_phishing_tld", 0) +
        url_feats.get("any_typosquatting", 0)
    )
    typosquatting = bool(url_feats.get("any_typosquatting", 0))

    return {
        "header_auth_failed":     header_auth_failed,
        "urgency_detected":       urgency_detected,
        "suspicious_urls":        suspicious_urls_count,
        "typosquatting_detected": typosquatting,
    }


# ---------------------------------------------------------------------------
# Escalation decision
# ---------------------------------------------------------------------------
def should_escalate(risk_score, confidence, verdict, signals):
    if verdict == "SUSPICIOUS":
        return True
    if verdict == "PHISHING" and confidence < CONFIDENCE_THRESHOLD:
        return True
    if verdict == "LEGITIMATE":
        signal_count = (
            int(signals.get("typosquatting_detected", False)) +
            int(signals.get("header_auth_failed", False)) +
            min(signals.get("suspicious_urls", 0), 2)
        )
        if signal_count >= 1:
            return True
    if TRAIN_ESCALATION_LOWER <= risk_score <= TRAIN_ESCALATION_UPPER:
        return True
    return False


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def generate_dataset(data_csv=DATA_CSV, batch_size=64):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not data_csv.exists():
        log.error(f"Data file not found: {data_csv}")
        log.info("Run 'python ml/data_pipeline.py' first to generate unified.csv")
        sys.exit(1)

    df = pd.read_csv(data_csv, dtype=str)
    df["label"] = df["label"].astype(int)
    log.info(f"Loaded {len(df):,} records from {data_csv}")
    log.info(f"Label distribution: {df['label'].value_counts().to_dict()}")

    model, tokenizer, config = load_tier1_model(device)
    max_len = config.get("max_length", 256)
    conf_thresh = config.get("confidence_threshold", 0.70)

    for col in HEADER_FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)

    log.info("Running Tier-1 inference on all records...")
    tier1_results = []
    texts = df["body_text"].fillna("").tolist()
    header_feats = df[HEADER_FEATURE_NAMES].values

    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        batch_texts = texts[start:end]
        batch_hdrs  = header_feats[start:end]

        enc = tokenizer(
            batch_texts, max_length=max_len, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        hdr_tensor = torch.tensor(batch_hdrs, dtype=torch.float32)

        with torch.no_grad():
            logits = model(
                enc["input_ids"].to(device),
                enc["attention_mask"].to(device),
                hdr_tensor.to(device),
            )
            proba = torch.softmax(logits, dim=1).cpu().numpy()

        for i in range(len(batch_texts)):
            risk_score = float(proba[i][1])
            confidence = float(max(proba[i]))
            pt = 0.25

            if confidence < conf_thresh or (0.08 <= risk_score < pt):
                verdict = "SUSPICIOUS"
            elif proba[i][1] > proba[i][0] or risk_score >= pt:
                verdict = "PHISHING"
            else:
                verdict = "LEGITIMATE"

            tier1_results.append({
                "risk_score": risk_score,
                "confidence": confidence,
                "verdict":    verdict,
            })

        if (start // batch_size) % 20 == 0:
            log.info(f"  Processed {end:,}/{len(df):,} records...")

    log.info(f"Tier-1 inference complete on {len(tier1_results):,} records")

    df["tier1_risk_score"] = [r["risk_score"] for r in tier1_results]
    df["tier1_confidence"] = [r["confidence"] for r in tier1_results]
    df["tier1_verdict"]    = [r["verdict"]    for r in tier1_results]

    log.info("Extracting threat signals...")
    signals_list = []
    for _, row in df.iterrows():
        signals_list.append(extract_signals(row))
    df["tier1_signals"] = [json.dumps(s) for s in signals_list]

    log.info("Applying escalation policy (widened corridor for training)...")
    escalation_mask = []
    for idx in range(len(df)):
        escalated = should_escalate(
            tier1_results[idx]["risk_score"],
            tier1_results[idx]["confidence"],
            tier1_results[idx]["verdict"],
            signals_list[idx],
        )
        escalation_mask.append(escalated)

    df["escalated"] = escalation_mask
    escalated_df = df[df["escalated"]].copy()

    log.info(f"Escalation results:")
    log.info(f"  Total records:    {len(df):,}")
    log.info(f"  Escalated:        {escalated_df.shape[0]:,} ({100*escalated_df.shape[0]/len(df):.1f}%)")
    log.info(f"  Not escalated:    {(len(df) - escalated_df.shape[0]):,}")

    def map_tier2_label(row):
        original_label = int(row["label"])
        tier1_verdict  = row["tier1_verdict"]
        if original_label == 1:
            return 2  # PHISHING
        elif tier1_verdict == "SUSPICIOUS":
            return 1  # SUSPICIOUS
        else:
            return 0  # LEGITIMATE

    escalated_df["tier2_label"] = escalated_df.apply(map_tier2_label, axis=1)

    log.info(f"Tier-2 label distribution:")
    label_map = {0: "LEGITIMATE", 1: "SUSPICIOUS", 2: "PHISHING"}
    for lbl in [0, 1, 2]:
        count = (escalated_df["tier2_label"] == lbl).sum()
        log.info(f"  {label_map[lbl]} (class {lbl}): {count:,}")

    output_cols = [
        "record_id", "source", "label", "tier2_label",
        "subject", "body_text", "urls",
        "sender", "reply_to", "headers_available",
        "tier1_risk_score", "tier1_confidence", "tier1_verdict", "tier1_signals",
        "split",
    ] + HEADER_FEATURE_NAMES
    output_cols = [c for c in output_cols if c in escalated_df.columns]
    output_df = escalated_df[output_cols].copy()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_CSV, index=False)
    log.info(f"Tier-2 training dataset saved: {OUTPUT_CSV}")
    log.info(f"  Rows: {len(output_df):,}")

    return output_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NETRA Tier-2 Dataset Generator")
    parser.add_argument("--data", type=str, default=str(DATA_CSV))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    generate_dataset(data_csv=Path(args.data), batch_size=args.batch_size)
