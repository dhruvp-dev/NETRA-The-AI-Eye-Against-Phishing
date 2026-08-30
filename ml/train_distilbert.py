# -*- coding: utf-8 -*-
"""
NETRA - DistilBERT Tier-1 Training Script
==========================================
Fine-tunes distilbert-base-uncased for phishing detection.
Designed to run in Google Colab on T4 GPU (~20-25 min).

Run:
    python ml/train_distilbert.py --data /path/to/unified.csv

Outputs (in ml/models/):
    distilbert_tier1.pt        -- model weights (~250MB)
    distilbert_tokenizer/      -- tokenizer folder
    distilbert_config.json     -- inference configuration

Architecture:
    [Email Body Text]
           |
    DistilBERT (distilbert-base-uncased)
           |
    [CLS] token embedding (768 dims)
           |                    [Header auth features (10)] --> Linear(10->32) --> ReLU
    Concatenate [768 + 32 = 800 dims]
           |
    Classifier head: Linear(800->256) -> ReLU -> Dropout(0.3) -> Linear(256->2)
           |
    [LEGITIMATE / PHISHING]

SUSPICIOUS is an INFERENCE-TIME state only:
    If max(softmax_probability) < confidence_threshold (0.70):
        classification = SUSPICIOUS (model is uncertain)
    else:
        classification = argmax(probabilities)
"""

import sys
import json
import logging
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODELS_DIR = ROOT / "ml" / "models"
EVAL_DIR   = ROOT / "ml" / "evaluation"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

DATA_CSV = ROOT / "data" / "processed" / "unified.csv"

# ---------------------------------------------------------------------------
# Header feature names (must match header_features.py exactly)
# ---------------------------------------------------------------------------
HEADER_FEATURE_NAMES = [
    "spf_pass", "spf_fail", "spf_none",
    "dkim_pass", "dkim_fail", "dkim_none",
    "dmarc_pass", "dmarc_fail", "dmarc_none",
    "sender_domain_match",
]

CONFIDENCE_THRESHOLD = 0.70    # below this -> SUSPICIOUS
MAX_LENGTH           = 256     # token limit (95th percentile of email body length)
BATCH_SIZE           = 32      # fits T4 16GB VRAM at max_length=256
EPOCHS               = 3       # standard for BERT fine-tuning (more = overfitting risk)
LEARNING_RATE        = 2e-5    # AdamW standard for BERT fine-tuning
WARMUP_STEPS         = 100


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PhishingEmailDataset:
    """
    PyTorch Dataset for phishing email classification.
    Stores pre-tokenized input_ids, attention_mask, header features, and labels.
    """

    def __init__(self, input_ids, attention_masks, header_features, labels):
        import torch
        self.input_ids       = torch.tensor(input_ids,       dtype=torch.long)
        self.attention_masks = torch.tensor(attention_masks, dtype=torch.long)
        self.header_features = torch.tensor(header_features, dtype=torch.float32)
        self.labels          = torch.tensor(labels,          dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":       self.input_ids[idx],
            "attention_mask":  self.attention_masks[idx],
            "header_features": self.header_features[idx],
            "labels":          self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model():
    """
    Custom PhishingClassifier that combines DistilBERT [CLS] embedding
    with a projected header feature vector before classification.
    """
    import torch
    import torch.nn as nn
    from transformers import DistilBertModel

    class PhishingClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            # DistilBERT encoder — outputs 768-dim hidden states
            self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")

            # Project 10 header features to 32 dims to match scale of CLS embedding
            self.header_projection = nn.Linear(10, 32)

            # Classifier: 768 (CLS) + 32 (header) = 800 input dims
            self.classifier = nn.Sequential(
                nn.Linear(768 + 32, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 2),   # 2 output classes: LEGITIMATE (0), PHISHING (1)
            )

        def forward(self, input_ids, attention_mask, header_features):
            # Extract [CLS] token representation from DistilBERT
            db_output  = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = db_output.last_hidden_state[:, 0, :]   # shape: (batch, 768)

            # Project header auth signals
            header_proj = torch.relu(self.header_projection(header_features))  # shape: (batch, 32)

            # Fuse and classify
            combined = torch.cat([cls_output, header_proj], dim=1)  # shape: (batch, 800)
            return self.classifier(combined)                          # shape: (batch, 2)

    return PhishingClassifier()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_and_prepare_data(csv_path: Path, tokenizer):
    """
    Load unified.csv, tokenize body_text, extract header features.
    Returns train and val datasets.
    """
    log.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path, dtype=str)

    # Filter to train + val splits
    df = df[df["split"].isin(["train", "val"])].reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    log.info(f"Records: {len(df):,} | Label dist: {df['label'].value_counts().to_dict()}")

    # Fill missing header feature columns with 0
    for col in HEADER_FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)

    # Tokenize body_text
    log.info(f"Tokenizing body_text (max_length={MAX_LENGTH})...")
    body_texts = df["body_text"].fillna("").tolist()
    encodings  = tokenizer(
        body_texts,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors=None,   # return as lists (we convert to tensors in Dataset)
    )

    header_feats = df[HEADER_FEATURE_NAMES].values.tolist()
    labels       = df["label"].tolist()
    splits       = df["split"].tolist()

    # Split into train and val
    train_idx = [i for i, s in enumerate(splits) if s == "train"]
    val_idx   = [i for i, s in enumerate(splits) if s == "val"]

    def make_dataset(idx_list):
        return PhishingEmailDataset(
            input_ids       = [encodings["input_ids"][i]      for i in idx_list],
            attention_masks = [encodings["attention_mask"][i]  for i in idx_list],
            header_features = [header_feats[i]                 for i in idx_list],
            labels          = [labels[i]                       for i in idx_list],
        )

    train_ds = make_dataset(train_idx)
    val_ds   = make_dataset(val_idx)
    log.info(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    # Class distribution in train
    train_labels = [labels[i] for i in train_idx]
    n_legit      = train_labels.count(0)
    n_phish      = train_labels.count(1)
    log.info(f"Train class dist — Legitimate: {n_legit:,} | Phishing: {n_phish:,}")

    return train_ds, val_ds, train_labels


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, loader, device, desc="Eval"):
    """Evaluate model on a DataLoader. Returns metrics dict."""
    import torch
    from sklearn.metrics import (accuracy_score, precision_score,
                                  recall_score, f1_score, average_precision_score)

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids       = batch["input_ids"].to(device)
            attention_mask  = batch["attention_mask"].to(device)
            header_features = batch["header_features"].to(device)
            labels          = batch["labels"]

            logits = model(input_ids, attention_mask, header_features)
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    fp = np.sum((all_preds == 1) & (all_labels == 0))
    tn = np.sum((all_preds == 0) & (all_labels == 0))
    fpr = fp / (fp + tn + 1e-9)

    return {
        "accuracy":  round(accuracy_score(all_labels, all_preds), 4),
        "precision": round(precision_score(all_labels, all_preds, zero_division=0), 4),
        "recall":    round(recall_score(all_labels, all_preds, zero_division=0), 4),
        "f1":        round(f1_score(all_labels, all_preds, zero_division=0), 4),
        "fpr":       round(float(fpr), 4),
        "pr_auc":    round(average_precision_score(all_labels, all_probs), 4),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(data_csv: Path = DATA_CSV):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from sklearn.utils.class_weight import compute_class_weight
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # --- Tokenizer ---
    log.info("Loading DistilBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # --- Data ---
    train_ds, val_ds, train_labels = load_and_prepare_data(data_csv, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # --- Model ---
    log.info("Building PhishingClassifier (DistilBERT + header fusion)...")
    model = build_model().to(device)

    # --- Class weights (handles imbalance without SMOTE) ---
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_labels)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion     = nn.CrossEntropyLoss(weight=weight_tensor)

    # --- Optimizer + Scheduler ---
    total_steps = len(train_loader) * EPOCHS
    optimizer   = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
    )

    # --- Training loop ---
    best_f1        = 0.0
    best_model_path = MODELS_DIR / "distilbert_tier1.pt"

    log.info("=" * 60)
    log.info("Starting training...")
    log.info(f"Epochs={EPOCHS} | BatchSize={BATCH_SIZE} | LR={LEARNING_RATE}")
    log.info("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss   = 0.0
        batch_count  = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", ncols=90)
        for batch in pbar:
            input_ids       = batch["input_ids"].to(device)
            attention_mask  = batch["attention_mask"].to(device)
            header_features = batch["header_features"].to(device)
            labels          = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, header_features)
            loss   = criterion(logits, labels)
            loss.backward()

            # Gradient clipping prevents exploding gradients during fine-tuning
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            total_loss  += loss.item()
            batch_count += 1
            pbar.set_postfix({"loss": f"{total_loss / batch_count:.4f}"})

        avg_loss = total_loss / max(batch_count, 1)

        # Evaluate on val set after each epoch
        metrics = evaluate(model, val_loader, device, desc=f"Val Epoch {epoch}")
        log.info(
            f"\nEpoch {epoch}/{EPOCHS} | AvgLoss={avg_loss:.4f} | "
            f"Acc={metrics['accuracy']} | P={metrics['precision']} | "
            f"R={metrics['recall']} | F1={metrics['f1']} | "
            f"FPR={metrics['fpr']} | PR-AUC={metrics['pr_auc']}"
        )

        # Best model checkpointing (save on val F1 improvement)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), best_model_path)
            log.info(f"  New best model saved (F1={best_f1:.4f})")

    log.info(f"\nTraining complete. Best val F1 = {best_f1:.4f}")

    # --- Final evaluation on val set ---
    log.info("\nFinal evaluation on validation set:")
    from sklearn.metrics import classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Load best checkpoint
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    final_metrics = evaluate(model, val_loader, device)

    log.info(f"Phishing Recall : {final_metrics['recall']}")
    log.info(f"FPR             : {final_metrics['fpr']}")
    log.info(f"PR-AUC          : {final_metrics['pr_auc']}")
    log.info(f"F1              : {final_metrics['f1']}")

    # Confusion matrix plot
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["header_features"].to(device),
            )
            all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            all_labels.extend(batch["labels"].numpy().tolist())

    from sklearn.metrics import confusion_matrix as sk_cm
    cm = sk_cm(all_labels, all_preds, labels=[0, 1])
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["LEGITIMATE", "PHISHING"],
                yticklabels=["LEGITIMATE", "PHISHING"])
    plt.title("NETRA DistilBERT — Confusion Matrix (Val Set)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = EVAL_DIR / "confusion_matrix_distilbert.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Confusion matrix saved: {cm_path}")

    # --- Save tokenizer ---
    tokenizer_dir = MODELS_DIR / "distilbert_tokenizer"
    tokenizer.save_pretrained(str(tokenizer_dir))
    log.info(f"Tokenizer saved: {tokenizer_dir}")

    # --- Save inference config ---
    config = {
        "model":                "distilbert-base-uncased",
        "max_length":           MAX_LENGTH,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "header_features":      HEADER_FEATURE_NAMES,
        "best_val_f1":          best_f1,
        "best_val_recall":      final_metrics["recall"],
        "best_val_fpr":         final_metrics["fpr"],
    }
    config_path = MODELS_DIR / "distilbert_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    log.info(f"Config saved: {config_path}")

    log.info("\n" + "=" * 60)
    log.info("DONE. Download these 3 items from Colab to ml/models/:")
    log.info("  1. distilbert_tier1.pt         (~250MB model weights)")
    log.info("  2. distilbert_tokenizer/        (tokenizer folder)")
    log.info("  3. distilbert_config.json       (inference config)")
    log.info("=" * 60)

    return model, tokenizer, config


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NETRA DistilBERT Training")
    parser.add_argument("--data", type=str, default=str(DATA_CSV),
                        help="Path to unified.csv")
    args = parser.parse_args()
    train(data_csv=Path(args.data))