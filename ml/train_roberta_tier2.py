# -*- coding: utf-8 -*-
"""
NETRA - Tier-2 RoBERTa Training Script
========================================
Fine-tunes roberta-base for 3-class phishing escalation.
Designed to run on RTX 5050 (8GB VRAM) or Colab T4.

Run:
    python ml/train_roberta_tier2.py --data data/processed/tier2_train.csv

Outputs (in ml/models/):
    roberta_tier2.pt                -- model weights
    roberta_tier2_tokenizer/        -- tokenizer folder
    roberta_tier2_config.json       -- inference configuration
    roberta_tier2_metrics.json      -- training metrics

Architecture:
    [Email Body Text]  (max 512 tokens)
           |
    RoBERTa-base (12 layers, 768 dims, 125M params)
           |
    [CLS] token embedding (768 dims)
           |                    [Header auth features (10)] -> Linear(10->32) -> ReLU
           |                    [Tier-1 confidence (1 dim)]
    Concatenate [768 + 32 + 1 = 801 dims]
           |
    Classifier head: Linear(801->256) -> ReLU -> Dropout(0.3) -> Linear(256->3)
           |
    [LEGITIMATE (0) / SUSPICIOUS (1) / PHISHING (2)]
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

DATA_CSV = ROOT / "data" / "processed" / "tier2_train.csv"

# ---------------------------------------------------------------------------
# Header feature names
# ---------------------------------------------------------------------------
HEADER_FEATURE_NAMES = [
    "spf_pass", "spf_fail", "spf_none",
    "dkim_pass", "dkim_fail", "dkim_none",
    "dmarc_pass", "dmarc_fail", "dmarc_none",
    "sender_domain_match",
]

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
NUM_CLASSES          = 3       # LEGITIMATE / SUSPICIOUS / PHISHING
MAX_LENGTH           = 512     # RoBERTa supports 512 tokens
BATCH_SIZE           = 16      # RoBERTa is larger, reduce from Tier-1's 32
EPOCHS               = 5       # early stopping on val macro-F1
LEARNING_RATE        = 2e-5
WARMUP_STEPS         = 100
DROPOUT              = 0.3
GRAD_CLIP            = 1.0
SEED                 = 42
CONFIDENCE_THRESHOLD = 0.60    # for 3-class, lower threshold makes sense


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class Tier2Dataset:
    def __init__(self, input_ids, attention_masks, header_features, tier1_confidences, labels):
        import torch
        self.input_ids       = torch.tensor(input_ids,        dtype=torch.long)
        self.attention_masks = torch.tensor(attention_masks,  dtype=torch.long)
        self.header_features = torch.tensor(header_features,  dtype=torch.float32)
        self.tier1_conf      = torch.tensor(tier1_confidences, dtype=torch.float32)
        self.labels          = torch.tensor(labels,           dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":       self.input_ids[idx],
            "attention_mask":  self.attention_masks[idx],
            "header_features": self.header_features[idx],
            "tier1_confidence": self.tier1_conf[idx],
            "labels":          self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model():
    import torch
    import torch.nn as nn
    from transformers import RobertaModel

    class RoBERTaTier2Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.roberta = RobertaModel.from_pretrained("roberta-base")
            self.header_projection = nn.Linear(10, 32)
            # 768 (CLS) + 32 (header) + 1 (tier1_confidence) = 801
            self.classifier = nn.Sequential(
                nn.Linear(801, 256),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(256, NUM_CLASSES),
            )

        def forward(self, input_ids, attention_mask, header_features, tier1_confidence):
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
            header_proj = torch.relu(self.header_projection(header_features))  # (batch, 32)
            tier1_conf = tier1_confidence.unsqueeze(1)  # (batch, 1)
            combined = torch.cat([cls_output, header_proj, tier1_conf], dim=1)  # (batch, 801)
            return self.classifier(combined)  # (batch, 3)

    return RoBERTaTier2Classifier()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_and_prepare_data(csv_path, tokenizer):
    log.info(f"Loading Tier-2 data from {csv_path}")
    df = pd.read_csv(csv_path, dtype=str)

    df = df[df["split"].isin(["train", "val"])].reset_index(drop=True)
    df["tier2_label"] = df["tier2_label"].astype(int)
    df["tier1_confidence"] = pd.to_numeric(df["tier1_confidence"], errors="coerce").fillna(0.5).astype(float)

    log.info(f"Records: {len(df):,}")
    log.info(f"Tier-2 label dist: {df['tier2_label'].value_counts().to_dict()}")

    for col in HEADER_FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)

    # Build enriched text prompt (same as Tier-1 style)
    log.info(f"Tokenizing (max_length={MAX_LENGTH})...")
    texts = []
    for _, row in df.iterrows():
        parts = []
        subject = str(row.get("subject", "")).strip()
        if subject and subject.lower() != "nan":
            parts.append(f"Subject: {subject}")
        body = str(row.get("body_text", "")).strip()
        if body and body.lower() != "nan":
            parts.append(f"Body: {body}")
        urls_str = str(row.get("urls", "")).strip()
        if urls_str and urls_str.lower() != "nan":
            try:
                url_list = json.loads(urls_str)
                for u in url_list[:5]:
                    parts.append(f"URL: {u}")
            except (json.JSONDecodeError, TypeError):
                pass
        texts.append("\n".join(parts) if parts else body)

    encodings = tokenizer(
        texts, max_length=MAX_LENGTH, truncation=True,
        padding="max_length", return_tensors=None,
    )

    header_feats     = df[HEADER_FEATURE_NAMES].values.tolist()
    tier1_confs      = df["tier1_confidence"].values.tolist()
    labels           = df["tier2_label"].tolist()
    splits           = df["split"].tolist()

    train_idx = [i for i, s in enumerate(splits) if s == "train"]
    val_idx   = [i for i, s in enumerate(splits) if s == "val"]

    def make_dataset(idx_list):
        return Tier2Dataset(
            input_ids       = [encodings["input_ids"][i]      for i in idx_list],
            attention_masks = [encodings["attention_mask"][i]  for i in idx_list],
            header_features = [header_feats[i]                for i in idx_list],
            tier1_confidences = [tier1_confs[i]               for i in idx_list],
            labels          = [labels[i]                      for i in idx_list],
        )

    train_ds = make_dataset(train_idx)
    val_ds   = make_dataset(val_idx)
    train_labels = [labels[i] for i in train_idx]

    log.info(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}")
    return train_ds, val_ds, train_labels


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, dataloader, device, desc="Eval"):
    import torch
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, classification_report)
    from tqdm import tqdm

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, ncols=90, leave=False):
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["header_features"].to(device),
                batch["tier1_confidence"].to(device),
            )
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(batch["labels"].numpy().tolist())
            all_probs.extend(probs.tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    return {
        "accuracy":  round(accuracy_score(all_labels, all_preds), 4),
        "precision_macro": round(precision_score(all_labels, all_preds, average="macro", zero_division=0), 4),
        "recall_macro":    round(recall_score(all_labels, all_preds, average="macro", zero_division=0), 4),
        "f1_macro":        round(f1_score(all_labels, all_preds, average="macro", zero_division=0), 4),
        "f1_weighted":     round(f1_score(all_labels, all_preds, average="weighted", zero_division=0), 4),
        "per_class_f1":    [round(x, 4) for x in f1_score(all_labels, all_preds, average=None, zero_division=0)],
        "classification_report": classification_report(
            all_labels, all_preds,
            target_names=["LEGITIMATE", "SUSPICIOUS", "PHISHING"],
            zero_division=0,
        ),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(data_csv=DATA_CSV):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.amp import autocast, GradScaler
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from sklearn.utils.class_weight import compute_class_weight
    from tqdm import tqdm

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # --- Tokenizer ---
    log.info("Loading RoBERTa tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # --- Data ---
    train_ds, val_ds, train_labels = load_and_prepare_data(data_csv, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # --- Model ---
    log.info("Building RoBERTaTier2Classifier...")
    model = build_model().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Total params: {total_params:,} | Trainable: {trainable:,}")

    # --- Class weights ---
    unique_labels = sorted(set(train_labels))
    class_weights = compute_class_weight("balanced", classes=np.array(unique_labels), y=np.array(train_labels))
    # Pad to NUM_CLASSES if some classes are missing
    weight_arr = np.ones(NUM_CLASSES)
    for i, lbl in enumerate(unique_labels):
        weight_arr[lbl] = class_weights[i]
    weight_tensor = torch.tensor(weight_arr, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    log.info(f"Class weights: {weight_arr}")

    # --- Optimizer + Scheduler ---
    total_steps = len(train_loader) * EPOCHS
    optimizer   = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
    )

    # --- FP16 mixed precision ---
    scaler = GradScaler("cuda") if device.type == "cuda" else None
    use_amp = device.type == "cuda"

    # --- Training ---
    best_f1 = 0.0
    best_model_path = MODELS_DIR / "roberta_tier2.pt"
    patience = 2
    patience_counter = 0

    log.info("=" * 60)
    log.info("Starting Tier-2 RoBERTa training...")
    log.info(f"Epochs={EPOCHS} | Batch={BATCH_SIZE} | LR={LEARNING_RATE} | MaxLen={MAX_LENGTH}")
    log.info("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss  = 0.0
        batch_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", ncols=90)
        for batch in pbar:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            header_feats   = batch["header_features"].to(device)
            tier1_conf     = batch["tier1_confidence"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()

            if use_amp:
                with autocast("cuda"):
                    logits = model(input_ids, attention_mask, header_feats, tier1_conf)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(input_ids, attention_mask, header_feats, tier1_conf)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            scheduler.step()
            total_loss  += loss.item()
            batch_count += 1
            pbar.set_postfix({"loss": f"{total_loss / batch_count:.4f}"})

        avg_loss = total_loss / max(batch_count, 1)
        metrics = evaluate(model, val_loader, device, desc=f"Val Epoch {epoch}")

        log.info(
            f"\nEpoch {epoch}/{EPOCHS} | AvgLoss={avg_loss:.4f} | "
            f"Acc={metrics['accuracy']} | "
            f"F1_macro={metrics['f1_macro']} | F1_weighted={metrics['f1_weighted']}"
        )
        log.info(f"Per-class F1: {metrics['per_class_f1']}")
        log.info(f"\n{metrics['classification_report']}")

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            torch.save(model.state_dict(), best_model_path)
            log.info(f"  >>> New best model saved (F1_macro={best_f1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            log.info(f"  No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                log.info("  Early stopping triggered!")
                break

    log.info(f"\nTraining complete. Best val F1_macro = {best_f1:.4f}")

    # --- Final eval ---
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    final_metrics = evaluate(model, val_loader, device, desc="Final Eval")

    log.info("\n" + "=" * 60)
    log.info("FINAL EVALUATION")
    log.info("=" * 60)
    log.info(f"\n{final_metrics['classification_report']}")

    # --- Confusion matrix ---
    model.eval()
    all_preds, all_labels_list = [], []
    import torch as _torch
    with _torch.no_grad():
        for batch in val_loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["header_features"].to(device),
                batch["tier1_confidence"].to(device),
            )
            all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            all_labels_list.extend(batch["labels"].numpy().tolist())

    from sklearn.metrics import confusion_matrix as sk_cm
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = sk_cm(all_labels_list, all_preds, labels=[0, 1, 2])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["LEGITIMATE", "SUSPICIOUS", "PHISHING"],
                yticklabels=["LEGITIMATE", "SUSPICIOUS", "PHISHING"])
    plt.title("NETRA RoBERTa Tier-2 - Confusion Matrix (Val Set)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = EVAL_DIR / "confusion_matrix_roberta_tier2.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Confusion matrix saved: {cm_path}")

    # --- Save tokenizer ---
    tokenizer_dir = MODELS_DIR / "roberta_tier2_tokenizer"
    tokenizer.save_pretrained(str(tokenizer_dir))
    log.info(f"Tokenizer saved: {tokenizer_dir}")

    # --- Save config ---
    config = {
        "model":                "roberta-base",
        "num_classes":          NUM_CLASSES,
        "max_length":           MAX_LENGTH,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "header_features":      HEADER_FEATURE_NAMES,
        "best_val_f1_macro":    best_f1,
        "final_metrics":        final_metrics,
    }
    config_path = MODELS_DIR / "roberta_tier2_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    log.info(f"Config saved: {config_path}")

    # --- Save metrics ---
    metrics_path = MODELS_DIR / "roberta_tier2_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2, default=str)
    log.info(f"Metrics saved: {metrics_path}")

    log.info("\n" + "=" * 60)
    log.info("DONE. Files saved to ml/models/:")
    log.info("  1. roberta_tier2.pt                (~500MB model weights)")
    log.info("  2. roberta_tier2_tokenizer/         (tokenizer folder)")
    log.info("  3. roberta_tier2_config.json         (inference config)")
    log.info("  4. roberta_tier2_metrics.json        (evaluation metrics)")
    log.info("=" * 60)

    return model, tokenizer, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NETRA RoBERTa Tier-2 Training")
    parser.add_argument("--data", type=str, default=str(DATA_CSV))
    args = parser.parse_args()
    train(data_csv=Path(args.data))
