# -*- coding: utf-8 -*-
"""
NETRA - DistilBERT Tier-1 Training Script
==========================================
Fine-tunes distilbert-base-uncased for phishing detection.

Run (examples):
    python ml/train_distilbert.py --data data/processed/unified.csv
    python ml/train_distilbert.py --epochs 3 --batch_size 16 --fp16 --seed 42

Outputs (in ml/models/):
    distilbert_tier1.pt        -- model weights
    distilbert_tokenizer/      -- tokenizer folder
    distilbert_config.json     -- inference configuration (merged, existing keys kept)

Train/serve parity:
    * Text is built by ml.features.text_builder.build_model_text (same as api/main.py).
    * Header features come from ml.features.header_features (same as api/main.py).
    * The architecture is ml.distilbert_model.PhishingClassifier (same as api/main.py).

Evaluation:
    * Checkpoint selection uses VAL only.
    * Reported test metrics use test_std and test_cross. URL-only rows are excluded
      because they have no email body and would inflate the scores.

SUSPICIOUS is an INFERENCE-TIME state only (threshold rules live in
ml/models/threshold_config.json, see api/main.py).
"""

import argparse
import hashlib
import json
import logging
import random
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from ml.features.header_features import HEADER_FEATURE_NAMES, extract_from_row
from ml.features.text_builder import build_model_text, extract_urls

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODELS_DIR = ROOT / "ml" / "models"
EVAL_DIR = ROOT / "ml" / "evaluation"
DATA_CSV = ROOT / "data" / "processed" / "unified.csv"
WEIGHTS_PATH = MODELS_DIR / "distilbert_tier1.pt"
TOKENIZER_DIR = MODELS_DIR / "distilbert_tokenizer"
CONFIG_PATH = MODELS_DIR / "distilbert_config.json"

# ---------------------------------------------------------------------------
# Defaults (overridable from the command line)
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.70   # inference-time SUSPICIOUS band (kept for the config file)
MAX_LENGTH = 256
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.10
WEIGHT_DECAY = 0.01
SEED = 42
IMBALANCE_WARN_RATIO = 10.0
TEXT_FORMAT = "v3"            # bump when build_model_text changes
PRETRAINED = "distilbert-base-uncased"


# ---------------------------------------------------------------------------
# Reproducibility and device
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _model_text_row(row) -> str:
    """Build the exact string the API will build for the same email."""
    try:
        explicit = json.loads(row.get("urls") or "[]")
    except (TypeError, ValueError):
        explicit = []
    urls = extract_urls(row.get("subject", ""), row.get("body_text", ""), explicit)
    return build_model_text(row.get("subject", ""), row.get("body_text", ""), urls)


def load_split_frame(csv_path: Path, splits):
    """Load unified.csv and return the requested splits with text, header features, labels."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df = df[df["split"].isin(splits)].reset_index(drop=True)

    texts = [_model_text_row(r) for _, r in df.iterrows()]
    hdr = np.array(
        [[float(extract_from_row(h, s, r)[k]) for k in HEADER_FEATURE_NAMES]
         for h, s, r in zip(df["headers_available"], df["sender"], df["reply_to"])],
        dtype=np.float32,
    )
    url_only = df["url_only"].astype(str).str.lower().eq("true").values \
        if "url_only" in df.columns else np.zeros(len(df), dtype=bool)

    out = pd.DataFrame({
        "split": df["split"].values,
        "label": df["label"].astype(int).values,
        "text": texts,
        "url_only": url_only,
    })
    return out, hdr


def class_balance_report(labels, name: str) -> None:
    n0, n1 = int((labels == 0).sum()), int((labels == 1).sum())
    log.info(f"{name}: legitimate={n0:,} malicious={n1:,}")
    if min(n0, n1) == 0:
        log.warning(f"{name} has a single class; metrics for it are undefined.")
    elif max(n0, n1) / min(n0, n1) > IMBALANCE_WARN_RATIO:
        log.warning(f"{name} class ratio above {IMBALANCE_WARN_RATIO:.0f}:1. "
                    "Check Enron/SpamAssassin ham is included.")


class EncodedEmails:
    """Pre-tokenised (no padding) examples. Padding is done per batch in collate()."""

    def __init__(self, input_ids, header, labels):
        self.input_ids = input_ids
        self.header = header
        self.labels = labels

    def __len__(self):
        return len(self.labels)


def encode(texts, tokenizer, max_length):
    enc = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
    return enc["input_ids"]


def make_loader(encoded: EncodedEmails, tokenizer, batch_size, shuffle, num_workers=0):
    import torch
    from torch.utils.data import DataLoader, Dataset

    class _DS(Dataset):
        def __len__(self):
            return len(encoded)

        def __getitem__(self, i):
            return encoded.input_ids[i], encoded.header[i], encoded.labels[i]

    def collate(batch):
        ids, hdr, lab = zip(*batch)
        padded = tokenizer.pad({"input_ids": list(ids)}, return_tensors="pt")
        return {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "header_features": torch.tensor(np.stack(hdr), dtype=torch.float32),
            "labels": torch.tensor(lab, dtype=torch.long),
        }

    return DataLoader(_DS(), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, num_workers=num_workers)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def classification_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                                 precision_score, recall_score)
    preds = (probs >= threshold).astype(int)
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    has_both = len(np.unique(labels)) == 2
    return {
        "n": int(len(labels)),
        "n_legitimate": int((labels == 0).sum()),
        "n_malicious": int((labels == 1).sum()),
        "accuracy": round(float(accuracy_score(labels, preds)), 4),
        "precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(labels, preds, zero_division=0)), 4),
        "fpr": round(fp / (fp + tn + 1e-9), 4),
        "pr_auc": round(float(average_precision_score(labels, probs)), 4) if has_both else None,
    }


def predict_probs(model, loader, device, use_amp: bool):
    import torch
    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(batch["input_ids"].to(device),
                               batch["attention_mask"].to(device),
                               batch["header_features"].to(device))
            probs_all.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
            labels_all.append(batch["labels"].numpy())
    return np.concatenate(probs_all), np.concatenate(labels_all)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def build_model():
    from ml.distilbert_model import PhishingClassifier
    return PhishingClassifier(PRETRAINED)


def train(args):
    import torch
    import torch.nn as nn
    from sklearn.utils.class_weight import compute_class_weight
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    set_seed(args.seed)
    device = pick_device()
    use_amp = bool(args.fp16 and device.type == "cuda")
    log.info(f"Device: {device} | fp16: {use_amp} | seed: {args.seed}")

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED)

    # --- Data ---
    frames = {}
    for name, splits in {"train": ["train"], "val": ["val"],
                         "test_std": ["test_std"], "test_cross": ["test_cross"]}.items():
        frames[name], hdr = load_split_frame(Path(args.data), splits)
        frames[name + "_hdr"] = hdr
    tr, va = frames["train"], frames["val"]
    class_balance_report(tr["label"].values, "train")
    class_balance_report(va["label"].values, "val")

    def encoded_for(name):
        f = frames[name]
        return EncodedEmails(encode(f["text"].tolist(), tokenizer, args.max_length),
                             frames[name + "_hdr"], f["label"].values)

    train_enc = encoded_for("train")
    val_enc = encoded_for("val")
    train_loader = make_loader(train_enc, tokenizer, args.batch_size, shuffle=True)
    val_loader = make_loader(val_enc, tokenizer, args.batch_size * 2, shuffle=False)

    # --- Model / loss / optimiser ---
    model = build_model().to(device)
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=tr["label"].values)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(WARMUP_RATIO * total_steps), num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_f1, best_epoch = -1.0, 0   # -1 so epoch 1 always saves a checkpoint
    log.info(f"Epochs={args.epochs} | BatchSize={args.batch_size} | LR={args.lr} | MaxLen={args.max_length}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(batch["input_ids"].to(device),
                               batch["attention_mask"].to(device),
                               batch["header_features"].to(device))
                loss = criterion(logits.float(), batch["labels"].to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
            n_batches += 1

        probs, labels = predict_probs(model, val_loader, device, use_amp)
        m = classification_metrics(labels, probs)
        log.info(f"Epoch {epoch}/{args.epochs} | loss={total_loss / max(n_batches, 1):.4f} | "
                 f"val F1={m['f1']} P={m['precision']} R={m['recall']} FPR={m['fpr']} PR-AUC={m['pr_auc']}")

        if m["f1"] > best_f1:
            best_f1, best_epoch = m["f1"], epoch
            torch.save(model.state_dict(), WEIGHTS_PATH)
            log.info(f"  New best checkpoint saved (val F1={best_f1:.4f})")

    # --- Reload best checkpoint (weights_only: no pickle code execution) ---
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
    val_probs, val_labels = predict_probs(model, val_loader, device, use_amp)
    val_metrics = classification_metrics(val_labels, val_probs)

    # --- Held-out test metrics (URL-only rows excluded) ---
    test_metrics = {}
    for name in ("test_std", "test_cross"):
        f = frames[name]
        keep = ~f["url_only"].values
        if keep.sum() == 0:
            log.warning(f"{name}: no email rows to evaluate.")
            continue
        enc = EncodedEmails(
            encode([t for t, k in zip(f["text"], keep) if k], tokenizer, args.max_length),
            frames[name + "_hdr"][keep], f["label"].values[keep])
        loader = make_loader(enc, tokenizer, args.batch_size * 2, shuffle=False)
        p, y = predict_probs(model, loader, device, use_amp)
        test_metrics[name] = classification_metrics(y, p)
        class_balance_report(y, name)
        log.info(f"{name}: {test_metrics[name]}")

    # --- Save tokenizer and config ---
    tokenizer.save_pretrained(str(TOKENIZER_DIR))
    weights_sha = sha256_of(WEIGHTS_PATH)

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
    config.update({
        "model": PRETRAINED,
        "max_length": args.max_length,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "header_features": HEADER_FEATURE_NAMES,
        "header_features_trained": True,
        "text_format": TEXT_FORMAT,
        "best_epoch": best_epoch,
        "best_val_f1": val_metrics["f1"],
        "best_val_recall": val_metrics["recall"],
        "best_val_fpr": val_metrics["fpr"],
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "weights_sha256": weights_sha,
        "training": {
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "max_length": args.max_length, "fp16": use_amp, "seed": args.seed,
            "data": str(args.data), "trained_at": datetime.now(timezone.utc).isoformat(),
        },
    })
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

    log.info("=" * 60)
    log.info(f"Done. Best epoch {best_epoch}, val F1 {best_f1:.4f}")
    log.info(f"Weights   : {WEIGHTS_PATH} (sha256 {weights_sha[:12]}...)")
    log.info(f"Config    : {CONFIG_PATH}")
    log.info("Next: python ml/calibrate_threshold.py  (sets thresholds from val probabilities)")
    log.info("=" * 60)
    return model, tokenizer, config


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NETRA DistilBERT Training")
    parser.add_argument("--data", type=str, default=str(DATA_CSV), help="Path to unified.csv")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--fp16", action="store_true", help="Mixed precision (CUDA only)")
    train(parser.parse_args())
