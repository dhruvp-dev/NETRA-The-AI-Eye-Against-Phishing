# -*- coding: utf-8 -*-
"""
NETRA - DistilBERT Tier-1 Training Script
==========================================
Fine-tunes distilbert-base-uncased for phishing detection.

Examples:
    # Full run on a local NVIDIA GPU (bf16 is chosen automatically on Ampere, e.g. RTX 3060)
    python ml/train_distilbert.py --epochs 3 --batch_size 32 --grad_accum 1 --seed 42

    # Out of memory? Halve the batch and accumulate to keep the same effective batch size
    python ml/train_distilbert.py --batch_size 16 --grad_accum 2

    # Smoke run into a scratch folder (never touches the shipped weights)
    python ml/train_distilbert.py --max_train_rows 2000 --epochs 1 --output_dir ml/models/_smoke

Outputs (in --output_dir, default ml/models/):
    distilbert_tier1.pt        -- model weights
    distilbert_tokenizer/      -- tokenizer folder
    distilbert_config.json     -- inference configuration and training metadata (merged)

Train/serve parity:
    * Text is built by ml.features.text_builder.build_model_text (same as api/main.py).
    * Header features come from ml.features.header_features (same as api/main.py).
    * The architecture is ml.distilbert_model.PhishingClassifier (same as api/main.py).

Evaluation:
    * Checkpoint selection uses EMAIL rows of the validation split only (URL-only rows are not emails).
      Score = PR-AUC, ties broken by F1.
    * Reported test metrics use test_std and test_cross, email rows only, with breakdowns by source and
      label_origin. URL-only recall is reported separately.
"""

import argparse
import contextlib
import hashlib
import json
import logging
import math
import random
import sys
import time
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
# Paths and defaults
# ---------------------------------------------------------------------------
MODELS_DIR = ROOT / "ml" / "models"
DATA_CSV = ROOT / "data" / "processed" / "unified.csv"

CONFIDENCE_THRESHOLD = 0.70   # inference-time SUSPICIOUS band (kept for the config file)
MAX_LENGTH = 256
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.10
WEIGHT_DECAY = 0.01
SEED = 42
PATIENCE = 1
MIN_GROUP_ROWS = 20           # per-source / per-origin metrics are skipped below this size
IMBALANCE_WARN_RATIO = 10.0
TEXT_FORMAT = "v3"            # bump when build_model_text changes
PRETRAINED = "distilbert-base-uncased"


# ---------------------------------------------------------------------------
# Reproducibility, device, precision
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


def pick_precision(device, requested: str = None, legacy_fp16: bool = False) -> str:
    """
    'fp32' | 'fp16' | 'bf16'. Auto: bf16 on CUDA GPUs that support it (Ampere+, e.g. RTX 3060),
    fp16 on older CUDA GPUs, fp32 elsewhere. fp16/bf16 only apply on CUDA.
    """
    import torch
    if device.type != "cuda":
        return "fp32"
    if requested:
        if requested == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU does not support bf16. Use --precision fp16 or fp32.")
        return requested
    if legacy_fp16:
        return "fp16"
    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"


def autocast_dtype(precision: str):
    import torch
    return {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)


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


def load_split_frame(df: pd.DataFrame, splits):
    """
    Build model inputs for the requested splits from an already-loaded unified.csv.
    Returns (frame, header_matrix). Rows of frame and header_matrix line up by position.
    """
    df = df[df["split"].isin(splits)].reset_index(drop=True)
    texts = [_model_text_row(r) for _, r in df.iterrows()]
    hdr = np.array(
        [[float(extract_from_row(h, s, r)[k]) for k in HEADER_FEATURE_NAMES]
         for h, s, r in zip(df["headers_available"], df["sender"], df["reply_to"])],
        dtype=np.float32,
    )
    url_only = df["url_only"].astype(str).str.lower().eq("true").values \
        if "url_only" in df.columns else np.zeros(len(df), dtype=bool)
    frame = pd.DataFrame({
        "split": df["split"].values,
        "label": df["label"].astype(int).values,
        "text": texts,
        "url_only": url_only,
        "source": df["source"].astype(str).values if "source" in df.columns else "unknown",
        "label_origin": df["label_origin"].astype(str).values if "label_origin" in df.columns else "unknown",
    })
    return frame, hdr


def stratified_indices(labels: np.ndarray, n: int, seed: int) -> np.ndarray:
    """
    Positions of a class-proportional sample of size ~n. Used for smoke runs only.
    Keeps at least one row of each class present.
    """
    labels = np.asarray(labels)
    if n >= len(labels):
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    picked = []
    for cls in np.unique(labels):
        pos = np.flatnonzero(labels == cls)
        k = max(1, int(round(n * len(pos) / len(labels))))
        picked.append(rng.choice(pos, size=min(k, len(pos)), replace=False))
    return np.sort(np.concatenate(picked))


def class_balance_report(labels, name: str) -> None:
    n0, n1 = int((labels == 0).sum()), int((labels == 1).sum())
    log.info(f"{name}: legitimate={n0:,} malicious={n1:,}")
    if min(n0, n1) == 0:
        log.warning(f"{name} has a single class; metrics for it are undefined.")
    elif max(n0, n1) / min(n0, n1) > IMBALANCE_WARN_RATIO:
        log.warning(f"{name} class ratio above {IMBALANCE_WARN_RATIO:.0f}:1. "
                    "Check that legitimate mail is included.")


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
    labels = np.asarray(labels).astype(int)
    preds = (np.asarray(probs) >= threshold).astype(int)
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


def grouped_metrics(frame: pd.DataFrame, probs: np.ndarray, column: str) -> dict:
    """Metrics per value of `column` for groups with at least MIN_GROUP_ROWS rows."""
    out = {}
    for value, idx in frame.groupby(column).indices.items():
        if len(idx) < MIN_GROUP_ROWS:
            continue
        out[str(value)] = classification_metrics(frame["label"].values[idx], probs[idx])
    return out


def predict_probs(model, loader, device, amp_dtype=None):
    """Returns (malicious probability, label) arrays in loader order."""
    import torch
    model.eval()
    probs_all, labels_all = [], []
    ctx = (torch.autocast(device_type=device.type, dtype=amp_dtype)
           if amp_dtype is not None else contextlib.nullcontext())
    with torch.no_grad(), ctx:
        for batch in loader:
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
    precision = pick_precision(device, args.precision, legacy_fp16=args.fp16)
    amp_dtype = autocast_dtype(precision)
    out_dir = Path(args.output_dir)
    weights_path = out_dir / "distilbert_tier1.pt"
    tokenizer_dir = out_dir / "distilbert_tokenizer"
    config_path = out_dir / "distilbert_config.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    effective_batch = args.batch_size * args.grad_accum
    log.info(f"Device: {device} | precision: {precision} | seed: {args.seed} | output: {out_dir}")
    log.info(f"Batch {args.batch_size} x accum {args.grad_accum} = effective {effective_batch}")

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED)

    # --- Data: read the CSV once ---
    raw = pd.read_csv(args.data, dtype=str, keep_default_na=False)
    pipeline_version = raw["pipeline_version"].iloc[0] if "pipeline_version" in raw.columns else "unknown"
    if pipeline_version != "4":
        log.warning(f"unified.csv pipeline_version={pipeline_version}; expected 4. "
                    "Rebuild it with ml/data_pipeline.py.")
    frames = {}
    for name, splits in {"train": ["train"], "val": ["val"],
                         "test_std": ["test_std"], "test_cross": ["test_cross"]}.items():
        frames[name] = load_split_frame(raw, splits)
    del raw

    tr, tr_hdr = frames["train"]
    if args.max_train_rows:
        keep = stratified_indices(tr["label"].values, args.max_train_rows, args.seed)
        tr, tr_hdr = tr.iloc[keep].reset_index(drop=True), tr_hdr[keep]
        log.info(f"Smoke sample: {len(tr):,} training rows (requested {args.max_train_rows:,})")
    va, va_hdr = frames["val"]
    class_balance_report(tr["label"].values, "train (email + url-only)")
    class_balance_report(va["label"].values[~va["url_only"].values], "val (email rows)")

    train_enc = EncodedEmails(encode(tr["text"].tolist(), tokenizer, args.max_length), tr_hdr, tr["label"].values)
    val_enc = EncodedEmails(encode(va["text"].tolist(), tokenizer, args.max_length), va_hdr, va["label"].values)
    lengths = np.array([len(x) for x in train_enc.input_ids])
    log.info(f"Token length (train): p50={np.percentile(lengths, 50):.0f} "
             f"p90={np.percentile(lengths, 90):.0f} p99={np.percentile(lengths, 99):.0f} "
             f"(max_length={args.max_length})")

    train_loader = make_loader(train_enc, tokenizer, args.batch_size, shuffle=True)
    val_loader = make_loader(val_enc, tokenizer, args.batch_size * 2, shuffle=False)
    email_val_mask = ~va["url_only"].values
    if email_val_mask.sum() == 0 or len(np.unique(va["label"].values[email_val_mask])) < 2:
        raise ValueError("Validation split needs email rows of both classes to select a checkpoint. "
                         "Check the data pipeline output (ml/data_pipeline.py) and the sources it read.")

    # --- Model / loss / optimiser ---
    model = build_model().to(device)
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=tr["label"].values)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(WARMUP_RATIO * total_steps), num_training_steps=total_steps)
    # Loss scaling is only needed for fp16. bf16 has fp32's exponent range.
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))

    best_key, best_epoch, best_val, bad_epochs = (-1.0, -1.0), 0, None, 0
    log.info(f"Epochs={args.epochs} | LR={args.lr} | MaxLen={args.max_length} | patience={args.patience}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        total_loss, n_samples = 0.0, 0
        t0 = time.time()
        for step, batch in enumerate(train_loader, 1):
            ctx = (torch.autocast(device_type=device.type, dtype=amp_dtype)
                   if amp_dtype is not None else contextlib.nullcontext())
            with ctx:
                logits = model(batch["input_ids"].to(device),
                               batch["attention_mask"].to(device),
                               batch["header_features"].to(device))
                loss = criterion(logits.float(), batch["labels"].to(device))
            scaler.scale(loss / args.grad_accum).backward()
            total_loss += loss.item()
            n_samples += batch["labels"].size(0)
            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
        elapsed = time.time() - t0

        # Selection uses email rows only.
        probs, labels = predict_probs(model, val_loader, device, amp_dtype)
        m = classification_metrics(labels[email_val_mask], probs[email_val_mask])
        url_only_val = ~email_val_mask
        url_recall = (float(((probs[url_only_val] >= 0.5) & (labels[url_only_val] == 1)).sum()
                            / max(((labels[url_only_val] == 1)).sum(), 1))
                      if url_only_val.any() else None)
        key = (m["pr_auc"] if m["pr_auc"] is not None else -1.0, m["f1"])
        log.info(f"Epoch {epoch}/{args.epochs} | loss={total_loss / max(len(train_loader), 1):.4f} | "
                 f"{n_samples / elapsed:.1f} samples/s | val(email) PR-AUC={m['pr_auc']} F1={m['f1']} "
                 f"P={m['precision']} R={m['recall']} FPR={m['fpr']} | url-only recall={url_recall}")

        if key > best_key:
            best_key, best_epoch, best_val = key, epoch, {**m, "url_only_recall": url_recall}
            bad_epochs = 0
            torch.save(model.state_dict(), weights_path)
            log.info(f"  New best checkpoint saved (val PR-AUC={key[0]}, F1={key[1]})")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                log.info(f"Early stop: no val improvement for {bad_epochs} epoch(s).")
                break

    if device.type == "cuda":
        log.info(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # --- Reload best checkpoint (weights_only: no pickle code execution) ---
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))

    # --- Held-out test metrics: every row is scored, then split into email / URL-only ---
    test_metrics = {}
    for name in ("test_std", "test_cross"):
        f, hdr = frames[name]
        if len(f) == 0:
            log.warning(f"{name}: empty split.")
            continue
        enc = EncodedEmails(encode(f["text"].tolist(), tokenizer, args.max_length), hdr, f["label"].values)
        probs, labels = predict_probs(model, make_loader(enc, tokenizer, args.batch_size * 2, shuffle=False),
                                      device, amp_dtype)
        email = ~f["url_only"].values
        url_only = ~email
        if email.sum() == 0:
            log.warning(f"{name}: no email rows to evaluate.")
            continue
        e_frame = f[email].reset_index(drop=True)
        e_probs, e_labels = probs[email], labels[email]
        class_balance_report(e_labels, f"{name} (email rows)")
        overall = classification_metrics(e_labels, e_probs)
        url_recall = (float(((probs[url_only] >= 0.5) & (labels[url_only] == 1)).sum()
                            / max((labels[url_only] == 1).sum(), 1)) if url_only.any() else None)
        test_metrics[name] = {
            **overall,
            "by_source": grouped_metrics(e_frame, e_probs, "source"),
            "by_label_origin": grouped_metrics(e_frame, e_probs, "label_origin"),
            "url_only_recall": url_recall,
            "n_url_only": int(url_only.sum()),
        }
        log.info(f"{name}: F1={overall['f1']} PR-AUC={overall['pr_auc']} FPR={overall['fpr']} "
                 f"(n={overall['n']}, url-only recall={url_recall})")

    # --- Save tokenizer and config ---
    tokenizer.save_pretrained(str(tokenizer_dir))
    weights_sha = sha256_of(weights_path)

    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    config.update({
        "model": PRETRAINED,
        "max_length": args.max_length,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "header_features": HEADER_FEATURE_NAMES,
        "header_features_trained": True,
        "text_format": TEXT_FORMAT,
        "best_epoch": best_epoch,
        "best_val_f1": best_val["f1"],
        "best_val_recall": best_val["recall"],
        "best_val_fpr": best_val["fpr"],
        "best_val_pr_auc": best_val["pr_auc"],
        "val_metrics": best_val,
        "test_metrics": test_metrics,
        "weights_sha256": weights_sha,
        "training": {
            "epochs": args.epochs, "batch_size": args.batch_size, "grad_accum": args.grad_accum,
            "effective_batch_size": effective_batch, "lr": args.lr, "max_length": args.max_length,
            "precision": precision, "seed": args.seed, "patience": args.patience,
            "max_train_rows": args.max_train_rows, "data": str(args.data),
            "pipeline_version": pipeline_version,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        },
    })
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    log.info("=" * 60)
    log.info(f"Done. Best epoch {best_epoch}. Val (email) PR-AUC {best_val['pr_auc']}, F1 {best_val['f1']}")
    log.info(f"Weights   : {weights_path} (sha256 {weights_sha[:12]}...)")
    log.info(f"Config    : {config_path}")
    log.info("Next: calibrate thresholds with ml/calibrate_threshold.py")
    log.info("=" * 60)
    return model, tokenizer, config


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="NETRA DistilBERT Training")
    p.add_argument("--data", type=str, default=str(DATA_CSV), help="Path to unified.csv")
    p.add_argument("--output_dir", type=str, default=str(MODELS_DIR),
                   help="Where weights, tokenizer and config are written (default: ml/models)")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--grad_accum", type=int, default=1,
                   help="Accumulate gradients over N batches (same effective batch, less VRAM)")
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--max_length", type=int, default=MAX_LENGTH)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--patience", type=int, default=PATIENCE,
                   help="Stop after this many epochs without val improvement")
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default=None,
                   help="Default: bf16 on Ampere+ GPUs, fp16 on older GPUs, fp32 on CPU")
    p.add_argument("--fp16", action="store_true", help="Alias for --precision fp16 (kept for old commands)")
    p.add_argument("--max_train_rows", type=int, default=None,
                   help="Stratified subsample of train, for smoke runs only")
    args = p.parse_args(argv)
    if args.grad_accum < 1 or args.batch_size < 1:
        p.error("--batch_size and --grad_accum must be >= 1")
    return args


if __name__ == "__main__":
    train(parse_args())
