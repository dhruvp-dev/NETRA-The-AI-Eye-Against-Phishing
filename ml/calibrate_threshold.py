# -*- coding: utf-8 -*-
"""
NETRA - Threshold Calibration
=============================
Sweeps thresholds on VALIDATION probabilities of the model that is actually served.

    python ml/calibrate_threshold.py                 # DistilBERT (default, served model)
    python ml/calibrate_threshold.py --model rf      # calibrated RF fallback

DistilBERT (default):
    * Loads ml/models/distilbert_tier1.pt and scores the val split with the same
      text builder and header features the API uses.
    * phishing_threshold  = threshold that maximises F1 on val.
    * suspicious_lower    = threshold at which val recall first reaches
                            --suspicious_recall (default 0.95), capped below phishing_threshold.
    * Writes ONLY these keys to ml/models/threshold_config.json (read-merge-write),
      so other keys are preserved.

RF (--model rf):
    * Calibrates the RF fallback and writes ml/models/rf_threshold_config.json.
      It never touches threshold_config.json, which belongs to DistilBERT.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from ml.features.url_features import extract as url_extract, URL_FEATURE_NAMES
from ml.features.header_features import extract_from_row, HEADER_FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = ROOT / "ml" / "models"
DATA_PATH = ROOT / "data" / "processed" / "unified.csv"
DISTILBERT_CONFIG_PATH = MODELS_DIR / "distilbert_config.json"
THRESHOLD_CONFIG_PATH = MODELS_DIR / "threshold_config.json"
RF_THRESHOLD_CONFIG_PATH = MODELS_DIR / "rf_threshold_config.json"


# ---------------------------------------------------------------------------
# Sweep (vectorised: one pass over sorted scores, no per-threshold matrices)
# ---------------------------------------------------------------------------
def sweep_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """Precision / recall / F1 / FPR at every distinct score. Predict positive if score >= threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    order = np.argsort(-y_prob, kind="stable")
    ys, ps = y_true[order], y_prob[order]

    tp = np.cumsum(ys == 1)
    fp = np.cumsum(ys == 0)
    # Keep only the last index of each distinct score (all ties are predicted together).
    last = np.r_[np.where(np.diff(ps) != 0)[0], len(ps) - 1]
    tp, fp, thr = tp[last], fp[last], ps[last]

    n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(n_pos, 1)
    fpr = fp / max(n_neg, 1)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    # Thresholds stay exact so a score that equals a threshold is classified the same way here and in the API.
    return pd.DataFrame({
        "threshold": thr,
        "precision": np.round(precision, 4),
        "recall": np.round(recall, 4),
        "f1": np.round(f1, 4),
        "fpr": np.round(fpr, 4),
    })


# The SUSPICIOUS band must be wide enough to hold real scores (audit B2: a 0.0002-wide band was unreachable).
MIN_SUSPICIOUS_BAND = 0.03
# F1 within this distance of the best counts as "the same"; a plateau wider than PLATEAU_MAX_WIDTH
# means the scores are saturated and max-F1 picks an arbitrary point on it.
PLATEAU_F1_TOLERANCE = 0.005
PLATEAU_MAX_WIDTH = 0.5


def f1_plateau_width(df_sweep: pd.DataFrame) -> float:
    """Width of the threshold range whose F1 is within PLATEAU_F1_TOLERANCE of the best."""
    near = df_sweep[df_sweep["f1"] >= df_sweep["f1"].max() - PLATEAU_F1_TOLERANCE]["threshold"]
    return float(near.max() - near.min()) if not near.empty else 0.0


def pick_thresholds(df_sweep: pd.DataFrame, suspicious_recall: float):
    best = df_sweep.loc[df_sweep["f1"].idxmax()]
    phishing = float(best["threshold"])

    # Lowest threshold that still reaches the recall target -> start of the SUSPICIOUS band.
    reach = df_sweep[(df_sweep["recall"] >= suspicious_recall) & (df_sweep["threshold"] < phishing)]
    if not reach.empty:
        susp_lower = float(reach["threshold"].max())
    else:
        susp_lower = round(phishing * 0.6, 4)
        log.warning(f"Recall {suspicious_recall} is not reached below the phishing threshold; "
                    f"falling back to suspicious_lower = 0.6 * phishing = {susp_lower}")
    if phishing - susp_lower < MIN_SUSPICIOUS_BAND:
        if phishing - MIN_SUSPICIOUS_BAND <= 0:
            phishing = 2 * MIN_SUSPICIOUS_BAND
        susp_lower = phishing - MIN_SUSPICIOUS_BAND
        log.warning(f"SUSPICIOUS band too narrow; widened to [{susp_lower:.4f}, {phishing:.4f})")
    return best, phishing, susp_lower


def saturated(df_sweep: pd.DataFrame) -> bool:
    width = f1_plateau_width(df_sweep)
    if width > PLATEAU_MAX_WIDTH:
        log.warning(f"F1 is flat over a {width:.2f}-wide threshold range: scores are saturated and "
                    "max-F1 is arbitrary. Thresholds are NOT written (use --force to write anyway).")
        return True
    return False


def print_sweep(df_sweep: pd.DataFrame, best_row) -> None:
    print("\n" + "=" * 70)
    print(f"{'threshold':>10} | {'precision':>10} | {'recall':>8} | {'f1':>8} | {'fpr':>8}")
    print("-" * 70)
    step = max(1, len(df_sweep) // 30)
    for _, row in df_sweep.iloc[::step].iterrows():
        marker = " <<< BEST F1" if abs(row["threshold"] - best_row["threshold"]) < 1e-6 else ""
        print(f"{row['threshold']:>10.4f} | {row['precision']:>10.4f} | {row['recall']:>8.4f} | "
              f"{row['f1']:>8.4f} | {row['fpr']:>8.4f}{marker}")
    print("=" * 70)


def merge_write(path: Path, updates: dict) -> dict:
    """Read existing JSON (if any), update only the given keys, write back atomically."""
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    existing.update(updates)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    tmp.replace(path)
    return existing


# ---------------------------------------------------------------------------
# DistilBERT (served model)
# ---------------------------------------------------------------------------
def val_probs_distilbert(data_path: Path, models_dir: Path, include_url_only: bool = False):
    """
    Score the validation split with the model in models_dir. URL-only rows are excluded by
    default: the API only sees emails, so thresholds must be set on emails.
    """
    import torch
    from transformers import AutoTokenizer
    from ml.distilbert_model import PhishingClassifier
    from ml.train_distilbert import (EncodedEmails, encode, load_split_frame,
                                     make_loader, pick_device, predict_probs)

    cfg = json.loads((models_dir / "distilbert_config.json").read_text(encoding="utf-8"))
    max_len = int(cfg.get("max_length", 256))
    device = pick_device()

    tokenizer = AutoTokenizer.from_pretrained(str(models_dir / "distilbert_tokenizer"))
    model = PhishingClassifier()
    model.load_state_dict(torch.load(models_dir / "distilbert_tier1.pt",
                                     map_location=device, weights_only=True))
    model.to(device)

    raw = pd.read_csv(data_path, dtype=str, keep_default_na=False)
    frame, hdr = load_split_frame(raw, ["val"])
    if not include_url_only:
        keep = ~frame["url_only"].values
        frame, hdr = frame[keep].reset_index(drop=True), hdr[keep]
        log.info(f"URL-only rows excluded from calibration (use --include-url-only to keep them)")
    enc = EncodedEmails(encode(frame["text"].tolist(), tokenizer, max_len), hdr, frame["label"].values)
    loader = make_loader(enc, tokenizer, 64, shuffle=False)
    probs, labels = predict_probs(model, loader, device)
    return labels, probs


def run_distilbert(args) -> dict:
    models_dir = Path(args.models_dir)
    y, p = val_probs_distilbert(Path(args.data), models_dir, include_url_only=args.include_url_only)
    log.info(f"Val rows: {len(y):,} | legitimate={int((y == 0).sum()):,} malicious={int((y == 1).sum()):,}")
    if len(np.unique(y)) < 2:
        raise SystemExit("Val split has a single class; cannot calibrate.")

    df_sweep = sweep_thresholds(y, p)
    best, phishing, susp_lower = pick_thresholds(df_sweep, args.suspicious_recall)
    print_sweep(df_sweep, best)
    print(f"\nPhishing threshold (max F1): {phishing:.4f} | P={best['precision']} R={best['recall']} "
          f"F1={best['f1']} FPR={best['fpr']}")
    print(f"Suspicious lower (recall>={args.suspicious_recall}): {susp_lower:.4f}")
    if saturated(df_sweep) and not args.force:
        existing = models_dir / "threshold_config.json"
        return json.loads(existing.read_text(encoding="utf-8")) if existing.exists() else {}

    updates = {
        "phishing_threshold": float(phishing),
        "suspicious_lower": float(susp_lower),
        "suspicious_upper": float(phishing),
        "method": "val_pr_curve_f1",
        "calibrated_model": "distilbert_tier1",
        "calibrated_on": "val",
        "val_precision": float(best["precision"]),
        "val_recall": float(best["recall"]),
        "val_f1": float(best["f1"]),
        "val_fpr": float(best["fpr"]),
    }
    threshold_path = models_dir / "threshold_config.json"
    config = merge_write(threshold_path, updates)
    log.info(f"Saved DistilBERT thresholds: {threshold_path}")
    return config


# ---------------------------------------------------------------------------
# RF fallback
# ---------------------------------------------------------------------------
def build_rf_feature_matrix(df, text_extractor):
    X_text = text_extractor.transform(df)
    url_rows = []
    for urls_json in df["urls"].fillna("[]"):
        try:
            urls = json.loads(urls_json)
        except Exception:
            urls = []
        url_rows.append([url_extract(urls)[k] for k in URL_FEATURE_NAMES])
    X_url = csr_matrix(np.array(url_rows, dtype=np.float32))

    hdr_rows = [
        [extract_from_row(h, s, r)[k] for k in HEADER_FEATURE_NAMES]
        for h, s, r in zip(df["headers_available"].fillna("{}"), df["sender"].fillna(""), df["reply_to"].fillna(""))
    ]
    X_hdr = csr_matrix(np.array(hdr_rows, dtype=np.float32))
    return hstack([X_text, X_url, X_hdr])


def run_rf(args) -> dict:
    import joblib

    model = joblib.load(MODELS_DIR / "tier1_model_calibrated.pkl"
                        if (MODELS_DIR / "tier1_model_calibrated.pkl").exists()
                        else MODELS_DIR / "tier1_model.pkl")
    text_extractor = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")

    df = pd.read_csv(args.data, dtype=str, keep_default_na=False)
    df_val = df[df["split"] == "val"].reset_index(drop=True)
    y = df_val["label"].astype(int).values
    p = model.predict_proba(build_rf_feature_matrix(df_val, text_extractor))[:, 1]

    df_sweep = sweep_thresholds(y, p)
    best, phishing, susp_lower = pick_thresholds(df_sweep, args.suspicious_recall)
    print_sweep(df_sweep, best)
    if saturated(df_sweep) and not args.force:
        return {}

    config = {
        "phishing_threshold": float(phishing),
        "suspicious_lower": float(susp_lower),
        "suspicious_upper": float(phishing),
        "method": "val_pr_curve_f1",
        "calibrated_model": "rf",
        "val_f1": float(best["f1"]),
        "val_fpr": float(best["fpr"]),
    }
    RF_THRESHOLD_CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    log.info(f"Saved RF thresholds: {RF_THRESHOLD_CONFIG_PATH}")
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NETRA threshold calibration")
    parser.add_argument("--model", choices=["distilbert", "rf"], default="distilbert")
    parser.add_argument("--data", type=str, default=str(DATA_PATH))
    parser.add_argument("--models_dir", type=str, default=str(MODELS_DIR),
                        help="Folder with distilbert_tier1.pt and its config (default: ml/models)")
    parser.add_argument("--include-url-only", action="store_true",
                        help="Also calibrate on PhishTank/OpenPhish URL-only rows (not recommended)")
    parser.add_argument("--suspicious_recall", type=float, default=0.95,
                        help="Recall target that sets the lower edge of the SUSPICIOUS band")
    parser.add_argument("--force", action="store_true",
                        help="Write thresholds even when the F1 curve is flat (saturated scores)")
    args = parser.parse_args()
    cfg = run_distilbert(args) if args.model == "distilbert" else run_rf(args)
    print(f"\nConfig:\n{json.dumps(cfg, indent=2)}")
