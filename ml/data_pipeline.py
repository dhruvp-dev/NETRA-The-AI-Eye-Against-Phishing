"""
NETRA - Data Pipeline
=====================
Reads raw CSVs/mbox files from data/raw/, parses email fields,
deduplicates, assigns labels and splits, and writes:
    data/processed/unified.csv

Schema:
    record_id, source, label, subject, body_text, urls (JSON list),
    sender, reply_to, headers_available (JSON dict), split

Run:
    python ml/data_pipeline.py
"""

import os
import re
import csv
import json
import sys
import hashlib
import mailbox
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

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
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = PROCESSED_DIR / "unified.csv"

# ---------------------------------------------------------------------------
# Split ratios
# ---------------------------------------------------------------------------
SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.10,
    "test_std": 0.10,
    "test_cross": 0.05,   # source-held-out
    "test_ext": 0.05,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
URL_REGEX = re.compile(
    r"https?://[^\s<>\"'{}|\\^`\[\]]+", re.IGNORECASE
)

HEADER_KEYS = ["received-spf", "authentication-results", "dkim-signature", "x-spam-status"]


def extract_urls(text: str) -> list:
    """Extract all HTTP/HTTPS URLs from a block of text."""
    if not text:
        return []
    return URL_REGEX.findall(text)


def body_hash(text: str) -> str:
    """SHA-256 hash of normalised body text for deduplication."""
    normalised = re.sub(r"\s+", " ", (text or "").strip().lower())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def parse_auth_results(header_value: str) -> dict:
    """
    Parse Authentication-Results header into SPF/DKIM/DMARC pass/fail/none.
    Returns dict with keys spf, dkim, dmarc → 'pass'|'fail'|'none'.
    """
    result = {"spf": "none", "dkim": "none", "dmarc": "none"}
    if not header_value:
        return result
    for proto in ("spf", "dkim", "dmarc"):
        match = re.search(rf"{proto}\s*=\s*(\w+)", header_value, re.IGNORECASE)
        if match:
            value = match.group(1).lower()
            result[proto] = value if value in ("pass", "fail") else "none"
    return result


def parse_spf_header(header_value: str) -> str:
    """Parse Received-SPF header → 'pass'|'fail'|'none'."""
    if not header_value:
        return "none"
    h = header_value.lower()
    if h.startswith("pass"):
        return "pass"
    if h.startswith("fail") or h.startswith("softfail"):
        return "fail"
    return "none"


def build_headers_dict(msg) -> dict:
    """
    Extract SPF/DKIM/DMARC authentication info from an email.message.Message.
    Returns a flat dict ready for JSON serialisation.
    """
    auth_header = msg.get("authentication-results", "")
    spf_header = msg.get("received-spf", "")
    auth = parse_auth_results(auth_header)

    # Received-SPF takes precedence if explicit
    if spf_header:
        auth["spf"] = parse_spf_header(spf_header)

    dkim_sig = msg.get("dkim-signature", "")
    if dkim_sig and auth["dkim"] == "none":
        # Presence of DKIM-Signature doesn't mean it passed — keep 'none'
        auth["dkim_signature_present"] = True
    else:
        auth["dkim_signature_present"] = False

    return auth


def safe_decode(payload: bytes) -> str:
    """Decode bytes to str, trying utf-8 then latin-1 as fallback."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return payload.decode(enc)
        except (UnicodeDecodeError, AttributeError):
            continue
    return payload.decode("utf-8", errors="replace")


def email_message_to_record(msg, source: str, label: int, record_id: str) -> dict:
    """Convert an email.message.Message into a unified record dict."""
    subject = msg.get("subject", "") or ""
    sender = msg.get("from", "") or ""
    reply_to = msg.get("reply-to", "") or ""

    # Extract body text
    body_parts = []
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type in ("text/plain", "text/html"):
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        body_parts.append(safe_decode(payload))
                except Exception:
                    pass
    else:
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                body_parts.append(safe_decode(payload))
        except Exception:
            pass

    body_text = " ".join(body_parts).strip()
    urls = extract_urls(body_text) + extract_urls(subject)
    headers_dict = build_headers_dict(msg)

    return {
        "record_id": record_id,
        "source": source,
        "label": label,
        "subject": subject.strip(),
        "body_text": body_text,
        "urls": json.dumps(urls),
        "sender": sender.strip(),
        "reply_to": reply_to.strip(),
        "headers_available": json.dumps(headers_dict),
        "split": "",           # assigned later
        "_body_hash": body_hash(body_text),
    }


# ---------------------------------------------------------------------------
# Dataset-specific readers
# ---------------------------------------------------------------------------

def read_enron_mbox(path: Path, label: int = 0) -> list:
    """
    Read Enron email dataset from mbox file.
    The Kaggle version is an mbox; treat all as legitimate (label=0).
    """
    records = []
    try:
        mbox = mailbox.mbox(str(path))
        for i, msg in enumerate(mbox):
            record_id = f"enron_{path.stem}_{i:07d}"
            rec = email_message_to_record(msg, source="enron", label=label, record_id=record_id)
            records.append(rec)
            if (i + 1) % 1000 == 0:
                log.info(f"  Enron: parsed {i+1} emails from {path.name}")
    except Exception as e:
        log.error(f"Failed to read Enron mbox {path}: {e}")
    log.info(f"Enron {path.name}: {len(records)} records")
    return records


def read_enron_csv(path: Path, label: int = 0) -> list:
    """
    Read Enron dataset from Kaggle CSV format.
    Expected columns: 'file' (path identifier) and 'message' (raw RFC-2822 text).
    All treated as legitimate (label=0).
    """
    import email as email_lib
    records = []
    try:
        log.info(f"Reading Enron CSV: {path.name} ({path.stat().st_size // 1024 // 1024} MB)")
        chunk_size = 5000
        chunk_num = 0
        for chunk in pd.read_csv(
            path,
            encoding="utf-8",
            on_bad_lines="skip",
            chunksize=chunk_size,
            dtype=str,
        ):
            chunk.columns = [c.lower().strip() for c in chunk.columns]
            for i, row in chunk.iterrows():
                raw_msg = row.get("message", "") or ""
                if not raw_msg.strip():
                    continue
                try:
                    msg = email_lib.message_from_string(raw_msg)
                except Exception:
                    continue
                record_id = f"enron_csv_{i:07d}"
                rec = email_message_to_record(
                    msg, source="enron", label=label, record_id=record_id
                )
                records.append(rec)
            chunk_num += 1
            if chunk_num % 10 == 0:
                log.info(f"  Enron CSV: processed {chunk_num * chunk_size:,} rows...")
    except Exception as e:
        log.error(f"Failed to read Enron CSV {path}: {e}")
    log.info(f"Enron CSV {path.name}: {len(records)} records")
    return records


def read_spamassassin_eml_folder(folder: Path, label: int) -> list:
    """
    Read SpamAssassin corpus from a folder of individual .eml files
    (no extension — files are named like 00001.7c53336b37003a9286aba55d2945844c).
    ham folders → label=0, spam folders → label=1.
    """
    import email as email_lib
    records = []
    files = list(folder.iterdir())
    for i, path in enumerate(files):
        if path.is_dir():
            continue
        try:
            raw = path.read_bytes()
            msg = email_lib.message_from_bytes(raw)
            record_id = f"spamassassin_{folder.name}_{i:06d}"
            rec = email_message_to_record(
                msg,
                source="spamassassin",
                label=label,
                record_id=record_id,
            )
            records.append(rec)
        except Exception as e:
            log.debug(f"Skipping {path.name}: {e}")
    log.info(f"SpamAssassin folder '{folder.name}': {len(records)} records, label={label}")
    return records


def read_nazario_mbox(path: Path) -> list:
    """
    Read Nazario phishing corpus in mbox format (e.g. phishing3.mbox from monkey.org).
    All messages are labelled phishing (label=1).
    """
    records = []
    try:
        mbox = mailbox.mbox(str(path))
        for i, msg in enumerate(mbox):
            record_id = f"nazario_mbox_{path.stem}_{i:06d}"
            rec = email_message_to_record(
                msg, source="nazario", label=1, record_id=record_id
            )
            records.append(rec)
        log.info(f"Nazario mbox '{path.name}': {len(records)} records")
    except Exception as e:
        log.error(f"Failed to read Nazario mbox {path}: {e}")
    return records


def read_nazario_csv(path: Path) -> list:
    """
    Read Nazario phishing corpus CSV.
    Expected columns (flexible): 'body' or 'text', optional 'subject'.
    All labelled phishing (label=1).
    """
    records = []
    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
        df.columns = [c.lower().strip() for c in df.columns]

        body_col = next((c for c in ("body", "text", "email", "content") if c in df.columns), None)
        subject_col = "subject" if "subject" in df.columns else None
        sender_col = "sender" if "sender" in df.columns else None

        for i, row in df.iterrows():
            body_text = str(row[body_col]) if body_col else ""
            subject = str(row[subject_col]) if subject_col else ""
            sender = str(row[sender_col]) if sender_col else ""
            record_id = f"nazario_{i:07d}"
            records.append({
                "record_id": record_id,
                "source": "nazario",
                "label": 1,
                "subject": subject.strip(),
                "body_text": body_text.strip(),
                "urls": json.dumps(extract_urls(body_text)),
                "sender": sender.strip(),
                "reply_to": "",
                "headers_available": json.dumps({}),
                "split": "",
                "_body_hash": body_hash(body_text),
            })
    except Exception as e:
        log.error(f"Failed to read Nazario CSV {path}: {e}")
    log.info(f"Nazario {path.name}: {len(records)} records")
    return records


def read_spamassassin_mbox(path: Path, label: int) -> list:
    """
    Read SpamAssassin corpus mbox files.
    ham/ directories → label=0, spam/ directories → label=1.
    """
    records = []
    try:
        mbox = mailbox.mbox(str(path))
        for i, msg in enumerate(mbox):
            record_id = f"spamassassin_{path.stem}_{i:07d}"
            rec = email_message_to_record(msg, source="spamassassin", label=label, record_id=record_id)
            records.append(rec)
    except Exception as e:
        log.error(f"Failed to read SpamAssassin mbox {path}: {e}")
    log.info(f"SpamAssassin {path.name}: {len(records)} records, label={label}")
    return records


def read_phishtank_csv(path: Path) -> list:
    """
    Read PhishTank CSV export.
    Expected columns: url, phish_detail_url, submission_time, verified, verification_time, online, target
    Each URL becomes a record with body_text=url (URL-only signal).
    label=1 (phishing URLs).
    """
    records = []
    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
        df.columns = [c.lower().strip() for c in df.columns]

        url_col = "url" if "url" in df.columns else df.columns[0]

        for i, row in df.iterrows():
            url = str(row.get(url_col, "")).strip()
            if not url:
                continue
            record_id = f"phishtank_{i:07d}"
            records.append({
                "record_id": record_id,
                "source": "phishtank",
                "label": 1,
                "subject": "",
                "body_text": f"Phishing URL: {url}",
                "urls": json.dumps([url]),
                "sender": "",
                "reply_to": "",
                "headers_available": json.dumps({}),
                "split": "",
                "_body_hash": body_hash(url),
            })
    except Exception as e:
        log.error(f"Failed to read PhishTank CSV {path}: {e}")
    log.info(f"PhishTank {path.name}: {len(records)} records")
    return records


def read_openphish_txt(path: Path) -> list:
    """
    Read OpenPhish feed text file (one URL per line).
    label=1 (phishing).
    """
    records = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                url = line.strip()
                if not url or not url.startswith("http"):
                    continue
                record_id = f"openphish_{i:07d}"
                records.append({
                    "record_id": record_id,
                    "source": "openphish",
                    "label": 1,
                    "subject": "",
                    "body_text": f"Phishing URL: {url}",
                    "urls": json.dumps([url]),
                    "sender": "",
                    "reply_to": "",
                    "headers_available": json.dumps({}),
                    "split": "",
                    "_body_hash": body_hash(url),
                })
    except Exception as e:
        log.error(f"Failed to read OpenPhish file {path}: {e}")
    log.info(f"OpenPhish {path.name}: {len(records)} records")
    return records


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

def assign_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Assign train/val/test_std/test_cross/test_ext splits.
    - test_cross: hold out one full source for cross-source evaluation
    - All others: stratified random split
    """
    np.random.seed(seed)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Identify smallest source for cross-source holdout
    source_counts = df["source"].value_counts()
    cross_source = source_counts.index[-1]  # smallest source
    log.info(f"Cross-source holdout: '{cross_source}' ({source_counts[cross_source]} records)")

    cross_mask = df["source"] == cross_source
    df.loc[cross_mask, "split"] = "test_cross"

    remaining = df[~cross_mask].copy()
    n = len(remaining)

    # Recalculate ratios for remaining splits (excluding test_cross)
    remaining_ratios = {k: v for k, v in SPLIT_RATIOS.items() if k != "test_cross"}
    total_remaining = sum(remaining_ratios.values())
    normalised = {k: v / total_remaining for k, v in remaining_ratios.items()}

    n_train = int(n * normalised["train"])
    n_val = int(n * normalised["val"])
    n_test_std = int(n * normalised["test_std"])

    splits_assigned = (
        ["train"] * n_train
        + ["val"] * n_val
        + ["test_std"] * n_test_std
        + ["test_ext"] * (n - n_train - n_val - n_test_std)
    )

    remaining["split"] = splits_assigned
    df.loc[remaining.index, "split"] = remaining["split"].values

    # Log split distribution
    for split_name, count in df["split"].value_counts().items():
        log.info(f"  Split '{split_name}': {count} records ({count/len(df)*100:.1f}%)")

    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def discover_files() -> dict:
    """Discover dataset files in data/raw/ by pattern matching."""
    files = {
        "enron_mbox": [],
        "enron_csv": [],         # Kaggle 'emails.csv' format
        "nazario_csv": [],
        "nazario_mbox": [],      # phishing3.mbox from monkey.org
        "spamassassin_ham_mbox": [],
        "spamassassin_spam_mbox": [],
        "spamassassin_ham_folder": [],   # folders of .eml files
        "spamassassin_spam_folder": [],  # folders of .eml files
        "phishtank_csv": [],
        "openphish_txt": [],
    }

    # --- Check for Kaggle Enron CSV ---
    enron_csv = RAW_DIR / "emails.csv"
    if enron_csv.exists():
        files["enron_csv"].append(enron_csv)

    # --- Check for SpamAssassin individual-file folders ---
    sa_ham_folders = ["easy_ham", "hard_ham"]
    sa_spam_folders = ["spam", "spam_2"]
    for folder_name in sa_ham_folders:
        folder = RAW_DIR / folder_name
        if folder.is_dir():
            files["spamassassin_ham_folder"].append(folder)
    for folder_name in sa_spam_folders:
        folder = RAW_DIR / folder_name
        if folder.is_dir():
            files["spamassassin_spam_folder"].append(folder)

    # --- Scan all other files ---
    for path in RAW_DIR.iterdir():
        if path.is_dir():
            continue  # handled above
        name = path.name.lower()

        if path.suffix == ".mbox" and "enron" in name:
            files["enron_mbox"].append(path)
        elif path.suffix == ".csv" and "nazario" in name:
            files["nazario_csv"].append(path)
        elif path.suffix == ".mbox" and ("phishing" in name or "nazario" in name):
            # Nazario corpus downloaded as mbox (e.g. phishing3.mbox from monkey.org)
            files["nazario_mbox"].append(path)
        elif path.suffix == ".csv" and "phishtank" in name:
            files["phishtank_csv"].append(path)
        elif "openphish" in name and path.suffix in (".txt", ".feed", ""):
            files["openphish_txt"].append(path)
        elif path.suffix == ".mbox":
            parent = path.parent.name.lower()
            if "ham" in name or "easy" in name or "hard" in name:
                files["spamassassin_ham_mbox"].append(path)
            elif "spam" in name:
                files["spamassassin_spam_mbox"].append(path)

    for key, paths in files.items():
        log.info(f"Discovered [{key}]: {len(paths)} file(s)")

    return files


def run_pipeline():
    log.info("=" * 60)
    log.info("NETRA Data Pipeline starting")
    log.info(f"Raw data dir  : {RAW_DIR}")
    log.info(f"Output CSV    : {OUTPUT_CSV}")
    log.info("=" * 60)

    files = discover_files()
    all_records = []

    # --- Enron (legitimate, label=0) ---
    # Kaggle CSV format (emails.csv with 'file' and 'message' columns)
    for path in files["enron_csv"]:
        all_records.extend(read_enron_csv(path, label=0))
    # Legacy mbox format
    for path in files["enron_mbox"]:
        all_records.extend(read_enron_mbox(path, label=0))

    # --- Nazario (phishing, label=1) ---
    for path in files["nazario_csv"]:
        all_records.extend(read_nazario_csv(path))
    for path in files["nazario_mbox"]:    # phishing3.mbox from monkey.org
        all_records.extend(read_nazario_mbox(path))

    # --- SpamAssassin (ham=0, spam=1) ---
    # Individual .eml file folders (Kaggle/SpamAssassin download format)
    for folder in files["spamassassin_ham_folder"]:
        all_records.extend(read_spamassassin_eml_folder(folder, label=0))
    for folder in files["spamassassin_spam_folder"]:
        all_records.extend(read_spamassassin_eml_folder(folder, label=1))
    # Legacy mbox format
    for path in files["spamassassin_ham_mbox"]:
        all_records.extend(read_spamassassin_mbox(path, label=0))
    for path in files["spamassassin_spam_mbox"]:
        all_records.extend(read_spamassassin_mbox(path, label=1))

    # --- PhishTank (phishing) ---
    for path in files["phishtank_csv"]:
        all_records.extend(read_phishtank_csv(path))

    # --- OpenPhish (phishing) ---
    for path in files["openphish_txt"]:
        all_records.extend(read_openphish_txt(path))

    if not all_records:
        log.warning("No records found. Make sure datasets are placed in data/raw/")
        log.warning("See README for exact file names expected.")
        return

    log.info(f"Total records before dedup: {len(all_records)}")

    # --- Build DataFrame ---
    df = pd.DataFrame(all_records)

    # --- Deduplicate on body_text hash ---
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["_body_hash"]).reset_index(drop=True)
    log.info(f"Removed {before_dedup - len(df)} duplicates. Remaining: {len(df)}")

    # --- Class balance report ---
    vc = df["label"].value_counts()
    log.info(f"Label distribution: Legitimate={vc.get(0, 0)}, Phishing={vc.get(1, 0)}")

    # --- Assign splits ---
    df = assign_splits(df)

    # --- Drop internal columns ---
    df = df.drop(columns=["_body_hash"])

    # --- Select and order final schema columns ---
    schema_cols = [
        "record_id", "source", "label", "subject", "body_text",
        "urls", "sender", "reply_to", "headers_available", "split",
    ]
    df = df[schema_cols]

    # --- Write output ---
    df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL)
    log.info(f"Written {len(df)} records to {OUTPUT_CSV}")
    log.info("Pipeline complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NETRA Data Pipeline")
    parser.add_argument(
        "--raw-dir", type=str, default=str(RAW_DIR),
        help="Path to raw data directory"
    )
    parser.add_argument(
        "--output", type=str, default=str(OUTPUT_CSV),
        help="Path for output unified CSV"
    )
    args = parser.parse_args()

    RAW_DIR = Path(args.raw_dir)
    OUTPUT_CSV = Path(args.output)

    run_pipeline()
