"""
NETRA - Data Pipeline
=====================
Reads raw CSVs/mbox files from data/raw/, parses email fields,
deduplicates, assigns labels and splits, and writes:
    data/processed/unified.csv

Schema:
    record_id, source, label, label_origin, subject, body_text, urls (JSON list),
    sender, reply_to, headers_available (JSON dict), url_only,
    <10 header feature columns>, split

Label semantics:
    label = 1 means "malicious/unwanted". SpamAssassin spam is included with
    label=1 and label_origin="spam" so spam can be told apart from phishing
    in evaluation (ML16). Enron and SpamAssassin ham are label=0.

URL-only records (PhishTank / OpenPhish) have an EMPTY body_text and the URL
in `urls`. They never receive a label-revealing prefix (ML2/H2). Pass
--exclude-url-only to drop them entirely.

Run:
    python ml/data_pipeline.py [--raw-dir ...] [--output ...] [--exclude-url-only]
"""

import re
import csv
import json
import sys
import html
import hashlib
import mailbox
import logging
import argparse
from email.header import decode_header, make_header
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from ml.features.header_features import (
    HEADER_FEATURE_NAMES,
    extract_from_row,
    normalize_auth,
)
from ml.features.text_builder import URL_PATTERN

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (output directory is created inside run_pipeline, after argparse)
# ---------------------------------------------------------------------------
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_CSV = PROCESSED_DIR / "unified.csv"

URL_ONLY_SOURCES = {"phishtank", "openphish"}

# ---------------------------------------------------------------------------
# Split ratios (train / val / test_std / test_ext). test_cross is a whole-source holdout.
# ---------------------------------------------------------------------------
SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.10,
    "test_std": 0.10,
    "test_ext": 0.05,
}

HEADER_KEYS = ["received-spf", "authentication-results", "dkim-signature", "x-spam-status"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_urls(text: str) -> list:
    """Extract http(s) and www. URLs (same pattern the API uses, ML2/A12)."""
    if not text:
        return []
    return URL_PATTERN.findall(text)


def body_hash(text: str) -> str:
    """SHA-256 hash of normalised body text for deduplication."""
    normalised = re.sub(r"\s+", " ", (text or "").strip().lower())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def _decode_header(raw) -> str:
    """
    Decode an RFC 2047 encoded header to plain str (ML5).
    `msg.get()` can return an email.header.Header object or raw 8-bit text;
    make_header/decode_header turn both into a str without raising.
    """
    if raw is None:
        return ""
    try:
        return str(make_header(decode_header(str(raw)))).strip()
    except Exception:
        return str(raw).strip()


def parse_auth_results(header_value: str) -> dict:
    """
    Parse Authentication-Results header into SPF/DKIM/DMARC pass/fail/none.
    Returns dict with keys spf, dkim, dmarc -> 'pass'|'fail'|'none'.
    """
    result = {"spf": "none", "dkim": "none", "dmarc": "none"}
    if not header_value:
        return result
    for proto in ("spf", "dkim", "dmarc"):
        match = re.search(rf"{proto}\s*=\s*(\w+)", header_value, re.IGNORECASE)
        if match:
            result[proto] = normalize_auth(match.group(1))
    return result


def parse_spf_header(header_value: str) -> str:
    """Parse Received-SPF header -> 'pass'|'fail'|'none' (first token, normalised)."""
    tokens = str(header_value or "").lower().split()
    return normalize_auth(tokens[0]) if tokens else "none"


def build_headers_dict(msg) -> dict:
    """
    Extract SPF/DKIM/DMARC authentication info from an email.message.Message.
    Returns a flat dict ready for JSON serialisation.
    """
    auth_header = str(msg.get("authentication-results", "") or "")
    spf_header = str(msg.get("received-spf", "") or "")
    auth = parse_auth_results(auth_header)

    # Received-SPF takes precedence if explicit
    if spf_header:
        auth["spf"] = parse_spf_header(spf_header)

    dkim_sig = msg.get("dkim-signature", "")
    # Presence of DKIM-Signature doesn't mean it passed; keep dkim verdict as parsed
    auth["dkim_signature_present"] = bool(dkim_sig) and auth["dkim"] == "none"

    return auth


def _decode_part(part) -> str:
    """Decode one MIME part using its declared charset (ML15)."""
    payload = part.get_payload(decode=True)
    if not payload:
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


_HTML_SCRIPT_STYLE = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")


def strip_html(text: str) -> str:
    """Remove markup so models see text, not tags (ML15)."""
    text = _HTML_SCRIPT_STYLE.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    return _WHITESPACE.sub(" ", html.unescape(text)).strip()


def extract_body(msg) -> str:
    """
    Prefer text/plain parts. Fall back to HTML (tags stripped) only when no
    plain part exists. Attachments are skipped so the same content is not
    counted twice (ML15).
    """
    plain, rich = [], []
    parts = msg.walk() if msg.is_multipart() else [msg]
    for part in parts:
        if part.is_multipart() or part.get_filename():
            continue
        ctype = part.get_content_type()
        try:
            if ctype == "text/plain":
                plain.append(_decode_part(part))
            elif ctype == "text/html":
                rich.append(_decode_part(part))
        except Exception as e:
            log.debug(f"Skipping undecodable MIME part: {e}")
    if plain:
        return "\n".join(p for p in plain if p).strip()
    return strip_html("\n".join(r for r in rich if r))


def email_message_to_record(msg, source: str, label: int, record_id: str) -> dict:
    """Convert an email.message.Message into a unified record dict."""
    subject = _decode_header(msg.get("subject", ""))
    sender = _decode_header(msg.get("from", ""))
    reply_to = _decode_header(msg.get("reply-to", ""))

    body_text = extract_body(msg)
    urls = extract_urls(body_text) + extract_urls(subject)
    headers_dict = build_headers_dict(msg)

    return {
        "record_id": record_id,
        "source": source,
        "label": label,
        "subject": subject,
        "body_text": body_text,
        "urls": json.dumps(list(dict.fromkeys(urls))),
        "sender": sender,
        "reply_to": reply_to,
        "headers_available": json.dumps(headers_dict),
        "url_only": False,
        "split": "",           # assigned later
        "_body_hash": body_hash(body_text),
    }


def _safe_record(msg, source: str, label: int, record_id: str):
    """Parse one message; return None (and log) instead of aborting the file (ML5)."""
    try:
        return email_message_to_record(msg, source=source, label=label, record_id=record_id)
    except Exception as e:
        log.debug(f"Skipping unparseable message {record_id}: {e}")
        return None


def _url_only_record(record_id: str, source: str, url: str) -> dict:
    """
    URL-only phishing record. Body is empty: no label-revealing text (ML2/H2).
    The URL is kept in `urls`, so the shared text builder emits it as "URL: ...".
    """
    return {
        "record_id": record_id,
        "source": source,
        "label": 1,
        "subject": "",
        "body_text": "",
        "urls": json.dumps([url]),
        "sender": "",
        "reply_to": "",
        "headers_available": json.dumps({}),
        "url_only": True,
        "split": "",
        "_body_hash": body_hash(url),   # hash of the URL, so distinct URLs do not collapse
    }


# ---------------------------------------------------------------------------
# Dataset-specific readers
# ---------------------------------------------------------------------------

def read_enron_mbox(path: Path, label: int = 0) -> list:
    """Read Enron email dataset from mbox file (all legitimate, label=0)."""
    records = []
    try:
        mbox = mailbox.mbox(str(path))
        for i, msg in enumerate(mbox):
            rec = _safe_record(msg, "enron", label, f"enron_{path.stem}_{i:07d}")
            if rec:
                records.append(rec)
            if (i + 1) % 1000 == 0:
                log.info(f"  Enron: parsed {i+1} emails from {path.name}")
    except Exception as e:
        log.error(f"Failed to open Enron mbox {path}: {e}")
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
                rec = _safe_record(msg, "enron", label, f"enron_csv_{i:07d}")
                if rec:
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
    (no extension). ham folders -> label=0, spam folders -> label=1.
    """
    import email as email_lib
    records = []
    for i, path in enumerate(folder.iterdir()):
        if path.is_dir():
            continue
        try:
            msg = email_lib.message_from_bytes(path.read_bytes())
        except Exception as e:
            log.debug(f"Skipping {path.name}: {e}")
            continue
        rec = _safe_record(msg, "spamassassin", label, f"spamassassin_{folder.name}_{i:06d}")
        if rec:
            records.append(rec)
    log.info(f"SpamAssassin folder '{folder.name}': {len(records)} records, label={label}")
    return records


def read_nazario_mbox(path: Path) -> list:
    """Read Nazario phishing corpus in mbox format. All labelled phishing (label=1)."""
    records = []
    try:
        mbox = mailbox.mbox(str(path))
        for i, msg in enumerate(mbox):
            rec = _safe_record(msg, "nazario", 1, f"nazario_mbox_{path.stem}_{i:06d}")
            if rec:
                records.append(rec)
        log.info(f"Nazario mbox '{path.name}': {len(records)} records")
    except Exception as e:
        log.error(f"Failed to open Nazario mbox {path}: {e}")
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
            records.append({
                "record_id": f"nazario_{i:07d}",
                "source": "nazario",
                "label": 1,
                "subject": subject.strip(),
                "body_text": body_text.strip(),
                "urls": json.dumps(list(dict.fromkeys(extract_urls(body_text)))),
                "sender": sender.strip(),
                "reply_to": "",
                "headers_available": json.dumps({}),
                "url_only": False,
                "split": "",
                "_body_hash": body_hash(body_text),
            })
    except Exception as e:
        log.error(f"Failed to read Nazario CSV {path}: {e}")
    log.info(f"Nazario {path.name}: {len(records)} records")
    return records


def read_spamassassin_mbox(path: Path, label: int) -> list:
    """Read SpamAssassin mbox files. ham -> label=0, spam -> label=1."""
    records = []
    try:
        mbox = mailbox.mbox(str(path))
        for i, msg in enumerate(mbox):
            rec = _safe_record(msg, "spamassassin", label, f"spamassassin_{path.stem}_{i:07d}")
            if rec:
                records.append(rec)
    except Exception as e:
        log.error(f"Failed to open SpamAssassin mbox {path}: {e}")
    log.info(f"SpamAssassin {path.name}: {len(records)} records, label={label}")
    return records


def read_phishtank_csv(path: Path) -> list:
    """
    Read PhishTank CSV export (column 'url'). Each URL becomes a URL-only
    phishing record (label=1).
    """
    records = []
    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
        df.columns = [c.lower().strip() for c in df.columns]
        url_col = "url" if "url" in df.columns else df.columns[0]

        for i, row in df.iterrows():
            url = str(row.get(url_col, "")).strip()
            if url:
                records.append(_url_only_record(f"phishtank_{i:07d}", "phishtank", url))
    except Exception as e:
        log.error(f"Failed to read PhishTank CSV {path}: {e}")
    log.info(f"PhishTank {path.name}: {len(records)} records")
    return records


def read_openphish_txt(path: Path) -> list:
    """Read OpenPhish feed (one URL per line). label=1."""
    records = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                url = line.strip()
                if url and url.startswith("http"):
                    records.append(_url_only_record(f"openphish_{i:07d}", "openphish", url))
    except Exception as e:
        log.error(f"Failed to read OpenPhish file {path}: {e}")
    log.info(f"OpenPhish {path.name}: {len(records)} records")
    return records


# ---------------------------------------------------------------------------
# Label origin (ML16)
# ---------------------------------------------------------------------------

def add_label_origin(df: pd.DataFrame) -> pd.DataFrame:
    """
    label_origin: 'phish' (Nazario/PhishTank/OpenPhish), 'spam' (SpamAssassin spam,
    label=1), or 'ham' (label=0). Lets evaluation separate spam from phishing.
    """
    is_phish_source = df["source"].isin({"nazario", "phishtank", "openphish"})
    df["label_origin"] = np.select(
        [is_phish_source, df["label"] == 1],
        ["phish", "spam"],
        default="ham",
    )
    return df


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

def _pick_cross_source(df: pd.DataFrame):
    """
    Smallest source to hold out whole. Two conditions:
      * the held-out source contains BOTH labels (otherwise precision/FPR are undefined, ML9b);
      * the rows left for training/validation still contain BOTH labels
        (otherwise holding it out leaves the model with one class, e.g. no legitimate mail).
    """
    for source, count in df["source"].value_counts(ascending=True).items():
        held = df.loc[df["source"] == source, "label"]
        rest = df.loc[df["source"] != source, "label"]
        if set(held.unique()) == {0, 1} and set(rest.unique()) == {0, 1}:
            return source, count
    return None, 0


def assign_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Assign train / val / test_std / test_ext / test_cross.
    - test_cross: the smallest source that contains both classes is held out whole.
    - Remaining rows are split STRATIFIED by (label, source), so each stratum
      keeps its proportions in every split.
    """
    df = df.reset_index(drop=True).copy()
    df["split"] = ""
    rng = np.random.default_rng(seed)

    cross_source, cross_count = _pick_cross_source(df)
    if cross_source is None:
        log.warning("No source contains both classes; test_cross split is empty.")
    else:
        log.info(f"Cross-source holdout: '{cross_source}' ({cross_count} records)")
        df.loc[df["source"] == cross_source, "split"] = "test_cross"

    names = list(SPLIT_RATIOS.keys())
    probs = np.array([SPLIT_RATIOS[k] for k in names], dtype=float)
    probs = probs / probs.sum()

    remaining = df.index[df["split"] == ""]
    strata = df.loc[remaining].groupby(["label", "source"]).groups
    for _, idx in strata.items():
        idx = rng.permutation(np.array(list(idx)))
        n = len(idx)
        counts = np.floor(probs * n).astype(int)
        counts[0] += n - counts.sum()           # train absorbs the rounding remainder
        pos = 0
        for name, c in zip(names, counts):
            df.loc[idx[pos:pos + c], "split"] = name
            pos += c

    for (split_name, label), count in df.groupby(["split", "label"]).size().items():
        log.info(f"  Split '{split_name}' label={label}: {count}")

    return df


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_files() -> dict:
    """Discover dataset files in data/raw/ by pattern matching."""
    if not RAW_DIR.is_dir():
        raise FileNotFoundError(
            f"Raw data directory not found: {RAW_DIR}. "
            "Create it and place the datasets there (see README), or pass --raw-dir."
        )

    files = {
        "enron_mbox": [],
        "enron_csv": [],
        "nazario_csv": [],
        "nazario_mbox": [],
        "spamassassin_ham_mbox": [],
        "spamassassin_spam_mbox": [],
        "spamassassin_ham_folder": [],
        "spamassassin_spam_folder": [],
        "phishtank_csv": [],
        "openphish_txt": [],
    }

    enron_csv = RAW_DIR / "emails.csv"
    if enron_csv.exists():
        files["enron_csv"].append(enron_csv)

    for folder_name in ["easy_ham", "hard_ham"]:
        folder = RAW_DIR / folder_name
        if folder.is_dir():
            files["spamassassin_ham_folder"].append(folder)
    for folder_name in ["spam", "spam_2"]:
        folder = RAW_DIR / folder_name
        if folder.is_dir():
            files["spamassassin_spam_folder"].append(folder)

    for path in RAW_DIR.iterdir():
        if path.is_dir():
            continue
        name = path.name.lower()

        if path.suffix == ".mbox" and "enron" in name:
            files["enron_mbox"].append(path)
        elif path.suffix == ".csv" and "nazario" in name:
            files["nazario_csv"].append(path)
        elif path.suffix == ".mbox" and ("phishing" in name or "nazario" in name):
            files["nazario_mbox"].append(path)
        elif path.suffix == ".csv" and "phishtank" in name:
            files["phishtank_csv"].append(path)
        elif "openphish" in name and path.suffix in (".txt", ".feed", ""):
            files["openphish_txt"].append(path)
        elif path.suffix == ".mbox":
            if "ham" in name or "easy" in name or "hard" in name:
                files["spamassassin_ham_mbox"].append(path)
            elif "spam" in name:
                files["spamassassin_spam_mbox"].append(path)

    for key, paths in files.items():
        log.info(f"Discovered [{key}]: {len(paths)} file(s)")

    return files


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(include_url_only: bool = True):
    log.info("=" * 60)
    log.info("NETRA Data Pipeline starting")
    log.info(f"Raw data dir  : {RAW_DIR}")
    log.info(f"Output CSV    : {OUTPUT_CSV}")
    log.info(f"URL-only rows : {'included' if include_url_only else 'excluded'}")
    log.info("=" * 60)

    files = discover_files()
    all_records = []

    for path in files["enron_csv"]:
        all_records.extend(read_enron_csv(path, label=0))
    for path in files["enron_mbox"]:
        all_records.extend(read_enron_mbox(path, label=0))

    for path in files["nazario_csv"]:
        all_records.extend(read_nazario_csv(path))
    for path in files["nazario_mbox"]:
        all_records.extend(read_nazario_mbox(path))

    for folder in files["spamassassin_ham_folder"]:
        all_records.extend(read_spamassassin_eml_folder(folder, label=0))
    for folder in files["spamassassin_spam_folder"]:
        all_records.extend(read_spamassassin_eml_folder(folder, label=1))
    for path in files["spamassassin_ham_mbox"]:
        all_records.extend(read_spamassassin_mbox(path, label=0))
    for path in files["spamassassin_spam_mbox"]:
        all_records.extend(read_spamassassin_mbox(path, label=1))

    for path in files["phishtank_csv"]:
        all_records.extend(read_phishtank_csv(path))
    for path in files["openphish_txt"]:
        all_records.extend(read_openphish_txt(path))

    if not all_records:
        log.warning("No records found. Make sure datasets are placed in data/raw/")
        log.warning("See README for exact file names expected.")
        return

    log.info(f"Total records before dedup: {len(all_records)}")
    df = pd.DataFrame(all_records)

    if not include_url_only:
        df = df[~df["source"].isin(URL_ONLY_SOURCES)].reset_index(drop=True)
        log.info(f"URL-only rows excluded: {len(all_records) - len(df)}")

    # --- Conflicting duplicates: same content with different labels (ML17) ---
    label_counts = df.groupby("_body_hash")["label"].nunique()
    conflicting = label_counts[label_counts > 1].index
    if len(conflicting):
        log.warning(f"Dropping {len(conflicting)} content hashes that carry conflicting labels")
        df = df[~df["_body_hash"].isin(conflicting)].reset_index(drop=True)

    # --- Deduplicate on content hash ---
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["_body_hash"]).reset_index(drop=True)
    log.info(f"Removed {before_dedup - len(df)} duplicates. Remaining: {len(df)}")

    vc = df["label"].value_counts()
    log.info(f"Label distribution: Legitimate={vc.get(0, 0)}, Malicious={vc.get(1, 0)}")

    # --- Real header feature columns (ML1/H1): training now sees what serving sees ---
    hdr_rows = [
        extract_from_row(h, s, r)
        for h, s, r in zip(df["headers_available"], df["sender"], df["reply_to"])
    ]
    hdr_df = pd.DataFrame(hdr_rows, columns=HEADER_FEATURE_NAMES)
    df = pd.concat([df.reset_index(drop=True), hdr_df], axis=1)

    df = add_label_origin(df)
    df = assign_splits(df)
    df = df.drop(columns=["_body_hash"])

    schema_cols = [
        "record_id", "source", "label", "label_origin", "subject", "body_text",
        "urls", "sender", "reply_to", "headers_available", "url_only",
        *HEADER_FEATURE_NAMES, "split",
    ]
    df = df[schema_cols]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL)
    log.info(f"Written {len(df)} records to {OUTPUT_CSV}")
    log.info("Pipeline complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NETRA Data Pipeline")
    parser.add_argument("--raw-dir", type=str, default=str(RAW_DIR),
                        help="Path to raw data directory")
    parser.add_argument("--output", type=str, default=str(OUTPUT_CSV),
                        help="Path for output unified CSV")
    parser.add_argument("--exclude-url-only", action="store_true",
                        help="Drop PhishTank/OpenPhish URL-only rows (kept by default)")
    args = parser.parse_args()

    RAW_DIR = Path(args.raw_dir)
    OUTPUT_CSV = Path(args.output)

    run_pipeline(include_url_only=not args.exclude_url_only)
