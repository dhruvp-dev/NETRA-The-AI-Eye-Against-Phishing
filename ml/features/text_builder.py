"""
NETRA - Shared model-input builder
==================================
Single source of truth for the text the DistilBERT Tier-1 model sees.
Training (ml/train_distilbert.py, notebooks) and serving (api/main.py) must
both call build_model_text() so the two prompts can never drift apart.

Usage:
    from ml.features.text_builder import build_model_text, extract_urls
    urls = extract_urls(subject, body, explicit_urls)
    text = build_model_text(subject, body, urls)
"""

import re
from typing import Iterable, List, Optional

# Matches http(s) and bare www. URLs in free text.
URL_PATTERN = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')

# Hard cap so a single email cannot blow up the prompt length.
MAX_URLS = 20


def extract_urls(
    subject: Optional[str],
    body: Optional[str],
    explicit_urls: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Merge caller-supplied URLs with URLs regex-extracted from subject and body.
    Order-preserving and deduplicated (dict.fromkeys), so the same input always
    yields the same list in every process.
    """
    found: List[str] = []
    for u in explicit_urls or []:
        if u and u.strip():
            found.append(u.strip())
    for source in (subject, body):
        if source:
            found.extend(URL_PATTERN.findall(source))
    return list(dict.fromkeys(found))[:MAX_URLS]


def build_model_text(subject: Optional[str], body: Optional[str], urls: Iterable[str]) -> str:
    """
    Build the Tier-1 model input.
    Neutral labels only: no label-revealing words such as "Phishing URL".
    """
    parts: List[str] = []
    if subject and subject.strip():
        parts.append(f"Subject: {subject.strip()}")
    if body and body.strip():
        parts.append(f"Body: {body.strip()}")
    for u in dict.fromkeys(urls or []):
        parts.append(f"URL: {u}")
    return "\n".join(parts)
