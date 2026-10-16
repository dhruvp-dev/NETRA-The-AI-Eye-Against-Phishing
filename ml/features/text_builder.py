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
    text, spans = build_model_text_with_spans(subject, body, urls)   # for explanations
"""

import re
from typing import Iterable, List, NamedTuple, Optional, Tuple

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


class Span(NamedTuple):
    """Where one request field's content sits inside the model text."""
    field: str            # "subject" | "body" | "url"
    index: Optional[int]  # position in the URL list for "url", else None
    start: int            # char range of the content in the model text
    end: int
    field_offset: int     # char offset of that content inside the original field


def build_model_text_with_spans(
    subject: Optional[str], body: Optional[str], urls: Iterable[str]
) -> Tuple[str, List[Span]]:
    """
    Build the Tier-1 model input and record where each field's content lies in it.
    Neutral labels only: no label-revealing words such as "Phishing URL".
    """
    parts: List[str] = []
    spans: List[Span] = []
    pos = 0

    def add(label: str, field: str, index: Optional[int], content: str, field_offset: int) -> None:
        nonlocal pos
        if parts:
            pos += 1  # "\n" separator
        start = pos + len(label)
        parts.append(label + content)
        spans.append(Span(field, index, start, start + len(content), field_offset))
        pos = start + len(content)

    def lead(raw: str) -> int:
        return len(raw) - len(raw.lstrip())

    if subject and subject.strip():
        add("Subject: ", "subject", None, subject.strip(), lead(subject))
    if body and body.strip():
        add("Body: ", "body", None, body.strip(), lead(body))
    for i, u in enumerate(dict.fromkeys(urls or [])):
        add("URL: ", "url", i, u, 0)
    return "\n".join(parts), spans


def build_model_text(subject: Optional[str], body: Optional[str], urls: Iterable[str]) -> str:
    """Build the Tier-1 model input (see build_model_text_with_spans)."""
    return build_model_text_with_spans(subject, body, urls)[0]
