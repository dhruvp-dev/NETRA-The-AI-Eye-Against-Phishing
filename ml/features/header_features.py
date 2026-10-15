"""
NETRA - Header Feature Engineering
====================================
Extracts SPF / DKIM / DMARC authentication signals and
sender/reply-to mismatch flag from email headers.

Usage:
    from ml.features.header_features import extract
    features = extract(headers_dict, sender="foo@bar.com", reply_to="other@domain.com")
"""

import json
import re
import logging
from email.utils import parseaddr
from typing import Dict, Any, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# One-hot encoding helpers
# ---------------------------------------------------------------------------

_AUTH_STATES = ["pass", "fail", "none"]

# One normaliser for every auth-result spelling (ML14). softfail counts as fail
# everywhere (training parsers and inference). neutral/temperror/permerror carry
# no usable pass/fail signal, so they map to none.
_AUTH_ALIASES = {
    "softfail": "fail",
    "neutral": "none",
    "temperror": "none",
    "permerror": "none",
}


def normalize_auth(value) -> str:
    """Map any SPF/DKIM/DMARC result string to 'pass' | 'fail' | 'none'."""
    v = str(value or "none").lower().strip()
    v = _AUTH_ALIASES.get(v, v)
    return v if v in _AUTH_STATES else "none"


def _one_hot_auth(value: str) -> Dict[str, int]:
    """
    Convert an auth value ('pass'|'fail'|'none'|other) to 3-dim one-hot dict.
    Unknown values map to 'none'.
    """
    value = normalize_auth(value)
    return {state: int(value == state) for state in _AUTH_STATES}


# ---------------------------------------------------------------------------
# Domain extraction
# ---------------------------------------------------------------------------

_EMAIL_DOMAIN_RE = re.compile(r"@([\w.\-]+)")


def _extract_domain(address) -> Optional[str]:
    """
    Extract the domain part from an address. Handles 'Name <a@b.com>' via
    email.utils.parseaddr so display names containing '@' or '>' cannot
    confuse the domain (A20).
    """
    if address is None:
        return None
    address_str = str(address).strip()
    if not address_str or address_str.lower() == "nan":
        return None
    _, addr = parseaddr(address_str)
    if "@" in addr:
        domain = addr.rsplit("@", 1)[1].strip().strip(">").lower()
        return domain or None
    match = _EMAIL_DOMAIN_RE.search(address_str)
    return match.group(1).lower() if match else None


# ---------------------------------------------------------------------------
# Main feature extractor
# ---------------------------------------------------------------------------

def extract(
    headers_dict: Dict[str, Any],
    sender: str = "",
    reply_to: str = "",
) -> Dict[str, int]:
    """
    Extract email header authentication features.

    Args:
        headers_dict:   Dict from headers_available column (after json.loads).
                        Expected keys: 'spf', 'dkim', 'dmarc' → 'pass'|'fail'|'none'
        sender:         From header string (e.g. "Name <addr@domain.com>")
        reply_to:       Reply-To header string

    Returns:
        Flat dict with 10 integer features:
            spf_pass, spf_fail, spf_none           (one-hot, 3 dims)
            dkim_pass, dkim_fail, dkim_none         (one-hot, 3 dims)
            dmarc_pass, dmarc_fail, dmarc_none      (one-hot, 3 dims)
            sender_domain_matches_reply_to          (bool int, 1 dim)
    """
    if not isinstance(headers_dict, dict):
        headers_dict = {}

    # --- SPF ---
    spf_value = headers_dict.get("spf", "none")
    spf_oh = _one_hot_auth(spf_value)

    # --- DKIM ---
    dkim_value = headers_dict.get("dkim", "none")
    dkim_oh = _one_hot_auth(dkim_value)

    # --- DMARC ---
    dmarc_value = headers_dict.get("dmarc", "none")
    dmarc_oh = _one_hot_auth(dmarc_value)

    # --- Sender / Reply-To domain mismatch ---
    sender_domain = _extract_domain(sender or "")
    reply_to_domain = _extract_domain(reply_to or "")

    if sender_domain and reply_to_domain:
        domain_match = int(sender_domain == reply_to_domain)
    else:
        # If reply-to is absent, no mismatch detected (treat as match)
        domain_match = 1

    return {
        # SPF one-hot
        "spf_pass": spf_oh["pass"],
        "spf_fail": spf_oh["fail"],
        "spf_none": spf_oh["none"],
        # DKIM one-hot
        "dkim_pass": dkim_oh["pass"],
        "dkim_fail": dkim_oh["fail"],
        "dkim_none": dkim_oh["none"],
        # DMARC one-hot
        "dmarc_pass": dmarc_oh["pass"],
        "dmarc_fail": dmarc_oh["fail"],
        "dmarc_none": dmarc_oh["none"],
        # Sender ↔ Reply-To domain match flag
        "sender_domain_matches_reply_to": domain_match,
    }


# ---------------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------------

HEADER_FEATURE_NAMES = list(
    extract({}, sender="a@x.com", reply_to="a@x.com").keys()
)


def extract_from_row(headers_json, sender="", reply_to="") -> Dict[str, int]:
    """
    Training helper: take the raw `headers_available` JSON string from a
    unified.csv row (or an already-parsed dict) and return the 10 features.
    Malformed JSON yields an all-'none' header block instead of raising.
    """
    if isinstance(headers_json, dict):
        headers = headers_json
    else:
        try:
            headers = json.loads(headers_json) if headers_json else {}
        except (TypeError, ValueError):
            headers = {}
    return extract(headers, sender=str(sender or ""), reply_to=str(reply_to or ""))
