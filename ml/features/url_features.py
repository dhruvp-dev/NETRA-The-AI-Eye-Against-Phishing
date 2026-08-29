"""
NETRA - URL Feature Engineering
================================
Extracts per-URL signals and aggregates them per email.

Usage:
    from ml.features.url_features import extract
    features_dict = extract(urls_list)  # urls_list is a Python list of URL strings
"""

import re
import math
import logging
from urllib.parse import urlparse
from typing import List, Dict, Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known phishing-associated TLDs
# ---------------------------------------------------------------------------
PHISHING_TLDS = {".tk", ".ml", ".ga", ".cf", ".gq", ".pw", ".top", ".xyz", ".buzz"}

# ---------------------------------------------------------------------------
# IP-in-URL detection (IPv4)
# ---------------------------------------------------------------------------
_IP_PATTERN = re.compile(
    r"^(\d{1,3}\.){3}\d{1,3}$"
)


def _is_ip_address(hostname: str) -> bool:
    """Return True if hostname is a raw IPv4 address."""
    return bool(_IP_PATTERN.match(hostname or ""))


def _subdomain_depth(hostname: str) -> int:
    """
    Count subdomain depth.
    e.g. 'mail.accounts.google.com' → 2 (two subdomains above registered domain)
    We use a simple heuristic: depth = number of dots minus 1 (assuming TLD+1).
    """
    if not hostname:
        return 0
    parts = hostname.split(".")
    # subtract 2 for the registered domain (SLD + TLD)
    return max(0, len(parts) - 2)


def _has_phishing_tld(hostname: str) -> bool:
    """Return True if the domain ends with a known phishing TLD."""
    if not hostname:
        return False
    for tld in PHISHING_TLDS:
        if hostname.lower().endswith(tld):
            return True
    return False


def _count_special_chars(url: str) -> int:
    """Count phishing-signal special characters: @, -, _"""
    return sum(url.count(c) for c in ("@", "-", "_"))


# ---------------------------------------------------------------------------
# Per-URL feature extraction
# ---------------------------------------------------------------------------

def extract_single_url(url: str) -> Dict[str, Any]:
    """
    Extract features from a single URL string.

    Returns:
        dict with keys:
            domain_length, subdomain_depth, has_ip, uses_https,
            special_char_count, has_phishing_tld
    """
    try:
        parsed = urlparse(url.strip())
        hostname = parsed.hostname or ""
        scheme = parsed.scheme.lower()

        return {
            "domain_length": len(hostname),
            "subdomain_depth": _subdomain_depth(hostname),
            "has_ip": _is_ip_address(hostname),
            "uses_https": scheme == "https",
            "special_char_count": _count_special_chars(url),
            "has_phishing_tld": _has_phishing_tld(hostname),
        }
    except Exception as e:
        log.debug(f"URL parse error for '{url}': {e}")
        return {
            "domain_length": 0,
            "subdomain_depth": 0,
            "has_ip": False,
            "uses_https": False,
            "special_char_count": 0,
            "has_phishing_tld": False,
        }


# ---------------------------------------------------------------------------
# Per-email aggregation
# ---------------------------------------------------------------------------

def extract(urls_list: List[str]) -> Dict[str, float]:
    """
    Aggregate URL features across all URLs in a single email.

    Args:
        urls_list: Python list of URL strings (from the 'urls' column after json.loads)

    Returns:
        Flat feature dict:
            url_count, max_domain_length, mean_domain_length,
            max_subdomain_depth, mean_subdomain_depth,
            any_ip_url, any_http_url, any_phishing_tld,
            max_special_chars, mean_special_chars
    """
    # Default (zero-valued) features for emails with no URLs
    if not urls_list:
        return {
            "url_count": 0,
            "max_domain_length": 0,
            "mean_domain_length": 0.0,
            "max_subdomain_depth": 0,
            "mean_subdomain_depth": 0.0,
            "any_ip_url": 0,
            "any_http_url": 0,        # 1 if ANY URL uses plain HTTP (not HTTPS)
            "any_phishing_tld": 0,
            "max_special_chars": 0,
            "mean_special_chars": 0.0,
        }

    per_url = [extract_single_url(u) for u in urls_list if u]

    if not per_url:
        return extract([])  # recurse with empty list for defaults

    domain_lengths = [f["domain_length"] for f in per_url]
    subdomain_depths = [f["subdomain_depth"] for f in per_url]
    special_chars = [f["special_char_count"] for f in per_url]
    any_ip = int(any(f["has_ip"] for f in per_url))
    # any_http_url = 1 if any URL uses plain HTTP (no HTTPS) — phishing signal
    any_http = int(any(not f["uses_https"] for f in per_url))
    any_phishing_tld = int(any(f["has_phishing_tld"] for f in per_url))

    return {
        "url_count": len(per_url),
        "max_domain_length": max(domain_lengths),
        "mean_domain_length": sum(domain_lengths) / len(domain_lengths),
        "max_subdomain_depth": max(subdomain_depths),
        "mean_subdomain_depth": sum(subdomain_depths) / len(subdomain_depths),
        "any_ip_url": any_ip,
        "any_http_url": any_http,
        "any_phishing_tld": any_phishing_tld,
        "max_special_chars": max(special_chars),
        "mean_special_chars": sum(special_chars) / len(special_chars),
    }


# ---------------------------------------------------------------------------
# Feature names (for logging / inspection)
# ---------------------------------------------------------------------------

URL_FEATURE_NAMES = list(extract([]).keys())
