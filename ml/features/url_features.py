# -*- coding: utf-8 -*-
"""
NETRA - URL Feature Engineering v2
=====================================
Extracts per-URL signals and aggregates them per email.
Includes typosquatting detection via inline Levenshtein distance.

NOTE: URL_FEATURE_NAMES contains exactly 10 features (matching the trained RF model).
Typosquatting features are computed and exposed via extract_signals() for API response
enrichment, but NOT included in URL_FEATURE_NAMES until the model is retrained with SMOTE.

Usage:
    from ml.features.url_features import extract, URL_FEATURE_NAMES
    features_dict = extract(urls_list)
"""

import re
import logging
from urllib.parse import urlparse
from typing import List, Dict, Any, Tuple

log = logging.getLogger(__name__)

PHISHING_TLDS = {".tk", ".ml", ".ga", ".cf", ".gq", ".pw", ".top", ".xyz", ".buzz", ".click", ".info"}

BRAND_DOMAINS = [
    "paypal", "microsoft", "apple", "google", "amazon", "netflix", "facebook",
    "instagram", "twitter", "linkedin", "dropbox", "github", "chase", "wellsfargo",
    "bankofamerica", "dhl", "fedex", "ups", "steam", "discord",
]

_IP_PATTERN = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")


def _levenshtein(s1: str, s2: str) -> int:
    """
    Pure Python Levenshtein edit distance (no external library).
    Minimum single-character edits (insert, delete, replace) to transform s1 into s2.
    """
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def check_typosquatting(domain: str) -> Tuple[int, float]:
    """
    Check if a domain is a typosquatted version of a high-value brand.
    Returns (flag: int, min_normalized_distance: float).
    A normalized distance < 0.25 is flagged as likely typosquatting.
    """
    if not domain:
        return 0, 1.0

    parts      = domain.lower().split(".")
    candidates = []
    if len(parts) >= 2: candidates.append(parts[-2])
    if len(parts) >= 3: candidates.append(parts[-3])
    candidates.append(parts[0])

    expanded = list(candidates)
    for c in candidates:
        stripped = c.replace("-", "").replace("_", "")
        if stripped not in expanded:
            expanded.append(stripped)

    min_dist = 1.0
    for candidate in expanded:
        if not candidate:
            continue
        for brand in BRAND_DOMAINS:
            if abs(len(candidate) - len(brand)) > 4:
                continue
            dist = _levenshtein(candidate, brand)
            norm = dist / max(len(brand), 1)
            if norm < min_dist:
                min_dist = norm

    flag = 1 if min_dist < 0.25 else 0
    return flag, round(min_dist, 4)


def _is_ip_address(hostname: str) -> bool:
    return bool(_IP_PATTERN.match(hostname or ""))


def _subdomain_depth(hostname: str) -> int:
    if not hostname:
        return 0
    parts = hostname.split(".")
    return max(0, len(parts) - 2)


def _has_phishing_tld(hostname: str) -> bool:
    if not hostname:
        return False
    return any(hostname.lower().endswith(tld) for tld in PHISHING_TLDS)


def _count_special_chars(url: str) -> int:
    return sum(url.count(c) for c in ("@", "-", "_"))


def extract_single_url(url: str) -> Dict[str, Any]:
    """Extract features from a single URL string."""
    try:
        parsed   = urlparse(url.strip())
        hostname = parsed.hostname or ""
        scheme   = parsed.scheme.lower()
        typo_flag, typo_dist = check_typosquatting(hostname)
        return {
            "domain_length":      len(hostname),
            "subdomain_depth":    _subdomain_depth(hostname),
            "has_ip":             _is_ip_address(hostname),
            "uses_https":         scheme == "https",
            "special_char_count": _count_special_chars(url),
            "has_phishing_tld":   _has_phishing_tld(hostname),
            "typosquatting_flag": typo_flag,
            "min_brand_distance": typo_dist,
        }
    except Exception as e:
        log.debug(f"URL parse error for '{url}': {e}")
        return {
            "domain_length": 0, "subdomain_depth": 0, "has_ip": False,
            "uses_https": False, "special_char_count": 0, "has_phishing_tld": False,
            "typosquatting_flag": 0, "min_brand_distance": 1.0,
        }


def extract(urls_list: List[str]) -> Dict[str, float]:
    """
    Aggregate URL features across all URLs in a single email.
    Returns 10 features compatible with the current trained RF model.
    Typosquatting signals are computed internally and available via
    the 'any_typosquatting' and 'min_brand_distance' keys (not in URL_FEATURE_NAMES
    until model is retrained post-SMOTE).
    """
    if not urls_list:
        return {
            "url_count":            0,
            "max_domain_length":    0,
            "mean_domain_length":   0.0,
            "max_subdomain_depth":  0,
            "mean_subdomain_depth": 0.0,
            "any_ip_url":           0,
            "any_http_url":         0,
            "any_phishing_tld":     0,
            "max_special_chars":    0,
            "mean_special_chars":   0.0,
            # Extra signals — used by /predict signals dict, NOT in URL_FEATURE_NAMES
            "any_typosquatting":    0,
            "min_brand_distance":   1.0,
        }

    per_url = [extract_single_url(u) for u in urls_list if u]
    if not per_url:
        return extract([])

    domain_lengths = [f["domain_length"]       for f in per_url]
    sub_depths     = [f["subdomain_depth"]      for f in per_url]
    special_chars  = [f["special_char_count"]   for f in per_url]
    brand_dists    = [f["min_brand_distance"]   for f in per_url]

    return {
        "url_count":            len(per_url),
        "max_domain_length":    max(domain_lengths),
        "mean_domain_length":   sum(domain_lengths) / len(domain_lengths),
        "max_subdomain_depth":  max(sub_depths),
        "mean_subdomain_depth": sum(sub_depths) / len(sub_depths),
        "any_ip_url":           int(any(f["has_ip"]           for f in per_url)),
        "any_http_url":         int(any(not f["uses_https"]    for f in per_url)),
        "any_phishing_tld":     int(any(f["has_phishing_tld"] for f in per_url)),
        "max_special_chars":    max(special_chars),
        "mean_special_chars":   sum(special_chars) / len(special_chars),
        # Typosquatting — available for signals/enrichment, excluded from model features
        "any_typosquatting":    int(any(f["typosquatting_flag"] for f in per_url)),
        "min_brand_distance":   min(brand_dists),
    }


# ---------------------------------------------------------------------------
# URL_FEATURE_NAMES: EXACTLY 10 entries — must match trained RF model input dims
# These are the first 10 keys of extract([]) — typosquatting excluded until retrain
# ---------------------------------------------------------------------------
URL_FEATURE_NAMES = [
    "url_count", "max_domain_length", "mean_domain_length",
    "max_subdomain_depth", "mean_subdomain_depth",
    "any_ip_url", "any_http_url", "any_phishing_tld",
    "max_special_chars", "mean_special_chars",
]