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

import tldextract

log = logging.getLogger(__name__)

# Bundled public-suffix snapshot only: no network fetch at request time.
_TLD = tldextract.TLDExtract(suffix_list_urls=())

PHISHING_TLDS = {".tk", ".ml", ".ga", ".cf", ".gq", ".pw", ".top", ".xyz", ".buzz", ".click", ".info"}

BRAND_DOMAINS = [
    "paypal", "microsoft", "apple", "google", "amazon", "netflix", "facebook",
    "instagram", "twitter", "linkedin", "dropbox", "github", "chase", "wellsfargo",
    "bankofamerica", "dhl", "fedex", "ups", "steam", "discord",
]

# Registrable domains that legitimately belong to each brand.
BRAND_REAL_DOMAINS = {
    "paypal": {"paypal.com"},
    "microsoft": {"microsoft.com", "live.com", "outlook.com", "office.com", "office365.com", "microsoftonline.com"},
    "apple": {"apple.com", "icloud.com"},
    "google": {"google.com", "gmail.com", "googleapis.com", "googleusercontent.com"},
    "amazon": {"amazon.com"},
    "netflix": {"netflix.com"},
    "facebook": {"facebook.com", "fb.com"},
    "instagram": {"instagram.com"},
    "twitter": {"twitter.com", "x.com"},
    "linkedin": {"linkedin.com"},
    "dropbox": {"dropbox.com"},
    "github": {"github.com", "githubusercontent.com"},
    "chase": {"chase.com"},
    "wellsfargo": {"wellsfargo.com"},
    "bankofamerica": {"bankofamerica.com"},
    "dhl": {"dhl.com", "dhl.de"},
    "fedex": {"fedex.com"},
    "ups": {"ups.com"},
    "steam": {"steampowered.com", "steamcommunity.com", "steamstatic.com"},
    "discord": {"discord.com", "discordapp.com"},
}

# Known-benign registrable domains that the fuzzy matcher would otherwise flag
# (e.g. stream.com is 0.2 from "steam").
TYPO_ALLOWLIST = {"stream.com", "upstream.com"}

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


def _split_host(hostname: str):
    """Return (tldextract result, registrable domain such as 'paypal.com')."""
    ext = _TLD(hostname)
    registrable = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
    return ext, registrable


def check_typosquatting(domain: str) -> Tuple[int, float]:
    """
    Check whether a hostname imitates a high-value brand.
    Returns (flag: int, min_normalized_distance: float).

    Rules (ML6 / ML20):
      * A host whose registrable domain is a real brand domain (paypal.com,
        mail.google.com) is never flagged.
      * Any brand token that appears as an exact label or hyphen piece of a
        NON-brand registrable domain is flagged: paypal.evil.com,
        secure-paypal-login.com, apple.com.evil.ru.
      * Otherwise a fuzzy match (0 < normalised distance < 0.25) is flagged,
        e.g. paypa1-secure.tk.
    """
    host = (domain or "").lower().strip(".")
    if not host:
        return 0, 1.0

    ext, registrable = _split_host(host)
    if registrable in TYPO_ALLOWLIST:
        return 0, 0.0
    if ext.domain in BRAND_REAL_DOMAINS and registrable in BRAND_REAL_DOMAINS[ext.domain]:
        return 0, 0.0

    # Brand candidates: registrable label, subdomain labels, and their hyphen/underscore pieces.
    tokens = set()
    for label in [ext.domain] + ext.subdomain.split("."):
        if not label:
            continue
        tokens.add(label)
        tokens.add(label.replace("-", "").replace("_", ""))
        tokens.update(t for t in re.split(r"[-_]", label) if t)
    tokens.discard("")

    exact = any(tok == brand for tok in tokens for brand in BRAND_DOMAINS)

    min_dist = 0.0 if exact else 1.0
    if not exact:
        for candidate in tokens:
            for brand in BRAND_DOMAINS:
                if abs(len(candidate) - len(brand)) > 4:
                    continue
                norm = _levenshtein(candidate, brand) / max(len(brand), 1)
                if norm < min_dist:
                    min_dist = norm

    fuzzy = 0.0 < min_dist < 0.25
    flag = 1 if (exact or fuzzy) else 0
    return flag, round(min_dist, 4)


def _is_ip_address(hostname: str) -> bool:
    return bool(_IP_PATTERN.match(hostname or ""))


def _subdomain_depth(hostname: str) -> int:
    """Number of labels left of the registrable domain (public-suffix aware, ML20)."""
    if not hostname:
        return 0
    ext = _TLD(hostname.lower())
    return len(ext.subdomain.split(".")) if ext.subdomain else 0


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