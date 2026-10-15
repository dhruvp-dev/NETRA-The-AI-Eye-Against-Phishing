"""
Feature and pipeline unit tests. No network, no model weights.
"""

import json

import numpy as np
import pandas as pd
import pytest

from ml.features.header_features import extract, extract_from_row, normalize_auth, HEADER_FEATURE_NAMES
from ml.features.text_builder import build_model_text, extract_urls, MAX_URLS
from ml.features.url_features import check_typosquatting, _subdomain_depth


# ---------------------------------------------------------------------------
# Typosquatting (ML6 / ML20)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("host", [
    "paypal.evil.com",          # brand as subdomain label
    "paypal-secure.tk",         # brand as hyphen piece
    "secure-paypal-login.com",  # brand inside a longer registrable domain
    "apple.com.evil.ru",        # brand-looking subdomain chain
    "paypa1-secure.tk",         # fuzzy (leet) variant
])
def test_typosquat_flagged(host):
    assert check_typosquatting(host)[0] == 1


@pytest.mark.parametrize("host", [
    "paypal.com",
    "mail.google.com",
    "www.microsoft.com",
    "stream.com",               # allow-listed near-miss of "steam"
    "example.com",
])
def test_legitimate_not_flagged(host):
    assert check_typosquatting(host)[0] == 0


def test_subdomain_depth_is_public_suffix_aware():
    assert _subdomain_depth("example.co.uk") == 0
    assert _subdomain_depth("a.b.example.com") == 2


# ---------------------------------------------------------------------------
# Header features (A20 / ML14)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [
    ("pass", "pass"), ("fail", "fail"), ("softfail", "fail"),
    ("neutral", "none"), ("temperror", "none"), ("permerror", "none"), ("", "none"), (None, "none"),
])
def test_normalize_auth(raw, expected):
    assert normalize_auth(raw) == expected


def test_sender_domain_parsed_from_display_name():
    feats = extract({}, sender='"Support @ Bank" <help@paypal.com>', reply_to="x@evil.example")
    assert feats["sender_domain_matches_reply_to"] == 0


def test_header_feature_vector_has_ten_names():
    assert len(HEADER_FEATURE_NAMES) == 10


def test_extract_from_row_handles_bad_json():
    feats = extract_from_row("not-json", sender="a@b.com", reply_to="")
    assert feats["spf_none"] == 1


def test_extract_from_row_matches_dict_input():
    raw = {"spf": "pass", "dkim": "fail", "dmarc": "none"}
    assert extract_from_row(json.dumps(raw), "a@x.com", "a@x.com") == extract(raw, "a@x.com", "a@x.com")


# ---------------------------------------------------------------------------
# Shared text builder (ML2 / H2)
# ---------------------------------------------------------------------------
def test_model_text_has_no_label_revealing_prefix():
    text = build_model_text("Hi", "Body", ["https://a.example.com"])
    assert "Phishing URL" not in text
    assert text == "Subject: Hi\nBody: Body\nURL: https://a.example.com"


def test_model_text_is_deterministic_and_dedupes_urls():
    urls = ["https://a.example.com", "https://a.example.com", "https://b.example.com"]
    assert build_model_text("", "x", urls) == build_model_text("", "x", urls)
    assert build_model_text("", "x", urls).count("URL:") == 2


def test_url_only_record_text_is_url_line_only():
    assert build_model_text("", "", ["http://evil.example.tk/a"]) == "URL: http://evil.example.tk/a"


def test_extract_urls_order_preserving_and_capped():
    body = " ".join(f"https://site{i}.example.com" for i in range(MAX_URLS + 5))
    urls = extract_urls("", body, ["https://first.example.com"])
    assert urls[0] == "https://first.example.com"
    assert len(urls) == MAX_URLS
    assert len(set(urls)) == len(urls)


# ---------------------------------------------------------------------------
# Pipeline parsing (ML5 / ML15 / ML17 / ML9)
# ---------------------------------------------------------------------------
def test_decode_header_rfc2047():
    import ml.data_pipeline as dp
    assert dp._decode_header("=?utf-8?q?Caf=C3=A9?=") == "Café"


def test_decode_header_plain_text_passthrough():
    import ml.data_pipeline as dp
    assert dp._decode_header("Weekly report") == "Weekly report"


def test_strip_html_removes_markup_and_scripts():
    import ml.data_pipeline as dp
    html = "<html><style>p{}</style><script>alert(1)</script><p>Hello &amp; welcome</p></html>"
    assert dp.strip_html(html) == "Hello & welcome"


def test_parse_auth_results_uses_shared_normaliser():
    import ml.data_pipeline as dp
    out = dp.parse_auth_results("mx.example; spf=softfail smtp.mailfrom=x; dkim=pass; dmarc=temperror")
    assert out == {"spf": "fail", "dkim": "pass", "dmarc": "none"}


def test_url_only_records_have_no_body_and_no_leak():
    import ml.data_pipeline as dp
    rec = dp._url_only_record("phishtank_0", "phishtank", "http://x.example.tk")
    assert rec["body_text"] == ""
    assert rec["url_only"] is True
    assert "Phishing URL" not in json.dumps(rec)


def test_assign_splits_is_stratified_and_holds_out_mixed_source():
    import ml.data_pipeline as dp
    rows = []
    for i in range(400):
        rows.append({"source": "enron", "label": 0, "record_id": f"e{i}"})
        rows.append({"source": "nazario", "label": 1, "record_id": f"n{i}"})
    for i in range(60):
        rows.append({"source": "spamassassin", "label": 0, "record_id": f"sh{i}"})
        rows.append({"source": "spamassassin", "label": 1, "record_id": f"ss{i}"})
    df = pd.DataFrame(rows)

    out = dp.assign_splits(df, seed=1)
    assert set(out["split"]) == {"train", "val", "test_std", "test_ext", "test_cross"}
    cross = out[out["split"] == "test_cross"]
    assert set(cross["label"].unique()) == {0, 1}          # holdout must contain both classes
    assert set(cross["source"]) == {"spamassassin"}
    for name in ("train", "val", "test_std"):
        part = out[out["split"] == name]
        assert set(part["label"].unique()) == {0, 1}       # stratified: both labels in every split


def test_label_origin_separates_spam_from_phish():
    import ml.data_pipeline as dp
    df = pd.DataFrame([
        {"source": "nazario", "label": 1},
        {"source": "spamassassin", "label": 1},
        {"source": "spamassassin", "label": 0},
    ])
    assert dp.add_label_origin(df)["label_origin"].tolist() == ["phish", "spam", "ham"]


# ---------------------------------------------------------------------------
# Threshold sweep (ML7 / ML11)
# ---------------------------------------------------------------------------
def test_sweep_matches_bruteforce():
    from ml.calibrate_threshold import sweep_thresholds
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.25, 300), 0, 1)
    df = sweep_thresholds(y, p)
    for _, row in df.sample(10, random_state=0).iterrows():
        pred = (p >= row["threshold"]).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        assert row["recall"] == pytest.approx(tp / (y == 1).sum(), abs=1e-3)
        assert row["precision"] == pytest.approx(tp / max(tp + fp, 1), abs=1e-3)


# ---------------------------------------------------------------------------
# Rate limiter memory bounds (A4)
# ---------------------------------------------------------------------------
def test_rate_limiter_evicts_idle_keys_and_caps_size():
    from api.rate_limiter import SlidingWindowRateLimiter
    now = [0.0]
    lim = SlidingWindowRateLimiter(5, 60, max_keys=10, sweep_interval=1, clock=lambda: now[0])
    for i in range(50):
        lim.check(f"ip{i}")
        now[0] += 0.1
    assert len(lim) <= 10
    now[0] += 1000                      # every window is empty now
    lim.check("fresh")
    assert len(lim) == 1


def test_rate_limiter_retry_after():
    from api.rate_limiter import SlidingWindowRateLimiter
    now = [0.0]
    lim = SlidingWindowRateLimiter(2, 60, clock=lambda: now[0])
    assert lim.check("a") == (True, 0)
    assert lim.check("a") == (True, 0)
    allowed, retry = lim.check("a")
    assert allowed is False and retry >= 60
