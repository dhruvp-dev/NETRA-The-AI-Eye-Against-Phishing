"""
Feature and pipeline unit tests. No network, no model weights.
"""

import json

import numpy as np
import pandas as pd
import pytest

from ml.features.header_features import extract, extract_from_row, normalize_auth, HEADER_FEATURE_NAMES
from ml.features.text_builder import build_model_text, build_model_text_with_spans, extract_urls, MAX_URLS
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


def _legacy_build_model_text(subject, body, urls):
    """The pre-span implementation, kept to prove the training prompt did not change."""
    parts = []
    if subject and subject.strip():
        parts.append(f"Subject: {subject.strip()}")
    if body and body.strip():
        parts.append(f"Body: {body.strip()}")
    for u in dict.fromkeys(urls or []):
        parts.append(f"URL: {u}")
    return "\n".join(parts)


@pytest.mark.parametrize("subject,body,urls", [
    ("Hi", "Body text", ["https://a.example.com"]),
    ("  padded subject  ", "\n\n  body with lead  \n", []),
    ("", "only body", ["u1", "u1", "u2"]),
    (None, None, ["http://evil.example.tk/a"]),
    ("   ", "   ", []),
    ("Subject: inside", "Body: inside\nURL: inside", ["x"]),
])
def test_model_text_with_spans_matches_prompt_and_offsets(subject, body, urls):
    text, spans = build_model_text_with_spans(subject, body, urls)
    assert text == build_model_text(subject, body, urls) == _legacy_build_model_text(subject, body, urls)
    fields = {"subject": subject, "body": body}
    deduped = list(dict.fromkeys(urls))
    for sp in spans:
        source = deduped[sp.index] if sp.field == "url" else fields[sp.field]
        content = text[sp.start:sp.end]
        assert source[sp.field_offset:sp.field_offset + len(content)] == content


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


# ---------------------------------------------------------------------------
# Local-training additions: caps, near-duplicates, new readers, label origin
# ---------------------------------------------------------------------------
def test_reservoir_sample_size_and_determinism():
    import random
    import ml.data_pipeline as dp
    a = dp.reservoir_sample(range(1000), 50, random.Random(7))
    b = dp.reservoir_sample(range(1000), 50, random.Random(7))
    assert len(a) == 50 and a == b
    assert len(set(a)) == 50
    assert dp.reservoir_sample(range(10), 50, random.Random(1)) == list(range(10))


def test_reservoir_sample_is_not_first_n():
    import random
    import ml.data_pipeline as dp
    s = dp.reservoir_sample(range(10000), 100, random.Random(3))
    assert max(s) > 1000          # a 'first N' sample would max out at 99


def test_body_hash_ignores_urls_and_digits_but_not_wording():
    import ml.data_pipeline as dp
    a = dp.body_hash("Verify at http://x.example.tk/a now. Ref 12345")
    b = dp.body_hash("Verify at http://y.example.ml/zz now. Ref 99")
    c = dp.body_hash("Please call the office tomorrow")
    assert a == b
    assert a != c


def test_url_only_records_keep_distinct_hashes():
    import ml.data_pipeline as dp
    r1 = dp._url_only_record("p1", "phishtank", "http://a.example.tk/x")
    r2 = dp._url_only_record("p2", "phishtank", "http://b.example.tk/y")
    assert r1["_body_hash"] != r2["_body_hash"]


def test_read_labeled_csv_maps_labels_and_skips_unknown(tmp_path):
    import ml.data_pipeline as dp
    p = tmp_path / "CEAS_08.csv"
    p.write_text(
        "sender,subject,body,label\n"
        "a@x.com,Hi,Lunch at noon,0\n"
        "b@y.biz,Verify,Click http://evil.example.tk now,1\n"
        "c@z.net,Odd,Some text,maybe\n",
        encoding="utf-8")
    recs = dp.read_labeled_csv(p, source="ceas_08")
    assert [r["label"] for r in recs] == [0, 1]
    assert all(r["source"] == "ceas_08" for r in recs)
    assert "http://evil.example.tk" in recs[1]["urls"]


def test_read_labeled_csv_requires_body_and_label(tmp_path):
    import ml.data_pipeline as dp
    p = tmp_path / "Ling.csv"
    p.write_text("foo,bar\n1,2\n", encoding="utf-8")
    assert dp.read_labeled_csv(p, source="ling") == []


def test_read_eml_folder_recursive_finds_nested_eml(tmp_path):
    import ml.data_pipeline as dp
    (tmp_path / "2024" / "08").mkdir(parents=True)
    msg = b"From: a@b.com\nSubject: Hi\n\nBody text\n"
    (tmp_path / "2024" / "08" / "one.eml").write_bytes(msg)
    (tmp_path / "two.eml").write_bytes(msg)
    (tmp_path / "notes.txt").write_bytes(msg)
    recs = dp.read_eml_folder(tmp_path, "phishing_pot", label=1, recursive=True)
    assert len(recs) == 2
    assert all(r["label"] == 1 and r["source"] == "phishing_pot" for r in recs)


def test_label_origin_mapping_for_new_sources():
    import pandas as pd
    import ml.data_pipeline as dp
    df = pd.DataFrame([
        {"source": "phishing_pot", "label": 1},
        {"source": "ceas_08", "label": 1},
        {"source": "ceas_08", "label": 0},
        {"source": "enron", "label": 0},
        {"source": "spamassassin", "label": 1},
    ])
    assert dp.add_label_origin(df)["label_origin"].tolist() == ["phish", "phish", "ham", "ham", "spam"]


def test_cross_source_override_validates():
    import pandas as pd
    import pytest
    import ml.data_pipeline as dp
    df = pd.DataFrame(
        [{"source": "enron", "label": 0}] * 5 + [{"source": "nazario", "label": 1}] * 5
        + [{"source": "spamassassin", "label": 0}] * 3 + [{"source": "spamassassin", "label": 1}] * 3)
    assert dp._pick_cross_source(df, forced="spamassassin")[0] == "spamassassin"
    with pytest.raises(ValueError):
        dp._pick_cross_source(df, forced="enron")     # single-class holdout
    with pytest.raises(ValueError):
        dp._pick_cross_source(df, forced="missing")


def test_stratified_indices_keeps_both_classes_and_size():
    import numpy as np
    from ml.train_distilbert import stratified_indices
    labels = np.array([0] * 900 + [1] * 100)
    idx = stratified_indices(labels, 100, seed=1)
    assert abs(len(idx) - 100) <= 2
    assert set(np.unique(labels[idx])) == {0, 1}
    assert len(stratified_indices(labels, 5000, seed=1)) == 1000
