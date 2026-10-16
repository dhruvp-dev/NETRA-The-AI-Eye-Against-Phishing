"""
Tests for the post-model rule layer added after the Tier-1 audit (bugs/tier1_model_bugs.md):
trusted-sender downgrade (api/trust.py), new threat signals, windowed scoring of long emails and
the SUSPICIOUS band guard. No model weights are needed.
"""

import numpy as np
import pandas as pd
import pytest

import api.main as main
from api.trust import assess


@pytest.fixture(autouse=True)
def thresholds(monkeypatch):
    monkeypatch.setattr(main.state, "phishing_threshold", 0.25)
    monkeypatch.setattr(main.state, "suspicious_lower", 0.08)


# ---------------------------------------------------------------------------
# Trust assessment
# ---------------------------------------------------------------------------
def test_known_brand_with_own_links_is_trusted():
    t = assess("Amazon <shipment-tracking@amazon.com>", None, None,
               ["https://www.amazon.com/gp/your-account/order-details", "https://m.media-amazon.com/x.png"])
    assert t.level == "known" and t.aligned


def test_known_brand_without_links_is_trusted():
    assert assess("orders@target.com", None, {"spf": "pass", "dkim": "pass", "dmarc": "pass"}, []).level == "known"


def test_one_off_domain_link_breaks_alignment():
    t = assess("security@paypal.com", None, None,
               ["https://www.paypal.com/myaccount", "https://paypal-login-check.com/verify"])
    assert t.level is None and any("paypal-login-check.com" in b or "suspicious" in b for b in t.blockers)


@pytest.mark.parametrize("url", [
    "https://bit.ly/3abcd",                      # shortener
    "http://192.168.1.10/login",                 # IP
    "https://docs.google.com/forms/d/e/x/view",  # user-content host on a known domain
    "https://secure-update.xyz/login",           # high-risk TLD
])
def test_risky_links_are_never_aligned(url):
    assert assess("no-reply@google.com", None, {"dmarc": "pass"}, [url]).level is None


def test_freemail_sender_is_never_trusted():
    assert assess("someone@gmail.com", None, {"dmarc": "pass"}, []).level is None


def test_auth_failure_blocks_trust():
    assert assess("orders@target.com", None, {"spf": "fail"}, []).level is None


def test_lookalike_sender_blocks_trust():
    assert assess("billing@netf1ix.com", None, {"dmarc": "pass"}, []).level is None


def test_reply_to_elsewhere_blocks_trust():
    assert assess("ceo@stripe.com", "ceo.private@protonmail.com", None, []).level is None


def test_authenticated_unknown_domain_is_only_verified():
    t = assess("news@small-bakery-shop.com", None, {"dmarc": "pass"}, ["https://small-bakery-shop.com/menu"])
    assert t.level == "verified" and not t.known_sender


def test_unknown_domain_without_auth_is_not_trusted():
    assert assess("benefits@portal-employee-update.com", None, None, []).level is None


# ---------------------------------------------------------------------------
# Decision with trust
# ---------------------------------------------------------------------------
def _decide(score, sender, urls=(), headers=None, body="Your order has shipped.", subject="Order"):
    req = main.PredictRequest(subject=subject, body_text=body, sender=sender, urls=list(urls), headers=headers)
    all_urls = main.extract_urls(req.subject, req.body_text, req.urls)
    return main._verdict(req, all_urls, main._extract_signals(req, all_urls), score, max(score, 1 - score))


def test_known_sender_downgrades_phishing_to_legitimate():
    r = _decide(0.997, "shipment-tracking@amazon.com", ["https://www.amazon.com/orders"])
    assert r["classification"] == "LEGITIMATE" and r["adjustment"] == "trusted_sender"
    assert r["risk_score"] < 0.08 and r["model_score"] == pytest.approx(0.997)


def test_verified_unknown_sender_only_drops_one_level():
    r = _decide(0.99, "news@small-bakery-shop.com", ["https://small-bakery-shop.com/menu"], {"dmarc": "pass"})
    assert r["classification"] == "SUSPICIOUS" and r["adjustment"] == "trusted_sender"


def test_bec_lure_from_known_domain_is_not_downgraded():
    body = "Are you available? I need you to buy gift cards for a client today. Keep this confidential."
    r = _decide(0.0, "ceo@stripe.com", body=body, subject="Quick favor")
    assert r["adjustment"] == "signal_override" and r["classification"] == "SUSPICIOUS"


def test_untrusted_sender_keeps_model_verdict():
    r = _decide(0.95, "alerts@account-review-center.com", ["https://account-review-center.com/login"])
    assert r["classification"] == "PHISHING" and r["adjustment"] is None


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------
def _signals(body="", subject="", urls=()):
    req = main.PredictRequest(subject=subject, body_text=body, urls=list(urls))
    return main._extract_signals(req, main.extract_urls(req.subject, req.body_text, req.urls))


def test_plain_http_link_is_not_counted():
    s = _signals("Read our blog", urls=["http://blog.example.com/post", "http://news.example.org/a"])
    assert s["suspicious_urls"] == 1 and main._signal_count(s, main.url_extract(
        ["http://blog.example.com/post", "http://news.example.org/a"])) == 0


def test_shortener_and_executable_links_are_counted():
    urls = ["https://bit.ly/inv-2291", "http://files.example.com/invoice.exe"]
    s = _signals("See invoice", urls=urls)
    assert s["shortened_url_detected"] and s["executable_link"]
    assert main._signal_count(s, main.url_extract(urls)) == 3          # shortener 1 + program link 2
    s = _signals("Files", urls=["https://files.example.com/photos.zip"])
    assert s["risky_file_link"] and main._signal_count(s, main.url_extract(["https://files.example.com/photos.zip"])) == 1


@pytest.mark.parametrize("body, expected", [
    ("Can you process a wire transfer of $48,000 today? Keep this confidential, I'm in a meeting.", True),
    ("Please update our bank account details for the next invoice payment. This is urgent.", True),
    ("Your payment of $12.00 was received. Thanks for your business.", False),
    ("Bitcoin prices rose today according to market data.", False),
])
def test_bec_detection(body, expected):
    assert _signals(body)["bec_detected"] is expected


# ---------------------------------------------------------------------------
# Windowed scoring (B3)
# ---------------------------------------------------------------------------
class _WordTok:
    """One token per whitespace word, plus [CLS]/[SEP]."""

    def __call__(self, text, **kw):
        import torch
        texts = text if isinstance(text, list) else [text]
        if kw.get("truncation") is False:
            return {"input_ids": list(range(len(texts[0].split()) + 2))}
        n = kw.get("max_length", 8)
        ids = torch.zeros((len(texts), n), dtype=torch.long)
        self.seen = texts
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class _KeywordModel:
    """Phishing logit high when the window text contains 'EVIL'."""

    def __init__(self, tok):
        self.tok = tok

    def __call__(self, ids, mask, hdr):
        import torch
        rows = [[0.0, 6.0] if "EVIL" in t else [6.0, 0.0] for t in self.tok.seen]
        return torch.tensor(rows)


def _setup_windows(monkeypatch, max_length=64):
    tok = _WordTok()
    monkeypatch.setattr(main.state, "db_tokenizer", tok)
    monkeypatch.setattr(main.state, "db_model", _KeywordModel(tok))
    monkeypatch.setattr(main.state, "db_config", {"max_length": max_length, "text_format": "v3",
                                                  "header_features_trained": True})


def test_short_email_is_one_window(monkeypatch):
    _setup_windows(monkeypatch)
    req = main.PredictRequest(subject="Hi", body_text="Short note EVIL")
    assert len(main._model_windows(req, [])) == 1
    assert main._score_distilbert(req, [])[0] > 0.99


def test_padded_phishing_is_found_in_a_later_window(monkeypatch):
    _setup_windows(monkeypatch)
    padding = "The weather was pleasant and the meeting went well. " * 60
    req = main.PredictRequest(subject="Notes", body_text=padding + "Click here EVIL to verify your password.")
    windows = main._model_windows(req, [])
    assert 1 < len(windows) <= 1 + main.MAX_EXTRA_WINDOWS
    assert "EVIL" in windows[-1][0]                  # the tail is always scored
    assert main._score_distilbert(req, [])[0] > 0.99


def test_window_spans_point_into_the_original_body(monkeypatch):
    _setup_windows(monkeypatch)
    body = "word " * 400 + "TARGET end"
    req = main.PredictRequest(subject="S", body_text=body, urls=["https://a.example.com"])
    text, spans = main._model_windows(req, ["https://a.example.com"])[-1]
    sp = next(s for s in spans if s.field == "body")
    local = text.index("TARGET", sp.start)
    assert body[local - sp.start + sp.field_offset:].startswith("TARGET")
    assert text.rstrip().endswith("URL: https://a.example.com")   # URL lines repeat in each window


def test_body_chunks_cover_the_whole_body():
    body = " ".join(f"w{i}" for i in range(1000))
    chunks = main._body_chunks(body)
    assert chunks[0][0] == 0 and chunks[-1][1].endswith("w999")
    for (s1, c1), (s2, _c2) in zip(chunks, chunks[1:]):
        assert s2 <= s1 + len(c1)                      # no gaps


# ---------------------------------------------------------------------------
# Threshold band guard (B2)
# ---------------------------------------------------------------------------
def test_degenerate_band_is_rejected():
    assert not main._valid_thresholds(0.02456, 0.02481)
    assert main._valid_thresholds(0.08, 0.25)


def test_calibration_widens_narrow_band_and_flags_plateau():
    from ml.calibrate_threshold import MIN_SUSPICIOUS_BAND, pick_thresholds, saturated, sweep_thresholds
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(500), np.ones(500)]
    p = np.r_[rng.uniform(0, 0.001, 500), rng.uniform(0.999, 1.0, 500)]   # saturated scores
    df = sweep_thresholds(y, p)
    _best, phishing, lower = pick_thresholds(df, 0.95)
    assert phishing - lower >= MIN_SUSPICIOUS_BAND - 1e-9
    assert saturated(df)
    assert not saturated(pd.DataFrame({"threshold": [0.1, 0.3, 0.5], "f1": [0.80, 0.90, 0.70]}))
