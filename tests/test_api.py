"""
API tests (FastAPI TestClient). No model weights are loaded: the scoring
functions are monkeypatched, so these tests run without GPU or downloads.
"""

import pytest
from fastapi.testclient import TestClient

import api.main as main
from api.rate_limiter import SlidingWindowRateLimiter


@pytest.fixture(autouse=True)
def reset_state(monkeypatch, tmp_path):
    """Fresh model state, fresh rate limiter and a temp threshold file for each test."""
    monkeypatch.setattr(main.state, "db_ready", False)
    monkeypatch.setattr(main.state, "rf_ready", False)
    monkeypatch.setattr(main.state, "phishing_threshold", 0.25)
    monkeypatch.setattr(main.state, "suspicious_lower", 0.08)
    monkeypatch.setattr(main.state, "suspicious_upper", 0.25)
    monkeypatch.setattr(main.state, "threshold_source", "test")
    monkeypatch.setattr(main.state, "inference_slots", None)
    monkeypatch.setattr(main.state, "explain_slots", None)
    monkeypatch.setattr(main, "_limiter", SlidingWindowRateLimiter(20, 60))
    monkeypatch.setattr(main, "_explain_limiter", SlidingWindowRateLimiter(5, 60))
    monkeypatch.setattr(main, "THRESHOLD_CONFIG_PATH", tmp_path / "threshold_config.json")
    monkeypatch.delenv("NETRA_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("NETRA_TRUSTED_PROXY_HOPS", raising=False)


@pytest.fixture
def client():
    # No `with` block, so the lifespan (model loading) does not run.
    return TestClient(main.app)


def loaded_with_score(monkeypatch, score: float, confidence: float = 0.9):
    """Make the API think DistilBERT is loaded and always return `score`."""
    monkeypatch.setattr(main.state, "db_ready", True)
    monkeypatch.setattr(main, "_score_distilbert", lambda req, urls: (score, confidence))


BENIGN = {"body_text": "Hi team, the weekly report is attached. Thanks."}


# ---------------------------------------------------------------------------
# Health (A8 / L5)
# ---------------------------------------------------------------------------
def test_health_503_when_no_model(client):
    r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["status"] == "not_ready"
    assert r.json()["model_loaded"] is False


def test_health_reports_real_model_type(client, monkeypatch):
    monkeypatch.setattr(main.state, "db_ready", True)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_type"] == "distilbert"
    assert body["distilbert_ready"] is True


def test_health_reports_rf_fallback_as_loaded(client, monkeypatch):
    monkeypatch.setattr(main.state, "rf_ready", True)
    monkeypatch.setattr(main.state, "rf_model_type", "calibrated_rf")
    body = client.get("/health").json()
    assert body["model_loaded"] is True
    assert body["model_type"] == "calibrated_rf"
    assert body["distilbert_ready"] is False


# ---------------------------------------------------------------------------
# Predict input validation (A11)
# ---------------------------------------------------------------------------
def test_predict_503_without_model(client):
    assert client.post("/predict", json=BENIGN).status_code == 503


def test_predict_422_on_empty_body(client, monkeypatch):
    loaded_with_score(monkeypatch, 0.0)
    assert client.post("/predict", json={"body_text": "   "}).status_code == 422


def test_predict_422_on_oversized_body(client, monkeypatch):
    loaded_with_score(monkeypatch, 0.0)
    too_big = {"body_text": "a" * (main.MAX_BODY_CHARS + 1)}
    assert client.post("/predict", json=too_big).status_code == 422


def test_predict_422_on_too_many_urls(client, monkeypatch):
    loaded_with_score(monkeypatch, 0.0)
    payload = {"body_text": "hello", "urls": [f"https://x{i}.example.com" for i in range(main.MAX_URLS + 1)]}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_422_on_too_many_header_keys(client, monkeypatch):
    loaded_with_score(monkeypatch, 0.0)
    payload = {"body_text": "hello", "headers": {f"k{i}": "x" for i in range(main.MAX_HEADER_KEYS + 1)}}
    assert client.post("/predict", json=payload).status_code == 422


# ---------------------------------------------------------------------------
# Decision boundaries (A1 / A7 / A21)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("score, expected", [
    (0.079, "LEGITIMATE"),
    (0.08, "SUSPICIOUS"),
    (0.249, "SUSPICIOUS"),
    (0.25, "PHISHING"),
    (0.60, "PHISHING"),
])
def test_classification_follows_score_thresholds(client, monkeypatch, score, expected):
    loaded_with_score(monkeypatch, score)
    r = client.post("/predict", json=BENIGN)
    assert r.status_code == 200
    assert r.json()["classification"] == expected


def test_risk_level_matches_verdict(client, monkeypatch):
    loaded_with_score(monkeypatch, 0.6)
    assert client.post("/predict", json=BENIGN).json()["risk_level"] == "CRITICAL"
    loaded_with_score(monkeypatch, 0.3)
    assert client.post("/predict", json=BENIGN).json()["risk_level"] == "HIGH"
    loaded_with_score(monkeypatch, 0.1)
    assert client.post("/predict", json=BENIGN).json()["risk_level"] == "MEDIUM"
    loaded_with_score(monkeypatch, 0.01)
    assert client.post("/predict", json=BENIGN).json()["risk_level"] == "LOW"


def test_signal_override_reaches_phishing_with_three_signals(client, monkeypatch):
    """Score is low, but 3 threat signals fire, so the verdict must be PHISHING (A1 / L4)."""
    loaded_with_score(monkeypatch, 0.01)
    payload = {
        "body_text": "Please review.",
        "sender": "it@paypa1-secure.com",                  # typosquatted sender domain
        "urls": ["http://example.com/login"],              # http -> suspicious URL
        "headers": {"spf": "fail", "dkim": "none", "dmarc": "none"},
    }
    body = client.post("/predict", json=payload).json()
    assert body["signals"]["typosquatting_detected"] is True
    assert body["signals"]["header_auth_failed"] is True
    assert body["classification"] == "PHISHING"
    assert body["risk_score"] >= 0.25


def test_signal_override_two_signals_is_suspicious(client, monkeypatch):
    loaded_with_score(monkeypatch, 0.01)
    payload = {
        "body_text": "Please review.",
        "sender": "it@paypa1-secure.com",
        "headers": {"spf": "fail"},
    }
    body = client.post("/predict", json=payload).json()
    assert body["classification"] == "SUSPICIOUS"
    assert body["risk_score"] >= 0.08


def test_model_text_matches_training_format(monkeypatch):
    """Serving must send the checkpoint the text format it was trained on (ML2 / H2)."""
    captured = {}

    class FakeTok:
        def __call__(self, text, **kw):
            captured["text"] = text
            import torch
            return {"input_ids": torch.zeros((1, 4), dtype=torch.long),
                    "attention_mask": torch.ones((1, 4), dtype=torch.long)}

    class FakeModel:
        def __call__(self, ids, mask, hdr):
            import torch
            captured["hdr"] = hdr.tolist()
            return torch.tensor([[0.0, 0.0]])

    monkeypatch.setattr(main.state, "db_tokenizer", FakeTok())
    monkeypatch.setattr(main.state, "db_model", FakeModel())
    monkeypatch.setattr(main.state, "db_config", {"max_length": 8, "text_format": "v3",
                                                  "header_features_trained": True})
    req = main.PredictRequest(subject="Hi", body_text="Body here",
                              urls=["https://a.example.com"], headers={"spf": "pass"})
    main._score_distilbert(req, main.extract_urls(req.subject, req.body_text, req.urls))
    assert captured["text"] == "Subject: Hi\nBody: Body here\nURL: https://a.example.com"
    assert "Phishing URL" not in captured["text"]
    assert captured["hdr"][0][0] == 1.0     # spf_pass is set: a real header vector, not zeros


def test_legacy_checkpoint_gets_body_only_and_zero_headers(monkeypatch):
    captured = {}

    class FakeTok:
        def __call__(self, text, **kw):
            captured["text"] = text
            import torch
            return {"input_ids": torch.zeros((1, 4), dtype=torch.long),
                    "attention_mask": torch.ones((1, 4), dtype=torch.long)}

    class FakeModel:
        def __call__(self, ids, mask, hdr):
            import torch
            captured["hdr"] = hdr.tolist()
            return torch.tensor([[0.0, 0.0]])

    monkeypatch.setattr(main.state, "db_tokenizer", FakeTok())
    monkeypatch.setattr(main.state, "db_model", FakeModel())
    monkeypatch.setattr(main.state, "db_config", {"max_length": 8})
    req = main.PredictRequest(subject="Hi", body_text="Body here", headers={"spf": "pass"})
    main._score_distilbert(req, [])
    assert captured["text"] == "Body here"
    assert captured["hdr"] == [[0.0] * 10]


# ---------------------------------------------------------------------------
# Rate limiting (A3 / A4)
# ---------------------------------------------------------------------------
def test_rate_limit_429_after_20(client, monkeypatch):
    loaded_with_score(monkeypatch, 0.0)
    for _ in range(20):
        assert client.post("/predict", json=BENIGN).status_code == 200
    r = client.post("/predict", json=BENIGN)
    assert r.status_code == 429
    assert "Retry-After" in r.headers


def test_spoofed_xff_does_not_bypass_limit_without_proxy(client, monkeypatch):
    """With NETRA_TRUSTED_PROXY_HOPS unset, a client cannot change its key with XFF (A3)."""
    loaded_with_score(monkeypatch, 0.0)
    codes = []
    for i in range(21):
        codes.append(client.post("/predict", json=BENIGN,
                                 headers={"X-Forwarded-For": f"10.0.0.{i}"}).status_code)
    assert codes[-1] == 429


def test_xff_honoured_behind_trusted_proxy(client, monkeypatch):
    """With one trusted proxy, the right-most XFF entry is the real client."""
    monkeypatch.setenv("NETRA_TRUSTED_PROXY_HOPS", "1")
    loaded_with_score(monkeypatch, 0.0)
    for _ in range(20):
        client.post("/predict", json=BENIGN, headers={"X-Forwarded-For": "1.1.1.1, 203.0.113.9"})
    r = client.post("/predict", json=BENIGN, headers={"X-Forwarded-For": "9.9.9.9, 203.0.113.9"})
    assert r.status_code == 429          # same real client (right-most entry), different spoofed left entry


def test_client_ip_rules():
    from starlette.requests import Request as StarletteRequest

    def make(headers):
        scope = {"type": "http", "method": "POST", "path": "/predict", "headers": [
            (k.lower().encode(), v.encode()) for k, v in headers.items()],
            "client": ("192.0.2.5", 1234)}
        return StarletteRequest(scope)

    assert main.client_ip(make({"X-Forwarded-For": "1.2.3.4"}), 0) == "192.0.2.5"
    assert main.client_ip(make({"X-Forwarded-For": "1.2.3.4, 5.6.7.8"}), 1) == "5.6.7.8"
    assert main.client_ip(make({}), 1) == "192.0.2.5"


# ---------------------------------------------------------------------------
# Admin auth and recalibration (A2 / A17)
# ---------------------------------------------------------------------------
def test_admin_disabled_without_token(client):
    r = client.post("/admin/recalibrate", json={"suspicious_lower": 0.1, "phishing_threshold": 0.3})
    assert r.status_code == 404


def test_admin_rejects_missing_or_wrong_token(client, monkeypatch):
    monkeypatch.setenv("NETRA_ADMIN_TOKEN", "s3cret")
    body = {"suspicious_lower": 0.1, "phishing_threshold": 0.3}
    assert client.post("/admin/recalibrate", json=body).status_code == 401
    assert client.post("/admin/recalibrate", json=body,
                       headers={"Authorization": "Bearer nope"}).status_code == 401


def test_admin_accepts_valid_token_and_persists(client, monkeypatch, tmp_path):
    monkeypatch.setenv("NETRA_ADMIN_TOKEN", "s3cret")
    r = client.post("/admin/recalibrate", json={"suspicious_lower": 0.1, "phishing_threshold": 0.3},
                    headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 200
    assert main.state.phishing_threshold == 0.3
    saved = main.THRESHOLD_CONFIG_PATH.read_text()
    assert '"phishing_threshold": 0.3' in saved


def test_admin_rejects_inverted_corridor(client, monkeypatch):
    monkeypatch.setenv("NETRA_ADMIN_TOKEN", "s3cret")
    r = client.post("/admin/recalibrate", json={"suspicious_lower": 0.5, "phishing_threshold": 0.3},
                    headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 422
    assert main.state.phishing_threshold == 0.25


def test_failed_persist_leaves_state_unchanged(client, monkeypatch):
    monkeypatch.setenv("NETRA_ADMIN_TOKEN", "s3cret")

    def boom(_updates):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(main, "_save_thresholds", boom)
    r = client.post("/admin/recalibrate", json={"suspicious_lower": 0.1, "phishing_threshold": 0.3},
                    headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 500
    assert main.state.phishing_threshold == 0.25      # unchanged (A17)


# ---------------------------------------------------------------------------
# Error hygiene (A22)
# ---------------------------------------------------------------------------
def test_internal_error_message_is_generic(client, monkeypatch):
    monkeypatch.setattr(main.state, "db_ready", True)

    def broken(req, urls):
        raise RuntimeError("secret internal detail /path/to/file")

    monkeypatch.setattr(main, "_score_distilbert", broken)
    r = client.post("/predict", json=BENIGN)
    assert r.status_code == 500
    assert r.json()["detail"] == "Prediction failed"
    assert "secret" not in r.text


# ---------------------------------------------------------------------------
# /explain
# ---------------------------------------------------------------------------
PREDICT_KEYS = {"classification", "risk_score", "confidence", "risk_level", "model_type",
                "threshold_used", "signals", "processing_time_ms"}

PHISH = {
    "subject": "Account suspended",
    "body_text": "URGENT: Your account is suspended. Verify immediately.",
    "urls": ["http://paypa1-secure.example.tk/login"],
    "sender": "security@paypa1-alert.com",
    "headers": {"spf": "fail", "dkim": "fail", "dmarc": "fail"},
}


def explained_with_margin(monkeypatch, f_input: float, f_baseline: float = 0.4):
    """DistilBERT 'loaded'; IG returns a fixed result whose margin is f_input."""
    from ml.explain import IGResult, WordAttribution

    def fake(req, urls):
        body = req.body_text
        words = [WordAttribution("Verify", "body", None, body.find("Verify"), body.find("Verify") + 6,
                                 (f_input - f_baseline) * 0.7),
                 WordAttribution("account", "body", None, body.find("account"), body.find("account") + 7,
                                 (f_input - f_baseline) * 0.3)]
        ig = IGResult(words=words, header=[0.0] * 10, f_input=f_input, f_baseline=f_baseline, delta=0.0,
                      n_tokens=12, n_steps=32, truncated=False)
        return ig, [0.0] * 10, False

    monkeypatch.setattr(main.state, "db_ready", True)
    monkeypatch.setattr(main, "_explain_distilbert", fake)


def test_predict_response_keys_unchanged(client, monkeypatch):
    loaded_with_score(monkeypatch, 0.6)
    assert set(client.post("/predict", json=BENIGN).json()) == PREDICT_KEYS


def test_explain_503_without_model(client):
    assert client.post("/explain", json=BENIGN).status_code == 503


def test_explain_matches_decision_and_adds_explanation(client, monkeypatch):
    explained_with_margin(monkeypatch, 3.0)
    r = client.post("/explain", json=PHISH)
    assert r.status_code == 200
    body = r.json()
    assert PREDICT_KEYS <= set(body)
    p = 1 / (1 + 2.718281828459045 ** -3.0)
    assert body["classification"] == "PHISHING"
    assert body["risk_score"] == pytest.approx(p, abs=1e-4)
    x = body["explanation"]
    assert x["method"] == "integrated_gradients"
    assert x["top_phishing_tokens"][0]["text"] == "Verify"
    tot = x["baseline_score"] + sum(f["contribution"] for f in x["threat_factors"])
    assert tot == pytest.approx(body["risk_score"], abs=1e-3)
    assert x["summary"].startswith("PHISHING")
    assert x["threat_factors"][0]["label"].lower() in x["summary"]


def test_explain_override_is_attributed_to_signals(client, monkeypatch):
    explained_with_margin(monkeypatch, -5.0)      # model alone: LEGITIMATE
    r = client.post("/explain", json=PHISH)
    body = r.json()
    assert body["classification"] == "PHISHING"   # >= 3 signals
    x = body["explanation"]
    assert x["override"] == "phishing_floor"
    rule = [c for f in x["threat_factors"] for c in f["components"] if c["source"] == "rule"]
    assert len(rule) == 4 and len({c["contribution"] for c in rule}) == 1


def test_explain_rf_fallback_is_signals_only(client, monkeypatch):
    monkeypatch.setattr(main.state, "rf_ready", True)
    monkeypatch.setattr(main, "_score_rf", lambda req, urls: (0.01, 0.99))
    r = client.post("/explain", json=PHISH)
    x = r.json()["explanation"]
    assert x["method"] == "signals_only"
    assert x["top_phishing_tokens"] == [] and x["header_attributions"] == []


def test_explain_has_its_own_rate_limit(client, monkeypatch):
    explained_with_margin(monkeypatch, 3.0)
    codes = [client.post("/explain", json=PHISH).status_code for _ in range(6)]
    assert codes == [200] * 5 + [429]
    loaded_with_score(monkeypatch, 0.6)
    assert client.post("/predict", json=BENIGN).status_code == 200   # /predict not starved


def test_explain_spoofed_xff_does_not_bypass_limit(client, monkeypatch):
    explained_with_margin(monkeypatch, 3.0)
    for i in range(5):
        client.post("/explain", json=PHISH, headers={"X-Forwarded-For": f"10.0.0.{i}"})
    r = client.post("/explain", json=PHISH, headers={"X-Forwarded-For": "10.0.0.99"})
    assert r.status_code == 429


def test_explain_top_k_validated(client, monkeypatch):
    explained_with_margin(monkeypatch, 3.0)
    assert client.post("/explain", json=dict(PHISH, top_k=0)).status_code == 422
    assert client.post("/explain", json=dict(PHISH, top_k=26)).status_code == 422


def test_explain_internal_error_is_generic(client, monkeypatch):
    monkeypatch.setattr(main.state, "db_ready", True)

    def broken(req, urls):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(main, "_explain_distilbert", broken)
    r = client.post("/explain", json=PHISH)
    assert r.status_code == 500 and "secret" not in r.text


def test_prepare_inputs_header_baseline_is_no_headers(monkeypatch):
    monkeypatch.setattr(main.state, "db_config", {"text_format": "v3", "header_features_trained": True})
    req = main.PredictRequest(**PHISH)
    text, spans, hv, hb = main._prepare_distilbert_inputs(req, main.extract_urls(req.subject, req.body_text, req.urls))
    assert text.startswith("Subject: Account suspended\nBody: ")
    assert [s.field for s in spans] == ["subject", "body", "url"]
    assert hb == [0, 0, 1, 0, 0, 1, 0, 0, 1, 1]
    assert hv[1] == 1 and hv[4] == 1 and hv[7] == 1     # spf/dkim/dmarc fail

    monkeypatch.setattr(main.state, "db_config", {})
    text, spans, hv, hb = main._prepare_distilbert_inputs(req, [])
    assert text == req.body_text and hv == hb == [0.0] * 10
