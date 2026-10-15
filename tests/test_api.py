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
    monkeypatch.setattr(main, "_limiter", SlidingWindowRateLimiter(20, 60))
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
