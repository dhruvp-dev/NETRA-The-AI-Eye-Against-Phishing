# -*- coding: utf-8 -*-
"""
NETRA - FastAPI Inference Server (Tier-1)
=========================================
Serves the Tier-1 phishing detector: DistilBERT (primary) with a calibrated
Random Forest fallback.

Start server:
    uvicorn api.main:app --host 127.0.0.1 --port 8000

Endpoints:
    GET  /health                 -> server + model status + active thresholds (503 if no model)
    POST /predict                -> classification + risk_score + signals (rate limited)
    POST /explain                -> /predict fields + token / header / signal attribution (on demand,
                                    seconds on CPU; separate rate limit and inference slot)
    POST /admin/recalibrate      -> live threshold update (needs NETRA_ADMIN_TOKEN)

Environment:
    NETRA_ADMIN_TOKEN            Bearer token for /admin/recalibrate. Unset = endpoint disabled (404).
    NETRA_TRUSTED_PROXY_HOPS     Reverse proxies in front of the API (Cloud Run: 1). Default 0 = ignore
                                 X-Forwarded-For entirely.
    NETRA_MAX_CONCURRENCY        Max concurrent model inferences. Default 2.
    NETRA_XAI_STEPS              Integrated Gradients steps for /explain. Default 32, clamped to 8..64.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import sys
import time
import urllib.request
from contextlib import asynccontextmanager
from email.utils import parseaddr
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Dict, List, Literal, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator
from typing import Annotated

from api.rate_limiter import SlidingWindowRateLimiter
from api.trust import TrustAssessment, assess, is_shortener, url_host
from api.xai import Explanation, fuse, override_units, sigmoid
from ml.features.header_features import HEADER_FEATURE_NAMES, extract as header_extract, normalize_auth
from ml.features.text_builder import Span, build_model_text_with_spans, extract_urls
from ml.features.text_features import URGENCY_KEYWORDS
from ml.features.url_features import check_typosquatting, extract as url_extract, URL_FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODELS_DIR = ROOT / "ml" / "models"
MODEL_CALIBRATED_PATH = MODELS_DIR / "tier1_model_calibrated.pkl"
MODEL_PATH = MODELS_DIR / "tier1_model.pkl"
TFIDF_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
THRESHOLD_CONFIG_PATH = MODELS_DIR / "threshold_config.json"           # live, gitignored
THRESHOLD_DEFAULT_PATH = MODELS_DIR / "threshold_config.default.json"  # committed defaults
DISTILBERT_CONFIG_PATH = MODELS_DIR / "distilbert_config.json"
DISTILBERT_MODEL_PATH = MODELS_DIR / "distilbert_tier1.pt"
DISTILBERT_TOKENIZER_PATH = MODELS_DIR / "distilbert_tokenizer"
DISTILBERT_RELEASE_URL = (
    "https://github.com/ramanan-2735/NETRA-The-AI-Eye-Against-Phishing/"
    "releases/download/v2.0.0/distilbert_tier1.pt"
)

# ---------------------------------------------------------------------------
# Decision rules (see CLAUDE.md "Threshold Consistency")
# ---------------------------------------------------------------------------
DEFAULT_PHISHING_THRESHOLD = 0.2500
DEFAULT_SUSPICIOUS_LOWER = 0.0800
CRITICAL_RISK_SCORE = 0.50          # PHISHING at or above this is reported as CRITICAL
SIGNAL_OVERRIDE_SUSPICIOUS = 2      # LEGITIMATE + this many threat signals -> SUSPICIOUS
SIGNAL_OVERRIDE_PHISHING = 3        # LEGITIMATE + this many threat signals -> PHISHING
MIN_SUSPICIOUS_BAND = 0.03          # SUSPICIOUS must be a real band (audit B2: a 0.0002 band was unreachable)

# Long emails are scored in windows (audit B3): the head plus up to MAX_EXTRA_WINDOWS body chunks,
# each with the subject and URL lines, and the highest score wins.
MAX_EXTRA_WINDOWS = 4
WINDOW_CHARS = 400
WINDOW_OVERLAP_CHARS = 80

# ---------------------------------------------------------------------------
# Request limits
# ---------------------------------------------------------------------------
MAX_BODY_CHARS = 100_000
MAX_SUBJECT_CHARS = 1_000
MAX_ADDRESS_CHARS = 512
MAX_URL_CHARS = 2_048
MAX_URLS = 50
MAX_HEADER_KEYS = 50

# ---------------------------------------------------------------------------
# Rate limiting (20 requests per minute per client)
# ---------------------------------------------------------------------------
RATE_LIMIT_PER_MINUTE = 20
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMITED_PATHS = {"/predict", "/explain", "/admin/recalibrate"}
_limiter = SlidingWindowRateLimiter(RATE_LIMIT_PER_MINUTE, RATE_LIMIT_WINDOW_SECONDS)

# /explain runs ~50 forward/backward passes, so it has its own, tighter limit on top of the shared one.
EXPLAIN_RATE_LIMIT_PER_MINUTE = 5
EXPLAIN_PATH = "/explain"
_explain_limiter = SlidingWindowRateLimiter(EXPLAIN_RATE_LIMIT_PER_MINUTE, RATE_LIMIT_WINDOW_SECONDS)


def _trusted_proxy_hops() -> int:
    try:
        return max(0, int(os.getenv("NETRA_TRUSTED_PROXY_HOPS", "0")))
    except ValueError:
        return 0


def _max_concurrency() -> int:
    try:
        return max(1, int(os.getenv("NETRA_MAX_CONCURRENCY", "2")))
    except ValueError:
        return 2


def _xai_steps() -> int:
    try:
        return min(64, max(8, int(os.getenv("NETRA_XAI_STEPS", "32"))))
    except ValueError:
        return 32


def client_ip(request: Request, trusted_hops: int) -> str:
    """
    Resolve the client IP. X-Forwarded-For is client-controlled, so it is only read
    when a known number of trusted proxies sits in front of the API. The proxy
    appends the real client on the right, so the entry `trusted_hops` from the
    right is used (Cloud Run: 1).
    """
    socket_ip = request.client.host if request.client else "unknown"
    if trusted_hops <= 0:
        return socket_ip
    entries = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
    if len(entries) >= trusted_hops:
        return entries[-trusted_hops]
    return socket_ip


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_models()
    yield


app = FastAPI(
    title="NETRA - AI-Powered Phishing Detection Engine",
    description="""
### Tier-1 Phishing Detection

- **Model**: fine-tuned DistilBERT with 10 RFC header features (SPF / DKIM / DMARC, sender/reply-to match)
- **Fallback**: calibrated Random Forest when DistilBERT is unavailable
- **Signals**: URL heuristics and Levenshtein typosquatting against top brand domains
    """,
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(chrome-extension://[a-p]{32}|http://(localhost|127\.0\.0\.1)(:\d+)?)$",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path in RATE_LIMITED_PATHS:
        ip = client_ip(request, _trusted_proxy_hops())
        checks = [(_limiter, RATE_LIMIT_PER_MINUTE)]
        if request.url.path == EXPLAIN_PATH:
            checks.append((_explain_limiter, EXPLAIN_RATE_LIMIT_PER_MINUTE))
        for limiter, limit in checks:
            allowed, retry_after = limiter.check(ip)
            if not allowed:
                log.warning(f"Rate limit exceeded for {ip} on {request.url.path}")
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "rate_limit_exceeded",
                        "message": f"Rate limit exceeded. Maximum {limit} requests per minute allowed.",
                        "retry_after_seconds": retry_after,
                    },
                    headers={"Retry-After": str(retry_after)},
                )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
class ModelState:
    # RF fallback
    rf_model = None
    text_extractor = None
    rf_ready: bool = False
    rf_model_type: str = "uncalibrated_rf"

    # DistilBERT (primary)
    db_model = None
    db_tokenizer = None
    db_config: dict = {}
    db_ready: bool = False

    # Active thresholds
    phishing_threshold: float = DEFAULT_PHISHING_THRESHOLD
    suspicious_lower: float = DEFAULT_SUSPICIOUS_LOWER
    suspicious_upper: float = DEFAULT_PHISHING_THRESHOLD
    threshold_source: str = "default_fallback"

    # Concurrency gates, created on first use inside the running event loop. /explain has its own
    # single slot so a slow explanation never holds up /predict.
    inference_slots: Optional[asyncio.Semaphore] = None
    explain_slots: Optional[asyncio.Semaphore] = None

    @property
    def model_type(self) -> str:
        if self.db_ready:
            return "distilbert"
        return self.rf_model_type

    @property
    def model_loaded(self) -> bool:
        return self.db_ready or self.rf_ready


state = ModelState()


# ---------------------------------------------------------------------------
# Threshold config
# ---------------------------------------------------------------------------
def _valid_thresholds(lower: float, phishing: float) -> bool:
    return 0.0 < lower and phishing <= 1.0 and phishing - lower >= MIN_SUSPICIOUS_BAND - 1e-9


def _load_thresholds() -> None:
    """Load thresholds from the live file, or the committed defaults if none exists."""
    path = THRESHOLD_CONFIG_PATH if THRESHOLD_CONFIG_PATH.exists() else THRESHOLD_DEFAULT_PATH
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        phishing = float(cfg.get("phishing_threshold", DEFAULT_PHISHING_THRESHOLD))
        lower = float(cfg.get("suspicious_lower", DEFAULT_SUSPICIOUS_LOWER))
        if not _valid_thresholds(lower, phishing):
            raise ValueError(f"invalid corridor lower={lower} phishing={phishing}")
        state.phishing_threshold = phishing
        state.suspicious_lower = lower
        state.suspicious_upper = phishing
        state.threshold_source = cfg.get("method", path.name)
        log.info(f"Thresholds loaded from {path.name}: phishing>={phishing:.4f} "
                 f"suspicious>={lower:.4f}")
    except Exception as e:
        log.warning(f"Could not use {path.name} ({e}); using built-in defaults")


def _save_thresholds(updates: dict) -> None:
    """
    Persist thresholds. Existing keys are preserved (read-merge-write) and the
    file is replaced atomically, so a crash cannot leave half-written JSON.
    Raises OSError when the write fails; the caller must not change in-memory state then.
    """
    existing = {}
    if THRESHOLD_CONFIG_PATH.exists():
        existing = json.loads(THRESHOLD_CONFIG_PATH.read_text(encoding="utf-8"))
    existing.update(updates)
    tmp = THRESHOLD_CONFIG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    os.replace(tmp, THRESHOLD_CONFIG_PATH)


# ---------------------------------------------------------------------------
# Weights download (verified, atomic)
# ---------------------------------------------------------------------------
def _download_weights(url: str, dest: Path, expected_sha256: Optional[str], timeout: int = 60) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(part, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                digest.update(chunk)
        got = digest.hexdigest()
        if expected_sha256 and got != expected_sha256:
            raise RuntimeError(f"SHA-256 mismatch: expected {expected_sha256}, got {got}")
        if not expected_sha256:
            log.warning("No weights_sha256 in distilbert_config.json; download is not verified.")
        os.replace(part, dest)
    finally:
        if part.exists():
            part.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
def load_models() -> None:
    log.info("=" * 60)
    log.info("NETRA API starting up...")
    log.info("=" * 60)

    _load_thresholds()

    db_cfg: dict = {}
    if DISTILBERT_CONFIG_PATH.exists():
        try:
            db_cfg = json.loads(DISTILBERT_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning(f"Could not read distilbert_config.json: {e}")

    if not DISTILBERT_MODEL_PATH.exists():
        log.info(f"{DISTILBERT_MODEL_PATH.name} not found. Downloading from GitHub release...")
        try:
            _download_weights(DISTILBERT_RELEASE_URL, DISTILBERT_MODEL_PATH, db_cfg.get("weights_sha256"))
            log.info(f"{DISTILBERT_MODEL_PATH.name} downloaded and verified.")
        except Exception as e:
            log.warning(f"Failed to download model from GitHub: {e}")

    # --- DistilBERT (primary) ---
    if DISTILBERT_CONFIG_PATH.exists() and DISTILBERT_MODEL_PATH.exists() and DISTILBERT_TOKENIZER_PATH.exists():
        try:
            import torch
            from transformers import AutoTokenizer
            from ml.distilbert_model import PhishingClassifier

            db_model = PhishingClassifier()
            db_model.load_state_dict(
                torch.load(DISTILBERT_MODEL_PATH, map_location="cpu", weights_only=True)
            )
            db_model.eval()
            # Frozen: /explain needs gradients w.r.t. inputs only, never w.r.t. weights.
            for p in db_model.parameters():
                p.requires_grad_(False)

            state.db_config = db_cfg
            state.db_tokenizer = AutoTokenizer.from_pretrained(str(DISTILBERT_TOKENIZER_PATH))
            state.db_model = db_model
            state.db_ready = True
            log.info("DistilBERT model loaded.")
        except Exception as e:
            log.warning(f"DistilBERT load failed: {e}; falling back to RF")

    # --- Random Forest (fallback) ---
    rf_path = MODEL_CALIBRATED_PATH if MODEL_CALIBRATED_PATH.exists() else MODEL_PATH
    if rf_path.exists() and TFIDF_PATH.exists():
        try:
            import joblib
            state.rf_model = joblib.load(rf_path)
            state.text_extractor = joblib.load(TFIDF_PATH)
            state.rf_ready = True
            state.rf_model_type = "calibrated_rf" if rf_path == MODEL_CALIBRATED_PATH else "uncalibrated_rf"
            log.info(f"RF model loaded [{state.rf_model_type}]: {rf_path.name}")
        except Exception as e:
            log.error(f"RF model load failed: {e}")

    log.info(f"Active model: {state.model_type} | loaded: {state.model_loaded}")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
Short = Annotated[str, StringConstraints(max_length=MAX_SUBJECT_CHARS)]
Address = Annotated[str, StringConstraints(max_length=MAX_ADDRESS_CHARS)]
UrlStr = Annotated[str, StringConstraints(max_length=MAX_URL_CHARS)]


class PredictRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "subject": "Account suspended",
        "body_text": "URGENT: Your account is suspended. Verify immediately.",
        "urls": ["http://paypa1-secure.example.tk/login"],
        "sender": "security@paypa1-alert.com",
        "reply_to": "verify@account-recovery.example",
        "headers": {"spf": "fail", "dkim": "fail", "dmarc": "fail"},
    }})

    subject: Optional[Short] = None
    body_text: Annotated[str, StringConstraints(max_length=MAX_BODY_CHARS)] = Field(
        "", description="Full email body text (may be empty when subject or URLs are given)")
    urls: Optional[List[UrlStr]] = Field(default=None, max_length=MAX_URLS,
                                         description="URLs extracted from the email (first 50 kept)")
    sender: Optional[Address] = None
    reply_to: Optional[Address] = None
    headers: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Parsed auth results. Keys: 'spf', 'dkim', 'dmarc' -> 'pass'|'fail'|'none'",
    )

    # Real emails exceed these limits (long tracking URLs, many links, long display names).
    # Clip them instead of rejecting the email with 422 (audit B11).
    @field_validator("urls", mode="before")
    @classmethod
    def _clip_urls(cls, v):
        if isinstance(v, list):
            v = [u for u in v if not isinstance(u, str) or len(u) <= MAX_URL_CHARS][:MAX_URLS]
        return v

    @field_validator("sender", "reply_to", mode="before")
    @classmethod
    def _clip_address(cls, v):
        return v[:MAX_ADDRESS_CHARS] if isinstance(v, str) else v

    @field_validator("subject", mode="before")
    @classmethod
    def _clip_subject(cls, v):
        return v[:MAX_SUBJECT_CHARS] if isinstance(v, str) else v

    @field_validator("body_text", mode="before")
    @classmethod
    def _none_body_is_empty(cls, v):
        return "" if v is None else v

    @field_validator("headers")
    @classmethod
    def _limit_header_keys(cls, v):
        if v is not None and len(v) > MAX_HEADER_KEYS:
            raise ValueError(f"headers may contain at most {MAX_HEADER_KEYS} keys")
        return v


class SignalsDict(BaseModel):
    header_auth_failed: bool
    urgency_detected: bool
    suspicious_urls: int
    typosquatting_detected: bool
    # Added after the Tier-1 audit (additive; older clients ignore them).
    ip_url_detected: bool = False
    shortened_url_detected: bool = False
    risky_file_link: bool = False
    executable_link: bool = False
    bec_detected: bool = False


class PredictResponse(BaseModel):
    classification: str
    risk_score: float
    confidence: float
    risk_level: str
    model_type: str
    threshold_used: float
    signals: SignalsDict
    processing_time_ms: float
    model_score: Optional[float] = Field(None, description="Model probability before the rule layer")
    adjustment: Optional[Literal["trusted_sender", "signal_override"]] = Field(
        None, description="Rule that changed the model verdict, if any")
    adjustment_reason: Optional[str] = None


class ExplainRequest(PredictRequest):
    top_k: int = Field(8, ge=1, le=25, description="Tokens per list (phishing / legitimate)")


class ExplainResponse(PredictResponse):
    explanation: Explanation


class HealthResponse(BaseModel):
    status: str
    model_type: str
    model_loaded: bool
    distilbert_ready: bool
    thresholds: Dict[str, float]
    threshold_source: str


class RecalibrateRequest(BaseModel):
    suspicious_lower: float = Field(..., gt=0.0, lt=1.0)
    phishing_threshold: float = Field(..., gt=0.0, le=1.0)


class RecalibrateResponse(BaseModel):
    status: str
    new_thresholds: Dict[str, float]


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------
def _sender_domain(sender: Optional[str]) -> str:
    _, addr = parseaddr(sender or "")
    return addr.rsplit("@", 1)[1].lower() if "@" in addr else ""


# Business-email-compromise lure: a request for money or credentials ...
_BEC_REQUEST = re.compile(
    r"\b(wire (the |a )?(transfer|payment|funds)|wire transfer|bank transfer|gift ?cards?|itunes cards?|"
    r"google play cards?|steam cards?|send (me )?the (codes?|card numbers?)|scratch (off )?the|"
    r"(change|update)d? (of |to )?(our |my |the )?(bank |banking )?(bank|banking|account|payment|direct deposit|payroll) "
    r"(details|information|info)|new bank account|update (my |the )?payroll|"
    r"process (a|the) payment|make (a|the) payment (today|now|asap)|outstanding invoice|overdue invoice|"
    r"invoice (is )?(overdue|past due)|pay(ment)? (is )?overdue|(send|pay|transfer)\b.{0,30}\b(bitcoin|btc|crypto)|"
    r"(send|share) (me )?(your )?(password|login|credentials|otp|verification code))\b",
    re.IGNORECASE)
# ... combined with pressure or secrecy.
_BEC_PRESSURE = re.compile(
    r"\b(urgent(ly)?|asap|immediately|right away|right now|within (the )?(hour|\d+ (hours?|minutes?))|"
    r"before (the )?end of (the )?day|by eod|today|confidential|discreet|between us|keep (this|it) (quiet|private)|"
    r"(don'?t|do not) (call|tell|discuss)|in a meeting|are you (available|at your desk|around)|quick favou?r|"
    r"can'?t talk|final notice|last (notice|reminder)|legal action|penalt(y|ies)|suspend(ed)?)\b",
    re.IGNORECASE)
_EXECUTABLE_FILE = re.compile(r"\.(exe|scr|bat|cmd|pif|js|jse|vbs|vbe|wsf|hta|msi|lnk|ps1)$", re.IGNORECASE)
_RISKY_FILE = re.compile(r"\.(iso|img|zip|rar|7z|html?)$", re.IGNORECASE)


def _link_path(url: str) -> str:
    try:
        return (urlparse(url if "://" in url else f"http://{url}").path or "").rstrip("/")
    except ValueError:
        return ""


def _executable_link(url: str) -> bool:
    return bool(_EXECUTABLE_FILE.search(_link_path(url)))


def _risky_file_link(url: str) -> bool:
    return bool(_RISKY_FILE.search(_link_path(url)))


def _extract_signals(req: PredictRequest, urls: List[str]) -> dict:
    """Interpretable signals for the response and for the multi-signal override."""
    hdrs = req.headers or {}
    auth_values = [normalize_auth(hdrs.get(k)) for k in ("spf", "dkim", "dmarc")]
    header_auth_failed = "fail" in auth_values

    body_lower = (req.body_text or "").lower()
    urgency_detected = sum(1 for kw in URGENCY_KEYWORDS if kw in body_lower) >= 2

    url_feats = url_extract(urls)
    suspicious_urls = int(
        url_feats["any_http_url"] + url_feats["any_phishing_tld"] + url_feats["any_typosquatting"]
    )
    typosquatting = bool(url_feats.get("any_typosquatting", 0))
    if not typosquatting:
        domain = _sender_domain(req.sender)
        if domain:
            typosquatting = bool(check_typosquatting(domain)[0])

    text = f"{req.subject or ''}\n{req.body_text or ''}"
    return {
        "header_auth_failed": header_auth_failed,
        "urgency_detected": urgency_detected,
        "suspicious_urls": suspicious_urls,
        "typosquatting_detected": typosquatting,
        "ip_url_detected": bool(url_feats["any_ip_url"]),
        "shortened_url_detected": any(is_shortener(url_host(u)) for u in urls),
        "risky_file_link": any(_risky_file_link(u) for u in urls),
        "executable_link": any(_executable_link(u) for u in urls),
        "bec_detected": bool(_BEC_REQUEST.search(text) and _BEC_PRESSURE.search(text)),
    }


def _signal_count(signals: dict, url_feats: Optional[dict] = None) -> int:
    """Number of threat units (api/xai.py: override_units is the single list of them)."""
    return len(override_units(signals, url_feats or {}))


# ---------------------------------------------------------------------------
# Decision logic (one rule set for every model)
# ---------------------------------------------------------------------------
def _classify(risk_score: float) -> str:
    """The only place the score-to-verdict rule lives."""
    if risk_score >= state.phishing_threshold:
        return "PHISHING"
    if risk_score >= state.suspicious_lower:
        return "SUSPICIOUS"
    return "LEGITIMATE"


def _risk_level(classification: str, risk_score: float) -> str:
    if classification == "LEGITIMATE":
        return "LOW"
    if classification == "SUSPICIOUS":
        return "MEDIUM"
    return "CRITICAL" if risk_score >= CRITICAL_RISK_SCORE else "HIGH"


def _decide(risk_score: float, confidence: float, signal_count: int,
            trust: Optional[TrustAssessment] = None, bec: bool = False) -> dict:
    """
    Verdict from the model score, then the rule layer:
      1. Trusted sender, links aligned, no BEC lure (api/trust.py) -> lower the verdict.
         Known brand: LEGITIMATE. Authenticated but unknown domain: PHISHING becomes SUSPICIOUS.
      2. Otherwise the multi-signal override can raise a LEGITIMATE verdict, using the active
         thresholds as floors.
    """
    model_score = risk_score
    cls = _classify(risk_score)
    adjustment, reason = None, None
    level = trust.level if trust is not None and not bec else None
    if level == "known" and cls != "LEGITIMATE":
        cls, risk_score = "LEGITIMATE", min(risk_score, state.suspicious_lower / 2)
        adjustment, reason = "trusted_sender", "; ".join(trust.reasons)
    elif level == "verified" and cls == "PHISHING":
        mid = (state.suspicious_lower + state.phishing_threshold) / 2
        cls, risk_score = "SUSPICIOUS", min(risk_score, mid)
        adjustment, reason = "trusted_sender", "; ".join(trust.reasons)
    elif cls == "LEGITIMATE":
        if signal_count >= SIGNAL_OVERRIDE_PHISHING:
            cls, risk_score = "PHISHING", max(risk_score, state.phishing_threshold)
        elif signal_count >= SIGNAL_OVERRIDE_SUSPICIOUS:
            cls, risk_score = "SUSPICIOUS", max(risk_score, state.suspicious_lower)
        if cls != "LEGITIMATE":
            adjustment, reason = "signal_override", f"{signal_count} threat signals"
    return {
        "classification": cls,
        "risk_score": round(risk_score, 4),
        "confidence": round(confidence, 4),
        "risk_level": _risk_level(cls, risk_score),
        "threshold_used": round(state.phishing_threshold, 4),
        "model_score": round(model_score, 4),
        "adjustment": adjustment,
        "adjustment_reason": reason,
    }


def _verdict(req: PredictRequest, urls: List[str], signals: dict, risk_score: float,
             confidence: float) -> dict:
    """Model score + request evidence -> final decision (shared by /predict and /explain)."""
    trust = assess(req.sender, req.reply_to, req.headers, urls)
    result = _decide(risk_score, confidence, _signal_count(signals, url_extract(urls)),
                     trust, signals["bec_detected"])
    result["trust"] = trust
    return result


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _model_text(req: PredictRequest, body: str, body_offset: int, urls: List[str]):
    """
    Model text for one body (or body chunk) in the checkpoint's own input format. `body_offset` is
    where `body` starts inside req.body_text, so explanation offsets map back to the original body.
    """
    if state.db_config.get("text_format", "body_only") == "body_only":
        return body, [Span("body", None, 0, len(body), body_offset)]
    text, spans = build_model_text_with_spans(req.subject, body, urls)
    return text, [sp._replace(field_offset=sp.field_offset + body_offset) if sp.field == "body" else sp
                  for sp in spans]


def _body_chunks(body: str) -> List[Tuple[int, str]]:
    """Overlapping ~WINDOW_CHARS chunks of the body, cut at whitespace, as (start offset, text)."""
    chunks, start, n = [], 0, len(body)
    while start < n:
        end = min(n, start + WINDOW_CHARS)
        if end < n:
            cut = body.rfind(" ", start + WINDOW_CHARS // 2, end)
            end = cut if cut > start else end
        chunks.append((start, body[start:end]))
        if end >= n:
            break
        nxt = body.find(" ", max(start + 1, end - WINDOW_OVERLAP_CHARS), end)
        start = nxt + 1 if nxt > start else end
    return chunks


def _n_tokens(text: str) -> int:
    ids = state.db_tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
    return int(ids.shape[-1]) if hasattr(ids, "shape") else len(ids)


def _model_windows(req: PredictRequest, urls: List[str]) -> List[Tuple[str, List[Span]]]:
    """
    Texts the model scores for this email. A text that fits in max_length is one window (scored
    exactly as before). A longer one is the head window plus up to MAX_EXTRA_WINDOWS body chunks
    (always the last one), each repeating the subject and URL lines, so content and links after the
    first 256 tokens are read too (audit B3).
    """
    max_len = int(state.db_config.get("max_length", 256))
    head = _model_text(req, req.body_text or "", 0, urls)
    # A token is at least one character, so a prefix of 16 chars per token is plenty to decide.
    if _n_tokens(head[0][: max_len * 16]) <= max_len:
        return [head]
    chunks = _body_chunks(req.body_text or "")[1:]   # the first chunk is inside the head window
    if len(chunks) > MAX_EXTRA_WINDOWS:
        picks = np.linspace(0, len(chunks) - 1, MAX_EXTRA_WINDOWS).round().astype(int)
        chunks = [chunks[i] for i in dict.fromkeys(picks.tolist())]
    return [head] + [_model_text(req, chunk, start, urls) for start, chunk in chunks]


def _header_values(req: PredictRequest) -> List[float]:
    return _prepare_distilbert_inputs(req, [])[2]


def _score_windows(windows: List[str], hdr_values: List[float]) -> np.ndarray:
    """Phishing-class softmax rows for each window text, in one batched forward pass."""
    import torch

    max_len = int(state.db_config.get("max_length", 256))
    enc = state.db_tokenizer(windows if len(windows) > 1 else windows[0], max_length=max_len,
                             truncation=True, padding="max_length", return_tensors="pt")
    hdr_tensor = torch.tensor([hdr_values] * len(windows), dtype=torch.float32)
    with torch.no_grad():
        logits = state.db_model(enc["input_ids"], enc["attention_mask"], hdr_tensor)
        return torch.softmax(logits, dim=1).numpy()


def _prepare_distilbert_inputs(req: PredictRequest, urls: List[str]):
    """
    Request -> (model text, field spans, header values, header baseline) in the checkpoint's own
    input format. Used by both /predict and /explain so the two always see the same input.

    Older checkpoints were trained on body text alone with zeroed header features; the config
    flags say so. New checkpoints (retrained with the fixed pipeline) set both flags.
    """
    cfg = state.db_config
    text, spans = _model_text(req, req.body_text or "", 0, urls)

    if cfg.get("header_features_trained", False):
        hdr = header_extract(req.headers or {}, req.sender or "", req.reply_to or "")
        hdr_values = [float(hdr[k]) for k in HEADER_FEATURE_NAMES]
        empty = header_extract({}, "", "")   # explanation baseline: "no headers provided"
        hdr_baseline = [float(empty[k]) for k in HEADER_FEATURE_NAMES]
    else:
        hdr_values = [0.0] * len(HEADER_FEATURE_NAMES)
        hdr_baseline = list(hdr_values)
    return text, spans, hdr_values, hdr_baseline


def _score_distilbert(req: PredictRequest, urls: List[str]) -> Tuple[float, float]:
    """
    Returns (phishing probability, model confidence) using the checkpoint's own input format.
    Long emails: the highest-scoring window decides (see _model_windows).
    """
    windows = _model_windows(req, urls)
    proba = _score_windows([w[0] for w in windows], _header_values(req))
    best = int(np.argmax(proba[:, 1]))
    return float(proba[best, 1]), float(max(proba[best]))


def _score_rf(req: PredictRequest, urls: List[str]) -> Tuple[float, float]:
    import pandas as pd
    from scipy.sparse import csr_matrix, hstack

    df = pd.DataFrame([{
        "body_text": req.body_text or "",
        "subject": req.subject or "",
        "urls": json.dumps(urls),
        "sender": req.sender or "",
        "reply_to": req.reply_to or "",
        "headers_available": json.dumps(req.headers or {}),
    }])
    X_text = state.text_extractor.transform(df)

    url_feat = url_extract(urls)
    X_url = csr_matrix(np.array([url_feat[k] for k in URL_FEATURE_NAMES],
                                dtype=np.float32).reshape(1, -1))

    hdr = header_extract(req.headers or {}, req.sender or "", req.reply_to or "")
    X_hdr = csr_matrix(np.array([hdr[k] for k in HEADER_FEATURE_NAMES],
                                dtype=np.float32).reshape(1, -1))

    proba = state.rf_model.predict_proba(hstack([X_text, X_url, X_hdr]))[0]
    return float(proba[1]), float(max(proba))


# ---------------------------------------------------------------------------
# Admin auth
# ---------------------------------------------------------------------------
def require_admin(authorization: Optional[str] = Header(default=None)) -> None:
    """
    Bearer-token guard. With no NETRA_ADMIN_TOKEN configured the endpoint looks
    absent (404), so it cannot be used by accident in production.
    """
    expected = os.getenv("NETRA_ADMIN_TOKEN", "")
    if not expected:
        raise HTTPException(status_code=404, detail="Not found")
    presented = ""
    if authorization and authorization.startswith("Bearer "):
        presented = authorization[len("Bearer "):]
    if not hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid admin token",
                            headers={"WWW-Authenticate": "Bearer"})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["health"])
def health():
    body = HealthResponse(
        status="ok" if state.model_loaded else "not_ready",
        model_type=state.model_type,
        model_loaded=state.model_loaded,
        distilbert_ready=state.db_ready,
        thresholds={
            "phishing": round(state.phishing_threshold, 4),
            "suspicious_lower": round(state.suspicious_lower, 4),
            "suspicious_upper": round(state.suspicious_upper, 4),
        },
        threshold_source=state.threshold_source,
    )
    return JSONResponse(status_code=200 if state.model_loaded else 503, content=body.model_dump())


def _check_ready(req: PredictRequest) -> None:
    if not state.model_loaded:
        raise HTTPException(status_code=503, detail={
            "error": "model_not_loaded",
            "message": "No model loaded. Train with ml/train_distilbert.py or place the weights in ml/models/.",
        })
    # HTML-only / image-only mail has no body text; subject and links are still worth scoring.
    if not (req.body_text or "").strip() and not (req.subject or "").strip() and not req.urls:
        raise HTTPException(status_code=422, detail="Provide body_text, subject or urls.")


def _explain_distilbert(req: PredictRequest, urls: List[str]):
    """
    Integrated Gradients on the served input. For a long email that is the window /predict took
    its score from, so both endpoints report the same score. Returns (IGResult, hdr_values,
    headers_used). Header attributions are only reported when the checkpoint learned auth results.
    """
    from ml.explain import integrated_gradients

    cfg = state.db_config
    _text, _spans, hdr_values, hdr_baseline = _prepare_distilbert_inputs(req, urls)
    windows = _model_windows(req, urls)
    best = 0
    if len(windows) > 1:
        best = int(np.argmax(_score_windows([w[0] for w in windows], hdr_values)[:, 1]))
    text, spans = windows[best]
    ig = integrated_gradients(state.db_model, state.db_tokenizer, text, spans, hdr_values, hdr_baseline,
                              max_length=int(cfg.get("max_length", 256)), n_steps=_xai_steps())
    headers_used = bool(cfg.get("header_features_trained", False)) and cfg.get("header_auth_learned", True)
    return ig, hdr_values, bool(headers_used)


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(req: PredictRequest):
    _check_ready(req)

    t0 = time.perf_counter()
    try:
        urls = extract_urls(req.subject, req.body_text, req.urls)
        signals = _extract_signals(req, urls)

        if state.inference_slots is None:
            state.inference_slots = asyncio.Semaphore(_max_concurrency())
        score_fn = _score_distilbert if state.db_ready else _score_rf
        async with state.inference_slots:
            # Model inference is CPU-bound. Run it in a worker thread so /health and
            # other requests are not blocked while it runs.
            risk_score, confidence = await asyncio.to_thread(score_fn, req, urls)

        result = _verdict(req, urls, signals, risk_score, confidence)
        elapsed = round((time.perf_counter() - t0) * 1000, 2)

        log.info(f"Predict [{state.model_type}]: {result['classification']} "
                 f"score={result['risk_score']:.4f} model={result['model_score']:.4f} "
                 f"adjustment={result['adjustment']} elapsed={elapsed}ms")

        return PredictResponse(
            classification=result["classification"],
            risk_score=result["risk_score"],
            confidence=result["confidence"],
            risk_level=result["risk_level"],
            model_type=state.model_type,
            threshold_used=result["threshold_used"],
            signals=SignalsDict(**signals),
            processing_time_ms=elapsed,
            model_score=result["model_score"],
            adjustment=result["adjustment"],
            adjustment_reason=result["adjustment_reason"],
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/explain", response_model=ExplainResponse, tags=["inference"])
async def explain(req: ExplainRequest):
    """
    Same verdict as /predict plus an explanation: Integrated Gradients over the model's word
    embeddings and header features, fused with the multi-signal override (see api/xai.py).
    With the RF fallback only the rule signals are explained (method 'signals_only').
    """
    _check_ready(req)

    t0 = time.perf_counter()
    try:
        urls = extract_urls(req.subject, req.body_text, req.urls)
        signals = _extract_signals(req, urls)

        if state.explain_slots is None:
            state.explain_slots = asyncio.Semaphore(1)
        async with state.explain_slots:
            if state.db_ready:
                ig, hdr_values, headers_used = await asyncio.to_thread(_explain_distilbert, req, urls)
                # Same forward pass as _score_distilbert (unpadded; padding is masked out).
                risk_score = sigmoid(ig.f_input)
                confidence = max(risk_score, 1.0 - risk_score)
            else:
                ig, hdr_values, headers_used = None, [], False
                risk_score, confidence = await asyncio.to_thread(_score_rf, req, urls)

        result = _verdict(req, urls, signals, risk_score, confidence)
        hdrs = req.headers or {}
        explanation = fuse(
            ig, signals, url_extract(urls), urls,
            auth={k: normalize_auth(hdrs.get(k)) for k in ("spf", "dkim", "dmarc")},
            reply_to_matches=bool(header_extract(hdrs, req.sender or "", req.reply_to or "")
                                  ["sender_domain_matches_reply_to"]),
            decided=result, model_classification=_classify(risk_score), p_model=risk_score,
            hdr_values=hdr_values, headers_used=headers_used, top_k=req.top_k,
            trust_reasons=result["trust"].reasons,
        )
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        explanation.explain_time_ms = elapsed

        log.info(f"Explain [{state.model_type}]: {result['classification']} score={result['risk_score']:.4f} "
                 f"delta={explanation.completeness_delta} elapsed={elapsed}ms")

        return ExplainResponse(
            classification=result["classification"],
            risk_score=result["risk_score"],
            confidence=result["confidence"],
            risk_level=result["risk_level"],
            model_type=state.model_type,
            threshold_used=result["threshold_used"],
            signals=SignalsDict(**signals),
            processing_time_ms=elapsed,
            model_score=result["model_score"],
            adjustment=result["adjustment"],
            adjustment_reason=result["adjustment_reason"],
            explanation=explanation,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Explain error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Explanation failed")


@app.post("/admin/recalibrate", response_model=RecalibrateResponse, tags=["admin"],
          dependencies=[Depends(require_admin)])
def recalibrate(req: RecalibrateRequest):
    if not _valid_thresholds(req.suspicious_lower, req.phishing_threshold):
        raise HTTPException(status_code=422,
                            detail=f"Require 0 < suspicious_lower, phishing_threshold <= 1 and a SUSPICIOUS "
                                   f"band of at least {MIN_SUSPICIOUS_BAND}.")

    updates = {
        "phishing_threshold": req.phishing_threshold,
        "suspicious_lower": req.suspicious_lower,
        "suspicious_upper": req.phishing_threshold,
        "method": "manual_recalibrate",
    }
    try:
        _save_thresholds(updates)
    except OSError as e:
        # Persist first, then mutate state. On failure the live thresholds stay unchanged.
        log.error(f"Could not persist thresholds: {e}")
        raise HTTPException(status_code=500, detail="Could not persist thresholds; no change applied.")

    state.phishing_threshold = req.phishing_threshold
    state.suspicious_lower = req.suspicious_lower
    state.suspicious_upper = req.phishing_threshold
    state.threshold_source = "manual_recalibrate"
    log.info(f"Thresholds updated: phishing>={req.phishing_threshold} suspicious>={req.suspicious_lower}")

    return RecalibrateResponse(
        status="ok",
        new_thresholds={
            "phishing": state.phishing_threshold,
            "suspicious_lower": state.suspicious_lower,
            "suspicious_upper": state.suspicious_upper,
        },
    )
