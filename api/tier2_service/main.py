# -*- coding: utf-8 -*-
"""
NETRA - Tier-2 FastAPI Service
================================
Standalone FastAPI application for Tier-2 RoBERTa escalation inference.
Deployed to Google Cloud Run as a container.

Start locally:
    uvicorn api.tier2_service.main:app --host 0.0.0.0 --port 8080

Endpoints:
    GET  /health    -> service status
    POST /predict   -> 3-class verdict + XAI token attributions
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "ml" / "models"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NETRA Tier-2 Escalation Service",
    description="RoBERTa-based 3-class phishing classifier with XAI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class Tier2PredictRequest(BaseModel):
    subject:           Optional[str] = None
    body_text:         str
    urls:              Optional[List[str]] = None
    sender:            Optional[str] = None
    reply_to:          Optional[str] = None
    headers_available: Optional[Dict[str, Any]] = None
    tier1_risk_score:  float = 0.0
    tier1_confidence:  float = 0.5
    tier1_verdict:     str = "SUSPICIOUS"
    tier1_signals:     Optional[Dict[str, Any]] = None


class TokenAttribution(BaseModel):
    token:       str
    attribution: float
    position:    int = 0


class Tier2PredictResponse(BaseModel):
    verdict:              str
    risk_score:           float
    confidence:           float
    class_probabilities:  Dict[str, float]
    top_tokens:           List[TokenAttribution] = []
    signals:              Dict[str, Any] = {}
    tier1_verdict:        str = ""
    tier1_risk_score:     float = 0.0
    processing_time_ms:   float = 0.0
    xai_latency_ms:       float = 0.0
    model_type:           str = "roberta-tier2"


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    model_type:   str = "roberta-tier2"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class ServiceState:
    def __init__(self):
        self.tier2_model = None
        self.xai_engine = None
        self.ready = False

state = ServiceState()


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    from api.tier2_service.model import Tier2Model
    from api.tier2_service.xai import TokenAttributor

    state.tier2_model = Tier2Model(MODELS_DIR)
    loaded = state.tier2_model.load()

    if loaded:
        state.xai_engine = TokenAttributor(
            model=state.tier2_model.model,
            tokenizer=state.tier2_model.tokenizer,
            device=state.tier2_model.device,
            top_k=10,
        )
        state.ready = True
        log.info("Tier-2 service ready")
    else:
        log.error("Failed to load Tier-2 model")


# ---------------------------------------------------------------------------
# Header feature extraction
# ---------------------------------------------------------------------------
HEADER_FEATURE_NAMES = [
    "spf_pass", "spf_fail", "spf_none",
    "dkim_pass", "dkim_fail", "dkim_none",
    "dmarc_pass", "dmarc_fail", "dmarc_none",
    "sender_domain_match",
]


def extract_header_features(headers: dict, sender: str = "", reply_to: str = "") -> list:
    """Extract 10-dim header feature vector."""
    try:
        sys.path.insert(0, str(ROOT))
        from ml.features.header_features import extract as header_extract
        feat = header_extract(headers or {}, sender=sender, reply_to=reply_to)
        return [float(feat.get(k, 0)) for k in HEADER_FEATURE_NAMES]
    except ImportError:
        # Fallback: manual extraction
        _auth_states = ["pass", "fail", "none"]
        features = []
        for proto in ["spf", "dkim", "dmarc"]:
            val = str(headers.get(proto, "none")).lower().strip()
            if val not in _auth_states:
                val = "none"
            for s in _auth_states:
                features.append(1.0 if val == s else 0.0)
        # sender_domain_match
        features.append(1.0)  # default to match
        return features


def build_input_text(req: Tier2PredictRequest) -> str:
    """Build enriched text prompt for RoBERTa."""
    parts = []
    if req.subject and req.subject.strip():
        parts.append(f"Subject: {req.subject.strip()}")
    if req.body_text and req.body_text.strip():
        parts.append(f"Body: {req.body_text.strip()}")
    if req.urls:
        for u in req.urls[:5]:
            parts.append(f"URL: {u}")
    return "\n".join(parts) if parts else req.body_text


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health():
    return HealthResponse(
        status="ok" if state.ready else "not_ready",
        model_loaded=state.ready,
    )


@app.post("/predict", response_model=Tier2PredictResponse, tags=["inference"])
async def predict(req: Tier2PredictRequest):
    if not state.ready:
        raise HTTPException(status_code=503, detail="Tier-2 model not loaded")

    if not req.body_text or not req.body_text.strip():
        raise HTTPException(status_code=422, detail="body_text must not be empty")

    t0 = time.perf_counter()

    # Build input
    text = build_input_text(req)
    header_feats = extract_header_features(
        req.headers_available or {},
        sender=req.sender or "",
        reply_to=req.reply_to or "",
    )

    # Predict
    result = state.tier2_model.predict(
        text=text,
        header_features=header_feats,
        tier1_confidence=req.tier1_confidence,
    )

    # XAI
    xai_result = {"top_tokens": [], "xai_latency_ms": 0}
    if state.xai_engine and result["verdict"] != "LEGITIMATE":
        try:
            xai_result = state.xai_engine.explain(
                text=text,
                header_features=header_feats,
                tier1_confidence=req.tier1_confidence,
            )
        except Exception as e:
            log.error(f"XAI error: {e}")

    elapsed = round((time.perf_counter() - t0) * 1000, 2)

    top_tokens = [
        TokenAttribution(**t) for t in xai_result.get("top_tokens", [])
    ]

    return Tier2PredictResponse(
        verdict=result["verdict"],
        risk_score=result["risk_score"],
        confidence=result["confidence"],
        class_probabilities=result["class_probabilities"],
        top_tokens=top_tokens,
        signals=req.tier1_signals or {},
        tier1_verdict=req.tier1_verdict,
        tier1_risk_score=req.tier1_risk_score,
        processing_time_ms=elapsed,
        xai_latency_ms=xai_result.get("xai_latency_ms", 0),
    )
