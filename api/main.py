"""
NETRA - FastAPI Inference Server
==================================
Serves Tier-1 phishing detection model over a local REST API.

Start server:
    uvicorn api.main:app --host 127.0.0.1 --port 8000

Endpoints:
    GET  /health      → {"status": "ok", "model_loaded": bool}
    POST /predict     → classification + risk_score + confidence + tier

IMPORTANT: This server binds to 127.0.0.1 only (not exposed to the network).
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "ml" / "models" / "tier1_model.pkl"
TFIDF_PATH = ROOT / "ml" / "models" / "tfidf_vectorizer.pkl"

# ---------------------------------------------------------------------------
# Risk thresholds
# ---------------------------------------------------------------------------
THRESHOLD_LEGITIMATE = 0.35   # below → LEGITIMATE
THRESHOLD_PHISHING = 0.65     # above → PHISHING
# between → SUSPICIOUS

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NETRA Phishing Detection API",
    description="Tier-1 ML-based email phishing detection. Local use only.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow only localhost origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        # Chrome extension uses chrome-extension:// origin
        "chrome-extension://*",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

class ModelState:
    model = None
    text_extractor = None
    loaded = False
    error: Optional[str] = None


state = ModelState()


@app.on_event("startup")
def load_models():
    """Load model artefacts on startup. Failure is non-fatal."""
    log.info("Loading NETRA Tier-1 model artefacts...")

    if not MODEL_PATH.exists():
        state.error = "model_not_trained"
        log.warning(f"Model file not found: {MODEL_PATH}")
        log.warning("Start the server anyway — /predict will return 503 until model is trained.")
        return

    if not TFIDF_PATH.exists():
        state.error = "tfidf_not_found"
        log.warning(f"TF-IDF vectorizer not found: {TFIDF_PATH}")
        return

    try:
        state.model = joblib.load(MODEL_PATH)
        state.text_extractor = joblib.load(TFIDF_PATH)
        state.loaded = True
        state.error = None
        log.info("Models loaded successfully.")
    except Exception as e:
        state.error = str(e)
        log.error(f"Failed to load models: {e}")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    body_text: str = Field(..., description="Email body content")
    urls: Optional[List[str]] = Field(default=[], description="List of URLs found in the email")
    sender: Optional[str] = Field(default=None, description="Sender email address")
    reply_to: Optional[str] = Field(default=None, description="Reply-To email address")
    headers: Optional[Dict[str, Any]] = Field(default=None, description="Parsed email headers dict")

    class Config:
        json_schema_extra = {
            "example": {
                "body_text": "Your account has been suspended. Click here immediately to verify.",
                "urls": ["http://phish.example.tk/login"],
                "sender": "security@bank-alerts.com",
                "reply_to": "reply@harvester.net",
                "headers": {"spf": "fail", "dkim": "none", "dmarc": "fail"}
            }
        }


class PredictResponse(BaseModel):
    classification: str   # "LEGITIMATE" | "PHISHING" | "SUSPICIOUS"
    risk_score: float     # raw phishing probability 0.0–1.0
    confidence: float     # distance from the nearest threshold
    tier: int = 1         # always 1 for this model


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Feature assembly (mirrors train.py logic but for single prediction)
# ---------------------------------------------------------------------------

def build_single_record_df(req: PredictRequest):
    """Convert a PredictRequest into a single-row DataFrame for feature extraction."""
    import pandas as pd

    return pd.DataFrame([{
        "body_text": req.body_text or "",
        "subject": "",          # not exposed in API — included as empty
        "urls": json.dumps(req.urls or []),
        "sender": req.sender or "",
        "reply_to": req.reply_to or "",
        "headers_available": json.dumps(req.headers or {}),
    }])


def predict_single(req: PredictRequest) -> dict:
    """
    Run the full feature pipeline on a single request and return
    classification, risk_score, and confidence.
    """
    from scipy.sparse import hstack, csr_matrix
    from ml.features.url_features import extract as url_extract, URL_FEATURE_NAMES
    from ml.features.header_features import extract as header_extract, HEADER_FEATURE_NAMES

    df = build_single_record_df(req)

    # Text features (TF-IDF + handcrafted)
    X_text = state.text_extractor.transform(df)

    # URL features
    urls = req.urls or []
    url_feat = url_extract(urls)
    X_url = csr_matrix(
        np.array([url_feat[k] for k in URL_FEATURE_NAMES], dtype=np.float32).reshape(1, -1)
    )

    # Header features
    headers = req.headers or {}
    header_feat = header_extract(
        headers_dict=headers,
        sender=req.sender or "",
        reply_to=req.reply_to or "",
    )
    X_header = csr_matrix(
        np.array([header_feat[k] for k in HEADER_FEATURE_NAMES], dtype=np.float32).reshape(1, -1)
    )

    X = hstack([X_text, X_url, X_header])

    # Predict
    proba = state.model.predict_proba(X)[0]
    risk_score = float(proba[1])  # probability of phishing class

    # Classification thresholds
    if risk_score < THRESHOLD_LEGITIMATE:
        classification = "LEGITIMATE"
        confidence = THRESHOLD_LEGITIMATE - risk_score
    elif risk_score > THRESHOLD_PHISHING:
        classification = "PHISHING"
        confidence = risk_score - THRESHOLD_PHISHING
    else:
        classification = "SUSPICIOUS"
        # Confidence: distance to nearer threshold
        confidence = min(
            risk_score - THRESHOLD_LEGITIMATE,
            THRESHOLD_PHISHING - risk_score,
        )

    return {
        "classification": classification,
        "risk_score": round(risk_score, 4),
        "confidence": round(confidence, 4),
        "tier": 1,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health():
    """
    Returns service health status and whether the model is loaded.
    """
    return HealthResponse(
        status="ok",
        model_loaded=state.loaded,
        error=state.error,
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(req: PredictRequest):
    """
    Classify an email as LEGITIMATE, PHISHING, or SUSPICIOUS.

    Returns:
        classification: LEGITIMATE | PHISHING | SUSPICIOUS
        risk_score:     phishing probability 0.0–1.0
        confidence:     margin from the nearest decision boundary
        tier:           1 (Tier-1 ML model)
    """
    # Model not loaded → 503
    if not state.loaded:
        raise HTTPException(
            status_code=503,
            detail={
                "error": state.error or "model_not_loaded",
                "message": (
                    "The ML model has not been trained yet. "
                    "Train the model first using ml/train.py and download "
                    "the .pkl files to ml/models/."
                ),
            },
        )

    if not req.body_text or not req.body_text.strip():
        raise HTTPException(
            status_code=422,
            detail="body_text must not be empty.",
        )

    try:
        result = predict_single(req)
        log.info(
            f"Prediction: {result['classification']} "
            f"(score={result['risk_score']:.3f}, "
            f"confidence={result['confidence']:.3f})"
        )
        return PredictResponse(**result)
    except Exception as e:
        log.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "path": str(request.url.path)},
    )
