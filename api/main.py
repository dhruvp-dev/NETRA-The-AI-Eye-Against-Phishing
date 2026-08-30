"""
NETRA - FastAPI Inference Server
==================================
Serves Tier-1 phishing detection model over a local REST API.

Start server:
    uvicorn api.main:app --host 127.0.0.1 --port 8000

Endpoints:
    GET  /health      -> {"status": "ok", "model_loaded": bool, "model_type": str, "thresholds": {...}}
    POST /predict     -> classification + risk_score + confidence + tier + threshold_used + model_type

IMPORTANT: This server binds to 127.0.0.1 only (not exposed to the network).
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

# --- Add project root to sys.path ---
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
MODELS_DIR              = ROOT / "ml" / "models"
MODEL_CALIBRATED_PATH   = MODELS_DIR / "tier1_model_calibrated.pkl"
MODEL_PATH              = MODELS_DIR / "tier1_model.pkl"
TFIDF_PATH              = MODELS_DIR / "tfidf_vectorizer.pkl"
THRESHOLD_CONFIG_PATH   = MODELS_DIR / "threshold_config.json"

# ---------------------------------------------------------------------------
# Safe fallback thresholds (empirically derived from raw RF score range 0.03-0.24)
# ---------------------------------------------------------------------------
DEFAULT_PHISHING_THRESHOLD  = 0.15
DEFAULT_SUSPICIOUS_LOWER    = 0.09

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NETRA Phishing Detection API",
    description="Tier-1 ML-based email phishing detection. Local use only.",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
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
    model         = None
    text_extractor = None
    loaded        = False
    error: Optional[str] = None
    model_type: str = "uncalibrated"
    phishing_threshold: float  = DEFAULT_PHISHING_THRESHOLD
    suspicious_lower: float    = DEFAULT_SUSPICIOUS_LOWER
    suspicious_upper: float    = DEFAULT_PHISHING_THRESHOLD
    threshold_source: str      = "default_fallback"

state = ModelState()


@app.on_event("startup")
def load_models():
    """Load model artifacts and threshold config on startup."""
    log.info("Loading NETRA Tier-1 model artifacts...")

    # --- Load threshold config ---
    if THRESHOLD_CONFIG_PATH.exists():
        try:
            with open(THRESHOLD_CONFIG_PATH) as f:
                cfg = json.load(f)
            state.phishing_threshold  = cfg.get("phishing_threshold",  DEFAULT_PHISHING_THRESHOLD)
            state.suspicious_lower    = cfg.get("suspicious_lower",    DEFAULT_SUSPICIOUS_LOWER)
            state.suspicious_upper    = cfg.get("suspicious_upper",    state.phishing_threshold)
            state.threshold_source    = cfg.get("method", "threshold_config.json")
            log.info(f"Loaded threshold config: phishing>={state.phishing_threshold:.4f} | suspicious [{state.suspicious_lower:.4f}, {state.suspicious_upper:.4f})")
        except Exception as e:
            log.warning(f"Could not load threshold_config.json: {e} — using defaults")
    else:
        log.warning(f"threshold_config.json not found — using safe defaults: phishing>={DEFAULT_PHISHING_THRESHOLD}, suspicious>={DEFAULT_SUSPICIOUS_LOWER}")

    # --- Load model (calibrated preferred, fall back to raw) ---
    if MODEL_CALIBRATED_PATH.exists():
        model_path = MODEL_CALIBRATED_PATH
        state.model_type = "calibrated"
    elif MODEL_PATH.exists():
        model_path = MODEL_PATH
        state.model_type = "uncalibrated"
    else:
        state.error = "model_not_trained"
        log.warning("No model file found. /predict will return 503 until model is trained.")
        return

    if not TFIDF_PATH.exists():
        state.error = "tfidf_not_found"
        log.warning(f"TF-IDF vectorizer not found: {TFIDF_PATH}")
        return

    try:
        state.model          = joblib.load(model_path)
        state.text_extractor = joblib.load(TFIDF_PATH)
        state.loaded         = True
        state.error          = None
        log.info(f"Loaded model [{state.model_type}]: {model_path.name}")
        log.info(f"Thresholds: phishing>={state.phishing_threshold:.4f} | suspicious [{state.suspicious_lower:.4f}, {state.phishing_threshold:.4f})")
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
    classification: str
    risk_score: float
    confidence: float
    tier: int = 1
    threshold_used: float
    model_type: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    thresholds: Dict[str, float]
    threshold_source: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------
def predict_single(req: PredictRequest) -> dict:
    from scipy.sparse import hstack, csr_matrix
    from ml.features.url_features import extract as url_extract, URL_FEATURE_NAMES
    from ml.features.header_features import extract as header_extract, HEADER_FEATURE_NAMES
    import pandas as pd

    df = pd.DataFrame([{
        "body_text":          req.body_text or "",
        "subject":            "",
        "urls":               json.dumps(req.urls or []),
        "sender":             req.sender or "",
        "reply_to":           req.reply_to or "",
        "headers_available":  json.dumps(req.headers or {}),
    }])

    X_text = state.text_extractor.transform(df)

    urls     = req.urls or []
    url_feat = url_extract(urls)
    X_url    = csr_matrix(
        np.array([url_feat[k] for k in URL_FEATURE_NAMES], dtype=np.float32).reshape(1, -1)
    )

    headers     = req.headers or {}
    header_feat = header_extract(
        headers_dict=headers,
        sender=req.sender or "",
        reply_to=req.reply_to or "",
    )
    X_header = csr_matrix(
        np.array([header_feat[k] for k in HEADER_FEATURE_NAMES], dtype=np.float32).reshape(1, -1)
    )

    X          = hstack([X_text, X_url, X_header])
    proba      = state.model.predict_proba(X)[0]
    risk_score = float(proba[1])

    # --- Dynamic threshold classification ---
    pt = state.phishing_threshold
    sl = state.suspicious_lower

    if risk_score >= pt:
        classification = "PHISHING"
        confidence     = risk_score - pt
    elif risk_score >= sl:
        classification = "SUSPICIOUS"
        confidence     = min(risk_score - sl, pt - risk_score)
    else:
        classification = "LEGITIMATE"
        confidence     = sl - risk_score

    return {
        "classification": classification,
        "risk_score":     round(risk_score, 4),
        "confidence":     round(confidence, 4),
        "tier":           1,
        "threshold_used": round(pt, 4),
        "model_type":     state.model_type,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=state.loaded,
        model_type=state.model_type,
        thresholds={
            "phishing":         round(state.phishing_threshold, 4),
            "suspicious_lower": round(state.suspicious_lower, 4),
            "suspicious_upper": round(state.suspicious_upper, 4),
        },
        threshold_source=state.threshold_source,
        error=state.error,
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(req: PredictRequest):
    if not state.loaded:
        raise HTTPException(
            status_code=503,
            detail={
                "error": state.error or "model_not_loaded",
                "message": "The ML model has not been trained yet. Train the model first using ml/train.py.",
            },
        )

    if not req.body_text or not req.body_text.strip():
        raise HTTPException(status_code=422, detail="body_text must not be empty.")

    try:
        result = predict_single(req)
        log.info(
            f"Prediction: {result['classification']} "
            f"(score={result['risk_score']:.4f}, threshold={result['threshold_used']:.4f}, "
            f"model={result['model_type']})"
        )
        return PredictResponse(**result)
    except Exception as e:
        log.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "path": str(request.url.path)},
    )
