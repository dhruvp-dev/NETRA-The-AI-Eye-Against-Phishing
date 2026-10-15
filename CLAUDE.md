# CLAUDE.md — NETRA Engineering & Development Guide

Welcome to **NETRA (The AI Eye Against Phishing)**! This document provides essential architectural context, command workflows, engineering guidelines, and operational patterns for Claude Code.

---

## 🛡️ Project Overview & Architecture

NETRA is a high-throughput, multi-signal AI phishing detection engine with a two-tier escalation architecture:

```
[ Incoming Email: Subject, Body, Sender, RFC Headers ]
                          │
                          ▼
            [ Tier-1 Real-Time Engine ]
    (DistilBERT 66M + RFC Auth + Typosquatting)
             Latency: ~180ms | Port: 8000
                          │
         ┌────────────────┼────────────────┐
         ▼                                 ▼
   Decisive Score                    Ambiguous Score
   (<0.08 Legitimate,                (0.08 <= score < 0.25)
    >=0.25 Phishing)                       │
         │                                 ▼
         │                     [ Tier-2 Deep Escalation ]
         │                     (RoBERTa-base 125M + Captum XAI)
         │                     Latency: ~800ms | Port: 8001
         │                                 │
         └────────────────┬────────────────┘
                          ▼
             [ Unified Phishing Verdict ]
```

### Core Components
1. **Tier-1 API Gateway (`api/main.py`)**:
   - FastAPI service running on port `8000`.
   - Embeds fine-tuned DistilBERT (`ml/models/distilbert_tier1.pt`, max length 256).
   - Extracts 10 RFC header features (SPF, DKIM, DMARC auth results).
   - Computes domain typosquatting distances (Levenshtein against top brand domains) and suspicious TLD counts.
   - Blends semantic probability with heuristic risk boost.
   - Built-in rate limiter: sliding-window, 20 requests/minute per client IP (reads `X-Forwarded-For` for Cloud Run reverse proxies).
   - Asynchronously forwards ambiguous emails (`0.08 <= score < 0.25`) to Tier-2 if configured.

2. **Tier-2 Deep Inspection Service (`api/tier2_service/`)**:
   - Microservice running on port `8001` (containerizable via `Dockerfile`).
   - Uses fine-tuned RoBERTa-base (`ml/models/roberta_tier2.pt`, max length 512) for 3-class classification (`LEGITIMATE`, `SUSPICIOUS`, `PHISHING`).
   - Meta Captum Integrated Gradients for Explainable AI (XAI) token attributions (`api/tier2_service/xai.py`).

3. **Chrome Extension (`extension/`)**:
   - Manifest V3 browser extension for scanning Gmail and active web pages.
   - Highlights risky domains and displays NETRA threat score badges.

---

## ⚡ Essential Commands

### Environment Setup
```bash
# Python 3.10+ recommended (virtualenv recommended)
pip install -r requirements.txt
```

### Running Local Servers
```bash
# Start Tier-1 API (Port 8000)
python -m uvicorn api.main:app --reload --port 8000

# Start Tier-2 Deep Service (Port 8001)
python -m uvicorn api.tier2_service.main:app --reload --port 8001
```

### Running Tests
```bash
# Run test suite
pytest tests/

# Run Tier-1 stress and latency benchmarking
python tests/stress_test_tier1.py
```

### Model Training & Calibration
```bash
# Train Tier-1 DistilBERT (Local NVIDIA GPU)
python ml/train_distilbert.py --epochs 3 --batch_size 16 --fp16

# Train Tier-2 RoBERTa (Local NVIDIA GPU)
python ml/train_roberta_tier2.py --epochs 6 --batch_size 16 --fp16

# Recalibrate Thresholds
python ml/calibrate_threshold.py
python ml/calibrate_tier2_threshold.py
```

---

## 📦 Model Weights & GitHub Release Rules

> **CRITICAL**: Never commit `.pt` or binary model files to Git! They exceed GitHub's 100MB file limit.

- **Weight Storage**:
  - Tier-1: `ml/models/distilbert_tier1.pt` (~268 MB)
  - Tier-2: `ml/models/roberta_tier2.pt` (~499 MB)
  - Both paths are strictly included in `.gitignore`.
- **Automatic Fallback Download**:
  - If weights are absent on startup, both `api/main.py` and `api/tier2_service/model.py` automatically download them from GitHub Releases:
    - Tier-1 weights download from Release `v2.0.0`
    - Tier-2 weights download from Release `v3.0.0`
- **Publishing New Weights**:
  - Always use `gh release create <tag> <file> --title <title> --notes <notes>` to distribute updated model checkpoints.

---

## 📁 Repository Map

```text
NETRA/
├── CLAUDE.md                       # Claude Code developer guide (this file)
├── README.md                       # Public documentation & Swagger guide
├── requirements.txt                # Unified Python dependencies
├── api/
│   ├── main.py                     # Tier-1 FastAPI gateway, heuristics & rate limiter
│   ├── tier2_client.py             # Async client communicating with Tier-2
│   └── tier2_service/
│       ├── Dockerfile              # Docker container for Tier-2 deployment
│       ├── main.py                 # Tier-2 FastAPI app (Port 8001)
│       ├── model.py                # RoBERTa-base classifier & auto-downloader
│       ├── xai.py                  # Captum Integrated Gradients token attribution
│       └── requirements.txt        # Isolated Tier-2 container requirements
├── ml/
│   ├── data_pipeline.py            # Dataset loading, cleaning, tokenization
│   ├── train_distilbert.py         # PyTorch training pipeline for Tier-1
│   ├── train_roberta_tier2.py      # PyTorch training pipeline for Tier-2
│   ├── calibrate_threshold.py      # Precision/Recall threshold optimization
│   ├── calibrate_tier2_threshold.py# Tier-2 threshold optimizer
│   ├── generate_tier2_dataset.py   # Synthesizes hard-negative & escalation datasets
│   ├── evaluation/                 # Metrics JSON & evaluation logs
│   ├── features/                   # Heuristic feature extractors
│   │   ├── header_features.py      # SPF/DKIM/DMARC RFC compliance parser
│   │   ├── text_features.py        # Urgency & social-engineering lexical features
│   │   └── url_features.py         # Levenshtein typosquatting & domain parsing
│   └── models/
│       ├── distilbert_config.json  # Metadata & active threshold boundaries
│       └── distilbert_tokenizer/   # DistilBERT tokenizer vocabulary
├── extension/                      # Manifest V3 Chrome Extension
│   ├── manifest.json               # Extension manifest
│   ├── popup.html / popup.js       # Extension UI & scanner
│   ├── content.js                  # Gmail DOM hook & scanner
│   └── service-worker.js           # Background extension worker
├── notebooks/                      # Google Colab training notebooks
└── tests/                          # Integration & stress testing
```

---

## 📐 Coding Conventions & Guidelines

1. **Python & Framework Standards**:
   - Target Python 3.10+.
   - Use Pydantic v2 schemas for all API request and response models.
   - Use async route handlers (`async def`) for endpoints that perform I/O or forward requests (`tier2_client.py`).
2. **Device Independence**:
   - Always resolve PyTorch device dynamically:
     ```python
     device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
     ```
   - Avoid hardcoding `cuda:0` without fallback to CPU.
3. **Threshold Consistency**:
   - Threshold definitions reside in `ml/models/distilbert_config.json`.
   - Default boundaries:
     - `score < 0.08`: **LEGITIMATE**
     - `0.08 <= score < 0.25`: **SUSPICIOUS**
     - `score >= 0.25`: **PHISHING**
4. **Cloud Run / Reverse Proxy Support**:
   - The rate limiter in `api/main.py` extracts IP addresses from `X-Forwarded-For` if present. Keep this proxy-aware pattern when modifying networking logic.
