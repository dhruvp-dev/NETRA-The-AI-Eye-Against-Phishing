# CLAUDE.md — NETRA Engineering & Development Guide

This document is the working guide for Claude Code in this repository: architecture,
commands, conventions and operational rules.

---

## 🛡️ Project Overview & Architecture

NETRA is a phishing detection service. It is a single tier: a DistilBERT classifier
with RFC header features, served by FastAPI, with a calibrated Random Forest as fallback.

```
[ Incoming Email: Subject, Body, Sender, URLs, RFC Headers ]
                          │
                          ▼
   ml.features.text_builder ─ build_model_text(subject, body, urls)
   ml.features.header_features ─ 10 SPF / DKIM / DMARC / sender-match features
                          │
                          ▼
   DistilBERT [CLS] (768) ⊕ header projection (32) → classifier → P(phishing)
                          │
                          ▼
   Score rule (ml/models/threshold_config.json):
       score >= phishing_threshold          → PHISHING
       suspicious_lower <= score < phishing → SUSPICIOUS
       otherwise                            → LEGITIMATE
                          │
                          ▼
   Long emails: head window + up to 4 body windows (400 chars, subject and URL lines repeated);
   the highest window score is the model score.
                          │
                          ▼
   Trusted-sender rule (api/trust.py), unless a BEC lure fired:
       known brand domain + all links aligned          → LEGITIMATE (score capped at suspicious_lower / 2)
       authenticated unknown domain + links aligned    → PHISHING becomes SUSPICIOUS
                          │
                          ▼
   Multi-signal override (LEGITIMATE only; units from api/xai.py: override_units):
       >= 3 threat units → PHISHING     (score floored at phishing_threshold)
       >= 2 threat units → SUSPICIOUS   (score floored at suspicious_lower)
```

### Core Components

1. **API (`api/main.py`)**
   - FastAPI on port `8000`. Endpoints: `GET /health`, `POST /predict`, `POST /explain`, `POST /admin/recalibrate`.
   - Loads DistilBERT (`ml/models/distilbert_tier1.pt`). Falls back to the calibrated RF
     (`tier1_model_calibrated.pkl` or `tier1_model.pkl`) if DistilBERT is unavailable.
   - Inference runs in a worker thread (`asyncio.to_thread`) behind a semaphore (`NETRA_MAX_CONCURRENCY`).
   - Weights missing at startup are downloaded from GitHub Release `v2.0.0`, verified by SHA-256
     (`weights_sha256` in `distilbert_config.json`), then atomically moved into place.
2. **Explanations (`ml/explain.py`, `api/xai.py`)**
   - `POST /explain` takes the `/predict` body plus `top_k`. It returns the `/predict` fields plus an `explanation`.
   - `ml/explain.py`: Integrated Gradients over word embeddings and header features together. The target is the
     logit margin. The baseline replaces content tokens with `[SEP]` and uses "no headers" for the header input.
     Steps are placed adaptively (refined forward scan) because F jumps sharply along the path.
     Uses `PhishingClassifier.forward_embeds`.
   - `api/xai.py`: the model part (p_m − p0) and the override part (p_f − p_m, split equally per counted
     signal unit, or one negative `trusted_sender` component when the trust rule lowered the verdict) are
     reported per threat factor. Invariant: `baseline_score + Σ contributions = risk_score`.
   - For a long email, IG runs on the window that gave the model score.
   - `/explain` has its own one-slot semaphore and a 5/min limit on top of the shared one.
     It takes seconds on CPU and never runs inside `/predict`.
3. **Rule layer (`api/main.py: _verdict / _decide`, `api/trust.py`)**
   - Added after the Tier-1 audit (`bugs/tier1_model_bugs.md`) because the shipped checkpoint over-flags brand
     and transactional mail and cannot be retrained in time. It works around the model; it does not fix it.
   - `api/trust.py: assess` → `TrustAssessment.level` is `known` (sender's registrable domain in `KNOWN_ORGS`, not
     freemail), `verified` (DMARC pass, or SPF+DKIM pass) or `None`. Either level needs every link to be on the
     sender's org, another known org or `NEUTRAL_LINK_DOMAINS`, with no IP / shortener / user-content host /
     high-risk TLD / look-alike, and Reply-To in the sender's org. Any auth failure or look-alike sender blocks it.
   - Threat units (`api/xai.py: override_units`, the one list): look-alike domain, auth failure, high-risk TLD, IP link,
     shortener, archive/HTML link, program-file link (2 units), BEC lure (2 units), urgency (only next to another unit).
     A plain `http://` link is reported in `suspicious_urls` but not counted.
   - `/predict` adds `model_score`, `adjustment` (`trusted_sender` | `signal_override`) and `adjustment_reason`.
4. **Rate limiter (`api/rate_limiter.py`)**
   - Sliding window, 20 requests/minute per client, on `/predict`, `/explain` and `/admin/recalibrate`.
   - Memory is bounded (LRU cap plus expiry sweep).
5. **Shared model code (`ml/distilbert_model.py`, `ml/features/*`)**
   - `PhishingClassifier` is defined once. Training and serving import it. Module names are part of the state dict.
   - `text_builder.py` builds the model input. Training and serving must use it. `build_model_text_with_spans`
     also returns where each field sits in that text (used by `/explain`). `build_model_text` wraps it.
6. **Chrome Extension (`extension/`)**
   - Manifest V3. Scans Gmail and active pages. Reads `classification`, `risk_score`, `signals` and `processing_time_ms` from `/predict`. Keep these fields stable.
   - "Why this verdict?" calls `/explain`. `content.js` handles `highlightTokens` / `clearHighlights` in the open
     Gmail / Outlook body (DOM APIs only, no `innerHTML`).

---

## ⚡ Essential Commands

```bash
# Environment (Python 3.10+; see requirements.txt for tested versions)
pip install -r requirements.txt

# Start the API
python -m uvicorn api.main:app --reload --port 8000

# Tests
pytest tests/                                   # unit + API tests, no weights needed
python tests/stress_test_tier1.py               # 46 verdicts (core / legit brand mail / phishing) + latency; needs weights; exit 1 on failure
python tests/xai_sanity.py                      # /explain deletion, randomisation, completeness, latency; needs weights
```

### Training and calibration

```bash
# 1. Build unified.csv from data/raw/ (URL-only rows are kept by default)
python ml/data_pipeline.py [--raw-dir ...] [--output ...] [--exclude-url-only]

# 2. Train DistilBERT (GPU recommended). bf16 is chosen automatically on Ampere+ GPUs (RTX 3060).
#    Out of memory? use --batch_size 16 --grad_accum 2 (same effective batch of 32).
#    --output_dir writes elsewhere (e.g. ml/models/_smoke for smoke runs). --max_train_rows is for smoke runs only.
python ml/train_distilbert.py --epochs 3 --batch_size 32 --grad_accum 1 --lr 2e-5 --max_length 256 --seed 42 --precision bf16 --patience 1

# 3. Calibrate thresholds on VALIDATION probabilities of the served model
python ml/calibrate_threshold.py                # DistilBERT → threshold_config.json
python ml/calibrate_threshold.py --model rf     # RF fallback → rf_threshold_config.json

# RF fallback training (separate pipeline)
python ml/train.py --data data/processed/unified.csv
```

Notebooks: `notebooks/NETRA_DistilBERT_Training.ipynb` (Colab) and
`notebooks/NETRA_DistilBERT_Training_Local.ipynb` run the same `train_distilbert.py`.

---

## 📦 Model Weights & Release Rules

> **CRITICAL**: Never commit `.pt`, `.bin` or other model binaries to Git. They exceed GitHub's
> 100 MB limit. `.gitignore` covers them.

- **Tier-1 DistilBERT**: `ml/models/distilbert_tier1.pt` (~266 MB). Release `v2.0.0` is the shipped
  checkpoint. Its digest is recorded in `ml/models/distilbert_config.json` (`weights_sha256`).
- **Config keys that control inference** (`ml/models/distilbert_config.json`):
  - `header_features_trained` (default `false`): when false, the header vector is zeros (legacy checkpoints).
  - `header_auth_learned` (default `true`): `false` for v2.0.0, whose training data had no auth results.
    The header vector is still fed to the model, but `/explain` does not report header attributions.
  - `text_format` (default `body_only`): `body_only` sends the raw body text. `v3` sends
    `build_model_text(subject, body, urls)`. Retrained checkpoints from this pipeline set `v3` and
    `header_features_trained: true`.
- **Publishing new weights**: `gh release create <tag> <file> --title <title> --notes <notes>`, then
  update `DISTILBERT_RELEASE_URL` in `api/main.py` and `weights_sha256` in `distilbert_config.json`.

---

## 📁 Repository Map

```text
NETRA/
├── CLAUDE.md                        # This guide
├── README.md                        # Public docs, sample verdicts, API reference
├── BUGFIX_PLAN.md                   # Bug-fix plan (from bugs/ reports)
├── requirements.txt                 # Python dependencies (tested versions)
├── api/
│   ├── main.py                      # FastAPI app: /health, /predict, /explain, /admin/recalibrate
│   ├── xai.py                       # /explain fusion: threat factors, Shapley override split, schemas, summary
│   ├── trust.py                     # Trusted-sender / link-alignment rule and curated domain lists
│   └── rate_limiter.py              # Sliding-window limiter with bounded memory
├── ml/
│   ├── data_pipeline.py             # Raw corpora → unified.csv (schema, labels, splits)
│   ├── train_distilbert.py          # Tier-1 DistilBERT training (shared with notebooks)
│   ├── distilbert_model.py          # PhishingClassifier (shared by training and API)
│   ├── explain.py                   # Integrated Gradients for PhishingClassifier (/explain)
│   ├── calibrate_threshold.py       # Threshold sweep on val probabilities
│   ├── train.py                     # RF fallback training (LR / calibrated SVM / RF)
│   ├── features/
│   │   ├── text_builder.py          # build_model_text / extract_urls (train/serve parity)
│   │   ├── header_features.py       # SPF/DKIM/DMARC features, normalize_auth, extract_from_row
│   │   ├── text_features.py         # TF-IDF + handcrafted text features (RF)
│   │   └── url_features.py          # URL features, typosquatting (registrable-domain aware)
│   └── models/
│       ├── distilbert_config.json   # Inference config and training metadata (committed)
│       ├── distilbert_tokenizer/    # Tokenizer files (committed)
│       ├── threshold_config.default.json  # Committed default thresholds
│       └── threshold_config.json    # Live thresholds (gitignored; written by calibration / admin)
├── extension/                       # Manifest V3 Chrome extension
├── notebooks/                       # Colab + local DistilBERT training, IPDS (RF) training
├── tests/
│   ├── conftest.py                  # Adds project root to sys.path
│   ├── test_api.py                  # API behaviour (health, validation, limits, auth, decisions)
│   ├── test_features.py             # Features, pipeline parsing, splits, sweep, limiter
│   ├── test_explain.py              # IG engine (tiny random model) + fusion algebra
│   ├── test_trust.py                # Trust rule, new signals, windowed scoring, threshold band guard
│   ├── stress_test_tier1.py         # End-to-end verdict + latency script (not collected by pytest)
│   └── xai_sanity.py                # /explain faithfulness + latency script (not collected by pytest)
└── docs/                            # Decisions and design notes
```

---

## 🔐 Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `NETRA_ADMIN_TOKEN` | unset | Bearer token for `/admin/recalibrate`. Unset means the endpoint returns 404. |
| `NETRA_TRUSTED_PROXY_HOPS` | `0` | Proxies in front of the API. Cloud Run: `1`. `0` ignores `X-Forwarded-For`. |
| `NETRA_MAX_CONCURRENCY` | `2` | Concurrent model inferences (`/predict`). `/explain` has its own single slot. |
| `NETRA_XAI_STEPS` | `32` | Integrated Gradients steps for `/explain`, clamped to 8..64. |

---

## 📐 Coding Conventions & Guidelines

1. **Python & framework**
   - Target Python 3.10+.
   - Pydantic v2 for all request and response models.
   - `async def` for endpoints that do I/O. CPU-bound model calls go through `asyncio.to_thread`.
2. **Device independence**
   - Resolve the device in this order: CUDA, then MPS, then CPU (`ml/train_distilbert.py: pick_device`).
   - Do not hard-code `cuda:0`.
3. **Single source of truth**
   - Model input text: `ml/features/text_builder.py`. Do not build prompts elsewhere.
   - Header features: `ml/features/header_features.py`. Do not hand-roll the 10 names.
   - Architecture: `ml/distilbert_model.py`. Do not redefine `PhishingClassifier`. State-dict names must not change.
   - Auth-result normalisation: `normalize_auth` (`softfail → fail`).
4. **Thresholds**
   - Live values come from `ml/models/threshold_config.json`, with a fallback to `threshold_config.default.json`.
   - Defaults: `score < 0.08` LEGITIMATE, `0.08 <= score < 0.25` SUSPICIOUS, `score >= 0.25` PHISHING.
   - The SUSPICIOUS band must be at least `MIN_SUSPICIOUS_BAND` (0.03) wide (API and calibration).
     Calibration does not write when the F1 curve is flat over more than 0.5 (saturated scores) unless `--force`.
   - Calibration writes only its own keys (read-merge-write). Admin updates persist before in-memory state changes.
   - `threshold_config.json` is machine-local (gitignored); a fresh checkout uses the committed defaults.
5. **Proxy-aware networking**
   - Client IP comes from `api/main.py: client_ip`. It reads `X-Forwarded-For` only when
     `NETRA_TRUSTED_PROXY_HOPS > 0`, and then takes the entry that many hops from the right.
   - Keep this pattern if you change the rate limiter.
6. **Reproducibility**
   - Training seeds `random`, `numpy` and `torch` (`--seed`).
   - Checkpoint selection uses validation data only. Reported test metrics use `test_std` and `test_cross`.
   - URL-only records (PhishTank / OpenPhish) have an empty body and are excluded from email test metrics.
7. **Labels**
   - `label = 1` means malicious. SpamAssassin spam is label 1 with `label_origin = spam`, so it is
     distinguishable from phishing.
