# 🛡️ NETRA — The AI Eye Against Phishing

> **State-of-the-Art Multi-Signal Deep Learning Phishing Detection Engine**  
> Powered by Fine-Tuned **DistilBERT** (66M Parameters), RFC Header Authentication Analysis, and Inline Domain Typosquatting Defense.

---

## 📋 Table of Contents
1. [Overview & Detection Pipeline](#-overview--detection-pipeline)
2. [Quick Start — Run the API in 3 Minutes](#-quick-start--run-the-api-in-3-minutes)
3. [Interactive Web UI Testing (/docs)](#-interactive-web-ui-testing-docs)
4. [Sample Test Cases (Copy & Paste Ready)](#-sample-test-cases)
   - [Test 1: Normal Safe Email (LEGITIMATE)](#test-1-normal-safe-email)
   - [Test 2: Urgent Credential Harvest (PHISHING)](#test-2-urgent-credential-harvest)
   - [Test 3: Fake Corporate Alert (SUSPICIOUS)](#test-3-fake-corporate-alert)
5. [How to Train the Model (Google Colab & Local GPU)](#-how-to-train-the-model)
6. [API Endpoints Reference](#-api-endpoints-reference)
7. [Project Structure](#-project-structure)

---

## 🧠 Overview & Detection Pipeline

NETRA combines a fine-tuned DistilBERT text classifier with a rule layer that checks who sent the email and where its links go:

```
[ Incoming Email (Subject, Body, Sender, URLs, Headers) ]
                      │
                      ▼
[ DistilBERT Tier-1 ]  long emails: head + up to 4 body windows, highest score wins
                      │  model score
                      ▼
[ Score rule ]  < 0.08 LEGITIMATE · 0.08–0.25 SUSPICIOUS · >= 0.25 PHISHING
                      │
                      ▼
[ Rule layer ]
 • Trusted sender: known brand domain or SPF/DKIM/DMARC pass, AND every link on the
   sender's own / known domains, no look-alike, shortener, IP or BEC lure
     → known brand: LEGITIMATE · authenticated unknown domain: PHISHING → SUSPICIOUS
 • Threat signals: look-alike domain, failed auth, high-risk TLD, IP link, shortener,
   program-file link (2), payment/credential request under pressure (2), urgency
     → LEGITIMATE + 2 signals: SUSPICIOUS · + 3 signals: PHISHING
                      │
                      ▼
🟢 LEGITIMATE      🟡 SUSPICIOUS      🔴 PHISHING
```

`/predict` returns `model_score` (the model alone) next to `risk_score` (final), and `adjustment` names the rule that changed the verdict (`trusted_sender` or `signal_override`).

**Current status (audit of 2026-10-17, see `bugs/tier1_model_bugs.md`):**
- `tests/stress_test_tier1.py`: 46/46 verdicts correct (11 core, 20 realistic legitimate brand / personal emails, 15 phishing incl. BEC, short-link invoice and padded phishing). p95 latency about 0.5 s on CPU for short emails; long emails take 1–2 s.
- The model alone flags most legitimate brand mail as phishing (for example Amazon "shipped" at 0.999). The trusted-sender rule corrects these verdicts; the model is not retrained yet. See [Known limitations](#-known-limitations).

---

## ⚡ Quick Start — Run the API in 3 Minutes

Follow these simple steps to run the server on your computer:

### Step 1: Clone Repository
```bash
git clone https://github.com/ramanan-2735/NETRA-The-AI-Eye-Against-Phishing.git
cd NETRA-The-AI-Eye-Against-Phishing
```

### Step 2: Install Requirements
Ensure you have Python 3.10+ installed:
```bash
pip install -r requirements.txt
```

### Step 3: Start the Backend Server
```bash
python -m uvicorn api.main:app --reload --port 8000
```
*(On Windows with Python launcher, you can also use `py -3.14 -m uvicorn api.main:app --reload --port 8000`)*

The server will initialize:
```text
============================================================
NETRA — AI-Powered Phishing Detection Engine starting up...
============================================================
Thresholds loaded: phishing>=0.2500 | suspicious [0.0800, 0.2500)
DistilBERT model loaded successfully.
Active model: distilbert | Status: Ready
Uvicorn running on http://127.0.0.1:8000
```

---

## 🌐 Interactive Web UI Testing (/docs)

NETRA provides a complete, interactive Swagger interface:

1. Open your browser to: **[http://localhost:8000/docs](http://localhost:8000/docs)**
2. Click **`POST /predict`** ➔ Click **Try it out**.
3. Paste any sample email from the section below and click **Execute**!

---

## 🧪 Sample Test Cases

You can test these directly in Swagger UI or via cURL / PowerShell:

### Test 0: Legitimate brand email (model over-flags it, the sender check corrects it)
**Verdict:** `LEGITIMATE` (`model_score` ≈ 0.999 → `risk_score` 0.04, `adjustment`: `trusted_sender`)
```json
{
  "subject": "Your Amazon.com order #112-4567890 has shipped",
  "body_text": "Hello, your package is on its way and will arrive Thursday. Track your package or view your order in Your Orders.",
  "sender": "shipment-tracking@amazon.com",
  "urls": ["https://www.amazon.com/gp/your-account/order-details?orderID=112-4567890"]
}
```
Change one URL to `https://amazon-order-help.com/track` and the sender check no longer applies: the verdict goes back to the model's `PHISHING`.

---

### Test 1: Normal Safe Email
**Verdict** (measured with the shipped `distilbert_tier1.pt`): `LEGITIMATE` (Score: 0.0029, Risk: LOW)
```json
{
  "subject": "Weekly project status update and meeting minutes",
  "body_text": "Hi team, please find attached the weekly notes and roadmap review for sprint 14. Next sync will be on Friday at 10 AM.",
  "sender": "sarah.connor@cyberdyne.com"
}
```

---

### Test 2: Urgent Credential Harvest
**Verdict** (measured with the shipped `distilbert_tier1.pt`): `PHISHING` (Score: 0.9977, Risk: CRITICAL)
```json
{
  "subject": "URGENT: Unauthorized login detected on your PayPal account!",
  "body_text": "Dear valued user, an unknown login attempt from Russia was detected. Verify your credentials immediately at http://paypal-security-verification.tk/login or your account will be permanently closed within 24 hours.",
  "sender": "service@paypa1-security.com"
}
```

---

### Test 3: Fake Corporate Alert
**Verdict** (measured with the shipped `distilbert_tier1.pt`): `PHISHING` (Score: 0.9761, Risk: CRITICAL).
> The shipped weights over-flag this email. The earlier expectation (SUSPICIOUS) was not what the model returns. Re-check this case after retraining.
```json
{
  "subject": "Password expiry notification for user",
  "body_text": "Your Microsoft Office 365 password expires today. Click here to retain your current password: http://login-microsoftonline.ml/auth",
  "sender": "admin@micros0ft-support.net",
  "headers": {
    "spf": "fail",
    "dkim": "fail",
    "dmarc": "fail"
  }
}
```

---

## 🖥️ Testing via Command Line (cURL)

In PowerShell or Linux/macOS terminal:

```powershell
curl.exe -X POST "http://localhost:8000/predict" `
  -H "Content-Type: application/json" `
  -d '{
    "subject": "URGENT: Your PayPal Account has been suspended!",
    "body_text": "Please verify your account immediately at http://paypal-security-update.tk",
    "sender": "service@paypa1-security.com"
  }'
```

---

## 🏋️ How to Train the Model

### Option A: Free Google Colab (Recommended)
1. Go to [colab.research.google.com](https://colab.research.google.com).
2. Click **GitHub tab** ➔ Paste: `https://github.com/ramanan-2735/NETRA-The-AI-Eye-Against-Phishing`
3. Select `notebooks/NETRA_DistilBERT_Training.ipynb`.
4. Select **Runtime ➔ Change runtime type ➔ T4 GPU**.
5. Click **Runtime ➔ Run all**.
6. The notebook will fine-tune DistilBERT in ~25 minutes and automatically download `distilbert_tier1.pt` and `distilbert_tokenizer.zip`.

### Option B: Local NVIDIA GPU (RTX 3050 / RTX 40-series)
If you have a dedicated NVIDIA GPU:
```bash
python ml/train_distilbert.py --epochs 3 --batch_size 32 --precision bf16 --seed 42
# Out of memory? --batch_size 16 --grad_accum 2
# Or run the step-by-step notebook: notebooks/NETRA_DistilBERT_Training_Local.ipynb (includes checks, backup and the stress gate)
```

---

## 📡 API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Active model status and thresholds. Returns `503` when no model is loaded. |
| `POST` | `/predict` | Evaluates email subject, body, sender, and headers. Limited to 20 requests/minute per client. |
| `POST` | `/explain` | Same body as `/predict` (plus optional `top_k`, 1-25). Returns the `/predict` fields and an `explanation`: words that raised and lowered the risk, header attributions, a threat-factor breakdown that adds up to the final score, and a one-line summary. Runs Integrated Gradients, so it takes seconds on CPU. Limited to 5 requests/minute per client and one explanation at a time, so it never delays `/predict`. |
| `POST` | `/admin/recalibrate` | Live threshold update. Requires `Authorization: Bearer <NETRA_ADMIN_TOKEN>`. Returns `404` when the token is not set. |
| `GET` | `/docs` | Interactive Swagger OpenAPI UI. |

**Environment variables**

| Variable | Default | Purpose |
|---|---|---|
| `NETRA_ADMIN_TOKEN` | unset (admin endpoint disabled) | Bearer token for `/admin/recalibrate`. |
| `NETRA_TRUSTED_PROXY_HOPS` | `0` | Reverse proxies in front of the API. Cloud Run: `1`. `0` ignores `X-Forwarded-For`. |
| `NETRA_MAX_CONCURRENCY` | `2` | Maximum concurrent model inferences. |
| `NETRA_XAI_STEPS` | `32` | Integrated Gradients steps for `/explain` (8-64). Fewer is faster and less exact. |

**How `/explain` works**

- **Model part:** Integrated Gradients on the frozen DistilBERT, over the word embeddings and the 10 header features together. The target is the phishing log-odds. Every content word is compared against a `[SEP]` token, which this checkpoint scores as neutral. Headers are compared against "no headers provided". The attributions add up to the change in the model's score (`completeness_delta` is the error).
- **Rule part:** when the multi-signal rule raises the verdict, the uplift is split equally across the signals it counted. When the trusted-sender rule lowers it, the reduction is one negative `trusted_sender` component under sender authentication.
- **Combined:** both parts are reported per threat factor (wording, links, sender authentication, Reply-To), in probability points that add up to `risk_score`.
- **Extension:** click **Why this verdict?** after a scan, then **Highlight in page** to mark the words in the open Gmail or Outlook email.

**Running the tests**

```bash
pytest tests/                                  # unit and API tests (no weights needed)
python tests/stress_test_tier1.py              # end-to-end verdicts and latency (needs weights)
python tests/xai_sanity.py                     # /explain faithfulness, completeness and latency (needs weights)
```

---

## ⚠️ Known limitations

The shipped checkpoint (`v2.0.0`) has known data problems. They are documented in `bugs/tier1_model_bugs.md`. The serving layer works around them, but only retraining fixes them.

| Problem | Effect | What serving does now | Real fix |
|---|---|---|---|
| Every legitimate training email is Enron; 82% of phishing rows are URL-only | Brand, transactional and newsletter mail with links scores ~1.0 | Trusted-sender rule (`api/trust.py`, curated domain list) | Retrain with real legitimate automated mail |
| Training data has no SPF/DKIM/DMARC results | The header branch learned nothing (`header_auth_learned: false`) | Auth is used by the rules only | Training data with real `Authentication-Results` |
| Saturated scores (near 0 or 1) | `risk_score` behaves like a class flag | Thresholds reset to 0.08 / 0.25; calibration refuses a flat F1 curve | Temperature scaling on a mixed validation set |
| Only the first 256 tokens were read | Phishing after a long benign preamble was missed | Windowed scoring (head + up to 4 body windows) | Same, or a longer-context model |
| BEC and short-link lures are rare in training | No-link wire-transfer requests scored 0 | BEC and shortener signals | Add BEC / invoice-lure data |

The benchmark F1 (0.99+) in `distilbert_config.json` measures held-out splits from the same corpora, not real-world mail.
`ml/models/threshold_config.json` is machine-local (gitignored). Without it the API uses `threshold_config.default.json` (0.08 / 0.25).

---

## 📁 Project Structure

```text
NETRA/
├── api/
│   ├── main.py                   # FastAPI application with multi-signal fusion
│   └── xai.py                    # /explain: factor breakdown, schemas, summary
├── ml/
│   ├── explain.py                # Integrated Gradients for the Tier-1 model
│   ├── features/
│   │   ├── header_features.py    # SPF, DKIM, DMARC parsing
│   │   ├── text_features.py      # Urgency keywords & lexical statistics
│   │   └── url_features.py       # Levenshtein typosquatting & TLD engine
│   ├── models/
│   │   ├── distilbert_config.json # Model metadata, F1 metrics & thresholds
│   │   ├── distilbert_tokenizer/  # Production tokenizer vocabulary
│   │   └── distilbert_tier1.pt    # PyTorch trained weights checkpoint
│   ├── data_pipeline.py          # Unified data processor
│   └── train_distilbert.py       # Local GPU training script
├── notebooks/
│   └── NETRA_DistilBERT_Training.ipynb # One-click Google Colab notebook
├── requirements.txt              # Production dependencies
└── README.md                     # Documentation
```
