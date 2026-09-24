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
7. [Browser Extension Guide](#browser-extension-guide)
8. [Project Structure](#-project-structure)

---

## 🧠 Overview & Detection Pipeline

NETRA delivers sub-200ms real-time email security by combining semantic transformer embeddings with cryptographic and lexical heuristics:

```
[ Incoming Email (Subject, Body, Sender, Headers) ]
                      │
     ┌────────────────┴────────────────────────┐
     ▼                                         ▼
[ DistilBERT Transformer ]         [ Multi-Signal Security Heuristics ]
 • 66M Parameter Base               • SPF / DKIM / DMARC Header Auth
 • Contextual Text Semantics        • Typosquatting Engine (Levenshtein)
 • Fine-Tuned on 180k+ emails       • Suspicious TLD & HTTP URL Parser
     └────────────────┬────────────────────────┘
                      ▼
       [ Multi-Signal Fusion Engine ]
                      │
   ┌──────────────────┼──────────────────┐
   ▼                  ▼                  ▼
🟢 LEGITIMATE      🟡 SUSPICIOUS      🔴 PHISHING
(Score < 0.08)    (Score 0.08–0.25)   (Score >= 0.25)
```

- **Validation F1 Score:** `0.8114`
- **False Positive Rate (FPR):** `0.0010` (Only **0.1%** false alarms!)
- **Latency:** ~180 ms per inference.

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

### Test 1: Normal Safe Email
**Expected Verdict:** `LEGITIMATE` (Score: ~0.0000, Risk: LOW)
```json
{
  "subject": "Weekly project status update and meeting minutes",
  "body_text": "Hi team, please find attached the weekly notes and roadmap review for sprint 14. Next sync will be on Friday at 10 AM.",
  "sender": "sarah.connor@cyberdyne.com"
}
```

---

### Test 2: Urgent Credential Harvest
**Expected Verdict:** `PHISHING` (Score: > 0.95, Risk: CRITICAL)
```json
{
  "subject": "URGENT: Unauthorized login detected on your PayPal account!",
  "body_text": "Dear valued user, an unknown login attempt from Russia was detected. Verify your credentials immediately at http://paypal-security-verification.tk/login or your account will be permanently closed within 24 hours.",
  "sender": "service@paypa1-security.com"
}
```

---

### Test 3: Fake Corporate Alert
**Expected Verdict:** `SUSPICIOUS` (Score: 0.08 – 0.25, Risk: MEDIUM)
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
python ml/train_distilbert.py --epochs 3 --batch_size 16 --fp16
```

---

## 📡 API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Returns active model status (`distilbert`), thresholds, and health. |
| `POST` | `/predict` | Evaluates email subject, body, sender, and headers. |
| `POST` | `/admin/recalibrate` | Live tuning of detection thresholds without server restart. |
| `GET` | `/docs` | Interactive Swagger OpenAPI UI. |

---

## Browser Extension Guide

The NETRA browser extension provides real-time phishing and typosquatting protection directly inside the browser, featuring one-click email fetching for Gmail and Outlook.

### Live Cloud Backend

The extension is configured to connect to the Google Cloud Run production API:
`https://netra-api-h47irmxbha-el.a.run.app`

### Installation

#### Google Chrome, Brave, and Microsoft Edge

1. Open your browser and navigate to: `chrome://extensions/`
2. Enable Developer mode using the toggle in the top-right corner.
3. Click the "Load unpacked" button in the top-left corner.
4. Select the `extension/` directory from this repository.
5. The NETRA extension icon will appear in your browser toolbar.

#### Mozilla Firefox

1. Open Firefox and navigate to: `about:debugging#/runtime/this-firefox`
2. In the Temporary Extensions section, click "Load Temporary Add-on...".
3. Select the `manifest.json` file inside the `extension/` directory.
4. The NETRA extension will be loaded and visible in the Firefox toolbar.

### How to Use

1. Click the NETRA extension icon in the toolbar.
2. When viewing an email in Gmail or Outlook web, click "Auto-Fetch Active Email" to extract sender, subject, and body text.
3. Alternatively, paste email content manually into the input fields or use the preset buttons for test cases.
4. Click "Analyze Email".
5. The extension displays:
   - Classification verdict: LEGITIMATE, SUSPICIOUS, or PHISHING
   - Calibrated risk score and severity level
   - Multi-signal breakdown: typosquatting detection, urgency signals, header authentication flags, and suspicious URL counts
   - Real-time inference latency (sub-200ms)

### Changing the Backend URL

To connect the extension to a local server or custom endpoint:
1. Open the extension popup.
2. In the backend configuration field, enter the desired API URL (for example: `http://127.0.0.1:8000`).
3. Click "Save".

---

## 📁 Project Structure

```text
NETRA/
├── api/
│   └── main.py                   # FastAPI application with multi-signal fusion
├── extension/                    # Cross-browser extension (Chrome & Firefox)
│   ├── manifest.json             # Manifest V3 configuration
│   ├── popup.html                # Scanner UI layout
│   ├── popup.js                  # Frontend controller & API client
│   ├── popup.css                 # Design system styles
│   ├── content.js                # DOM content extraction script
│   └── service-worker.js         # Background worker & context menus
├── ml/
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
├── Dockerfile                    # Container definition for Cloud Run
├── deploy.sh                     # Automated Cloud Run deployment script
├── requirements.txt              # Production dependencies
└── README.md                     # Documentation
```
