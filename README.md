# NETRA — The AI Eye Against Phishing

> **An intelligent, multi-tier phishing email detection system.**
> Phase 1: Calibrated Random Forest · Phase 2: DistilBERT Transformer

---

## What is NETRA?

NETRA is a local-first AI system that classifies incoming emails as:
- LEGITIMATE — safe to read
- SUSPICIOUS — treat with caution
- PHISHING — block immediately

It runs as a REST API on your laptop and can be integrated into email clients or browser extensions.

---

## Architecture

```
Phase 1 (Active):
  TF-IDF (5000 features) + URL features (10) + Header features (10)
  -> Calibrated Random Forest -> LEGITIMATE / SUSPICIOUS / PHISHING

Phase 2 (In Training):
  DistilBERT [CLS] embedding (768d) + Header features (10->32d)
  -> Classifier head -> LEGITIMATE / SUSPICIOUS / PHISHING
  Expected: Phishing Recall 94-97%, FPR < 3%
```

---

## Project Structure

```
NETRA/
+-- api/
|   +-- main.py                  <- FastAPI server (serves /predict, /health, /admin/recalibrate)
+-- ml/
|   +-- features/
|   |   +-- url_features.py      <- URL signal extraction + typosquatting detection
|   |   +-- header_features.py   <- SPF/DKIM/DMARC feature extraction
|   |   +-- text_features.py     <- TF-IDF text feature builder
|   +-- models/                  <- Trained model artifacts (download separately)
|   |   +-- tier1_model_calibrated.pkl
|   |   +-- tfidf_vectorizer.pkl
|   |   +-- threshold_config.json
|   +-- data_pipeline.py         <- Parses raw datasets -> unified.csv
|   +-- train.py                 <- Phase 1 Random Forest training
|   +-- train_distilbert.py      <- Phase 2 DistilBERT fine-tuning
|   +-- calibrate_threshold.py   <- PR-curve threshold optimizer
+-- notebooks/
|   +-- NETRA_IPDS_Training.ipynb           <- Phase 1 Colab training notebook
|   +-- NETRA_DistilBERT_Training.ipynb     <- Phase 2 Colab training notebook (MAIN)
+-- data/
|   +-- raw/           <- Raw dataset files (download separately, not committed)
|   +-- processed/     <- Generated: unified.csv (not committed)
+-- requirements.txt
```

---

## Quick Start — Running the API Locally

### Prerequisites
- Python 3.10+
- ~500 MB free RAM
- Pre-trained model files (.pkl) — see Downloading Models below

### 1. Clone the Repository
```
git clone https://github.com/ramanan-2735/NETRA-The-AI-Eye-Against-Phishing.git
cd NETRA-The-AI-Eye-Against-Phishing
```

### 2. Install Dependencies
```
pip install -r requirements.txt
```

### 3. Download Model Files
Download the following and place them in ml/models/:

| File | Description |
|------|-------------|
| tier1_model_calibrated.pkl | Calibrated Random Forest model (~64 MB) |
| tfidf_vectorizer.pkl | TF-IDF text vectorizer |
| threshold_config.json | Classification thresholds (already in repo) |

Contact the project owner or train them yourself (see Training section below).

### 4. Start the API Server
```
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### 5. Test It
```
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"body_text\": \"URGENT: Your account is suspended. Verify immediately.\", \"urls\": [\"http://paypa1-secure.example.tk/login\"], \"sender\": \"security@paypa1-alert.com\", \"headers\": {\"spf\": \"fail\", \"dkim\": \"fail\", \"dmarc\": \"fail\"}}"
```

### API Response Format
```json
{
  "classification": "SUSPICIOUS",
  "risk_score": 0.1731,
  "confidence": 0.0931,
  "risk_level": "MEDIUM",
  "model_type": "calibrated_rf",
  "signals": {
    "header_auth_failed": true,
    "urgency_detected": true,
    "suspicious_urls": 2,
    "typosquatting_detected": true
  },
  "processing_time_ms": 343.0
}
```

---

## Training Phase 2 — DistilBERT (For Collaborators)

> Trains a transformer model that replaces Random Forest.
> Expected: Phishing Recall 94-97%, FPR < 3%
> Time: ~25-30 minutes on Google Colab T4 GPU

### Step 1: Open the Notebook in Colab
Go to https://colab.research.google.com and upload:
notebooks/NETRA_DistilBERT_Training.ipynb

### Step 2: Enable T4 GPU
In Colab menu: Runtime -> Change runtime type -> T4 GPU -> Save

### Step 3: Run All Cells
Click Runtime -> Run all (or Ctrl+F9)

The notebook is FULLY SELF-CONTAINED. It will automatically:
1. Clone this repo to your Google Drive
2. Download all required datasets (SpamAssassin, Nazario, PhishTank, OpenPhish)
3. Run the data pipeline to build unified.csv
4. Fine-tune DistilBERT for 3 epochs
5. Generate evaluation plots
6. Download the 3 model artifacts to your browser

### Step 4: Add Enron Dataset (Optional, Recommended for Best Accuracy)
1. Create a Kaggle account at https://www.kaggle.com
2. Go to Profile -> Account -> API -> Create New Token -> download kaggle.json
3. Upload kaggle.json to your Colab session
4. Uncomment the Enron cell in the notebook and run it

### Step 5: Deploy Locally
After training, 3 files download automatically:
- distilbert_tier1.pt (~250 MB)
- distilbert_tokenizer.zip -> unzip into distilbert_tokenizer/ folder
- distilbert_config.json

Place all 3 in your local ml/models/ folder and restart the API:
```
uvicorn api.main:app --host 127.0.0.1 --port 8000
```
The server auto-detects DistilBERT and switches to it!

---

## Training Phase 1 — Random Forest (From Scratch)

### 1. Download Raw Datasets

| Dataset | Source | Place in |
|---------|--------|----------|
| SpamAssassin | https://spamassassin.apache.org/old/publiccorpus/ | data/raw/easy_ham/, data/raw/hard_ham/, data/raw/spam/, data/raw/spam_2/ |
| Enron emails | https://www.kaggle.com/datasets/wcukierski/enron-email-dataset | data/raw/emails.csv |
| Nazario phishing | https://monkey.org/~jose/phishing/phishing3.mbox | data/raw/phishing3.mbox |
| PhishTank | https://www.phishtank.com/developer_info.php | data/raw/phishtank_online_valid.csv |
| OpenPhish | https://openphish.com/feed.txt | data/raw/openphish_feed.txt |

### 2. Run Data Pipeline
```
python ml/data_pipeline.py
```

### 3. Train (Google Colab Recommended)
Upload notebooks/NETRA_IPDS_Training.ipynb to Colab and run all cells.

---

## Live Threshold Tuning (During Demo)

Adjust classification thresholds without restarting the server:
```
curl -X POST http://127.0.0.1:8000/admin/recalibrate \
  -H "Content-Type: application/json" \
  -d "{\"suspicious_lower\": 0.08, \"suspicious_upper\": 0.25, \"phishing_threshold\": 0.25}"
```

---

## Current Performance (Phase 1 - Calibrated RF)

| Metric | Value |
|--------|-------|
| Model | Calibrated RandomForest (500 trees) |
| Thresholds | suspicious >= 0.08, phishing >= 0.25 |
| Legitimate score range | 0.005 - 0.066 |
| Phishing score range | 0.173 - 0.296 |
| Typosquatting detection | YES (paypa1, micros0ft, g00gle, etc.) |
| Header auth signals | YES (SPF / DKIM / DMARC) |

---

## Running Tests
```
python scratch/test_v2.py
```

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | Complete | TF-IDF + Calibrated RF + Threshold tuning |
| Phase 2 | In Training | DistilBERT Tier-1 with header fusion |
| Phase 3 | Planned | Tier-2 URL deep inspection |
| Phase 4 | Planned | Feedback loop + active learning |
| Phase 5 | Planned | Explainability dashboard |
| Phase 6 | Planned | Browser extension integration |

---

## License

MIT License