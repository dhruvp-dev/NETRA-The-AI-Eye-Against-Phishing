# 🛡️ NETRA — The AI Eye Against Phishing

> **An Intelligent Multi-Tier Phishing Email Detection System**  
> Classifies emails in real-time as **LEGITIMATE**, **SUSPICIOUS**, or **PHISHING**.

---

## 📌 Table of Contents
1. [Overview & Architecture](#-overview--architecture)
2. [Option A: Quick Start (Run Pre-trained API)](#-option-a-quick-start-run-pre-trained-api)
3. [Option B: Train on Local GPU (RTX 3050 / GTX / RTX)](#-option-b-train-on-local-gpu-nvidia-rtx-3050--gtx--rtx)
4. [Option C: Train on Google Colab (Free T4 Cloud GPU)](#-option-c-train-on-google-colab-free-cloud-t4-gpu)
5. [Testing the API](#-testing-the-api)
6. [Live Threshold Tuning](#-live-threshold-tuning-during-demos)
7. [Project Structure](#-project-structure)

---

## 🏗️ Overview & Architecture

NETRA uses a **two-tier machine learning pipeline** to evaluate email body text, URL signals, and SPF/DKIM/DMARC authentication headers:

- 🟢 **LEGITIMATE (Score < 0.08)** — Safe email, normal business language.
- 🟡 **SUSPICIOUS (Score 0.08 – 0.25)** — Borderline signals, missing headers, or potential typosquatting.
- 🔴 **PHISHING (Score > 0.25)** — High-risk attack, fake domain, failed authentication.

```
[Incoming Email Payload]
          │
          ├──> 1. Header Auth Check (SPF / DKIM / DMARC)
          ├──> 2. URL Inspection (Typosquatting: paypa1, micros0ft, etc.)
          └──> 3. Text Intent Engine (DistilBERT / Random Forest)
          │
          ▼
   [ Prediction: LEGITIMATE / SUSPICIOUS / PHISHING ]
```

---

## ⚡ Option A: Quick Start (Run Pre-trained API)

Follow these steps to get the API running locally in **less than 3 minutes**:

### Step 1: Clone the Repository
Open your terminal or PowerShell:
```bash
git clone https://github.com/ramanan-2735/NETRA-The-AI-Eye-Against-Phishing.git
cd NETRA-The-AI-Eye-Against-Phishing
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Verify Model Artifacts
Ensure the following files exist in `ml/models/`:
- `tier1_model_calibrated.pkl`
- `tfidf_vectorizer.pkl`
- `threshold_config.json`

*(Note: These basic models are already included in the repo. For maximum accuracy, train DistilBERT using Option B or C).*

### Step 4: Start the API Server
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```
Your API is now live at `http://127.0.0.1:8000`! You can view the interactive Swagger docs at `http://127.0.0.1:8000/docs`.

---

## 💻 Option B: Train on Local GPU (NVIDIA RTX 3050 / GTX / RTX)

If you have a gaming laptop or PC with an **NVIDIA GPU (e.g., RTX 3050)**, you can train the **DistilBERT Transformer model** locally without using Google Colab.

### Step 1: Install PyTorch with CUDA Support
Make sure PyTorch is configured to use your NVIDIA graphics card:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 2: Verify Your GPU
Run this quick Python command to confirm PyTorch detects your GPU:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU Detected:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```
> **Expected Output:** `GPU Detected: NVIDIA GeForce RTX 3050 ...`

### Step 3: Run the Data Pipeline
Processes the raw datasets into `data/processed/unified.csv`:
```bash
python ml/data_pipeline.py
```

### Step 4: Run DistilBERT Fine-Tuning
Start training the transformer model:
```bash
python ml/train_distilbert.py
```
- **Training Time:** ~10 to 15 minutes on an RTX 3050.
- **What Happens Automatically:** The script trains for 3 epochs and saves `distilbert_tier1.pt`, `distilbert_tokenizer/`, and `distilbert_config.json` straight into `ml/models/`.

### Step 5: Start Server with Your New Model
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```
FastAPI will print `Active model: distilbert` on startup!

---

## ☁️ Option C: Train on Google Colab (Free Cloud T4 GPU)

If you don't have an NVIDIA GPU on your laptop, use Google Colab for free:

1. Open **[Google Colab](https://colab.research.google.com/)**.
2. Click **Upload** and select `notebooks/NETRA_DistilBERT_Training.ipynb` from your local repo.
3. Change runtime setting:
   > Go to **Runtime** ➔ **Change runtime type** ➔ Select **T4 GPU** ➔ Click **Save**.
4. Run the entire notebook:
   > Click **Runtime** ➔ **Run all** (or press `Ctrl + F9`).
5. **Download Artifacts:** Once training finishes (~25 min), 3 files will download to your browser:
   - `distilbert_tier1.pt`
   - `distilbert_tokenizer.zip` *(extract into a folder named `distilbert_tokenizer/`)*
   - `distilbert_config.json`
6. Move all 3 files into your local `ml/models/` directory and restart uvicorn.

---

## 🧪 Testing the API

### 1. Run Automated Test Suite
We included 5 graded test cases ranging from clean emails to phishing attacks:
```bash
python scratch/test_v2.py
```

### 2. Manual cURL Test (Phishing Sample)
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "body_text": "URGENT: Your account access has been restricted. Verify immediately.",
    "urls": ["http://paypa1-security-check.example.tk/login"],
    "sender": "security@paypa1-alert.com",
    "headers": {"spf": "fail", "dkim": "fail", "dmarc": "fail"}
  }'
```

---

## 🌡️ Live Threshold Tuning (During Demos)

You can adjust classification boundaries in real-time **without restarting the server**:

```bash
curl -X POST http://127.0.0.1:8000/admin/recalibrate \
  -H "Content-Type: application/json" \
  -d '{
    "suspicious_lower": 0.08,
    "suspicious_upper": 0.25,
    "phishing_threshold": 0.25
  }'
```

---

## 📁 Project Structure

```
NETRA/
├── api/
│   └── main.py                  # FastAPI inference server & endpoints
├── ml/
│   ├── features/
│   │   ├── url_features.py      # URL signals & typosquatting detection
│   │   ├── header_features.py   # SPF/DKIM/DMARC header parsing
│   │   └── text_features.py     # TF-IDF vector building
│   ├── models/                  # Saved weights (.pkl & .pt files)
│   ├── data_pipeline.py         # Raw data ingestion & dataset builder
│   ├── train.py                 # Phase 1 Random Forest training
│   ├── train_distilbert.py      # Phase 2 Local DistilBERT GPU training
│   └── calibrate_threshold.py   # Threshold optimization sweeper
├── notebooks/
│   └── NETRA_DistilBERT_Training.ipynb  # Cloud Colab training notebook
├── scratch/
│   └── test_v2.py               # Automated 5-case test runner
└── requirements.txt             # Project dependencies
```

---

## 📄 License
Distributed under the **MIT License**. Free for academic and personal security research.