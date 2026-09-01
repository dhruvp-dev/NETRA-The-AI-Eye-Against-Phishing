# NETRA — The AI Eye Against Phishing 🛡️

> **An intelligent, multi-tier phishing email detection system.**  
> Phase 1: Calibrated Random Forest · Phase 2: DistilBERT Transformer

---

## 📖 What is NETRA?

NETRA is a local-first AI system that classifies incoming emails as:
- 🟢 **LEGITIMATE** — safe to read
- 🟡 **SUSPICIOUS** — treat with caution
- 🔴 **PHISHING** — block immediately

It runs as a REST API on your laptop and can be integrated into email clients or browser extensions.

---

## 🏗️ Architecture

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

## 📁 Project Structure

```
NETRA/
├── api/
│   └── main.py                  <- FastAPI server (serves /predict, /health, /admin/recalibrate)
├── ml/
│   ├── features/
│   │   ├── url_features.py      <- URL signal extraction + typosquatting detection
│   │   ├── header_features.py   <- SPF/DKIM/DMARC feature extraction
│   │   └── text_features.py     <- TF-IDF text feature builder
│   ├── models/                  <- Trained model artifacts (download separately)
│   │   ├── tier1_model_calibrated.pkl
│   │   ├── tfidf_vectorizer.pkl
│   │   └── threshold_config.json
│   ├── data_pipeline.py         <- Parses raw datasets -> unified.csv
│   ├── train.py                 <- Phase 1 Random Forest training
│   ├── train_distilbert.py      <- Phase 2 DistilBERT fine-tuning
│   └── calibrate_threshold.py   <- PR-curve threshold optimizer
├── notebooks/
│   ├── NETRA_IPDS_Training.ipynb           <- Phase 1 Colab training notebook
│   └── NETRA_DistilBERT_Training.ipynb     <- Phase 2 Colab training notebook
├── data/
│   ├── raw/           <- Raw dataset files (download separately, not committed)
│   └── processed/     <- Generated: unified.csv (not committed)
└── requirements.txt
```

---

## ⚡ Quick Start — Running the API Locally

### Prerequisites
- Python 3.10+
- ~500 MB free RAM
- Pre-trained model files (`.pkl` or `.pt`)

### 1. Clone the Repository
```bash
git clone https://github.com/ramanan-2735/NETRA-The-AI-Eye-Against-Phishing.git
cd NETRA-The-AI-Eye-Against-Phishing
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Model Files
Download the following and place them in `ml/models/`:

| File | Description |
|------|-------------|
| `tier1_model_calibrated.pkl` | Calibrated Random Forest model (~64 MB) |
| `tfidf_vectorizer.pkl` | TF-IDF text vectorizer |
| `threshold_config.json` | Classification thresholds (already in repo) |

*(Alternatively, train the model locally using your local GPU as shown below!)*

### 4. Start the API Server
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### 5. Test It
```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"body_text\": \"URGENT: Your account is suspended. Verify immediately.\", \"urls\": [\"http://paypa1-secure.example.tk/login\"], \"sender\": \"security@paypa1-alert.com\", \"headers\": {\"spf\": \"fail\", \"dkim\": \"fail\", \"dmarc\": \"fail\"}}"
```

---

## 💻 Local GPU Training (NVIDIA RTX 3050 / GTX / RTX GPUs)

If you have a dedicated NVIDIA GPU (like an RTX 3050 with 4GB/6GB VRAM), you can train DistilBERT **directly on your laptop** without using Google Colab!

### 1. Install PyTorch with CUDA Support
Ensure PyTorch detects your NVIDIA RTX 3050 GPU:
```bash
# Install PyTorch with CUDA 12.1 acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Verify GPU availability in Python:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```
*(Should print: `GPU Name: NVIDIA GeForce RTX 3050 ...`)*

### 2. Download Raw Datasets
Place raw datasets in `data/raw/`:
- **SpamAssassin**: `easy_ham/`, `hard_ham/`, `spam/`, `spam_2/`
- **Nazario phishing**: `phishing3.mbox`
- **PhishTank**: `phishtank_online_valid.csv`
- **OpenPhish**: `openphish_feed.txt`
- **Enron (Optional)**: `emails.csv`

### 3. Run Data Pipeline
Parse raw files into `data/processed/unified.csv`:
```bash
python ml/data_pipeline.py
```

### 4. Train DistilBERT Locally
Run the DistilBERT training script directly:
```bash
python ml/train_distilbert.py
```
- **Training Time:** ~10–15 minutes on an RTX 3050 GPU.
- **Output:** Saves `distilbert_tier1.pt`, `distilbert_tokenizer/`, and `distilbert_config.json` directly into `ml/models/`.

### 5. Start Server with Local DistilBERT
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```
FastAPI will auto-detect the freshly trained local DistilBERT weights!

---

## ☁️ Google Colab GPU Training (Alternative)

If training on the cloud with T4 GPU instead:
1. Open Google Colab and upload `notebooks/NETRA_DistilBERT_Training.ipynb`
2. Change runtime: `Runtime -> Change runtime type -> T4 GPU`
3. Click `Runtime -> Run all` (Ctrl+F9)
4. Download the 3 generated files and place them in `ml/models/`.

---

## 🌡️ Live Threshold Tuning (During Demo)

Adjust classification thresholds on the fly without restarting:
```bash
curl -X POST http://127.0.0.1:8000/admin/recalibrate \
  -H "Content-Type: application/json" \
  -d "{\"suspicious_lower\": 0.08, \"suspicious_upper\": 0.25, \"phishing_threshold\": 0.25}"
```

---

## 🧪 Running Test Suite
```bash
python scratch/test_v2.py
```

---

## 📄 License
MIT License