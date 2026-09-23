# NETRA — Tier-2 Implementation Plan
## RoBERTa-base Escalation Model

**Project:** NETRA — The AI Eye Against Phishing  
**Document:** Tier-2 Model Implementation Plan  
**Version:** 1.0  
**Status:** Planning — Not Yet Implemented  
**Depends On:** Tier-1 DistilBERT (distilbert_tier1.pt) — LOCKED & DEPLOYED

---

## 1. Context — What Tier-1 Already Does

Before designing Tier-2, understand exactly what Tier-1 produces:

```
Input Text (256 tokens)
    ↓
DistilBERT backbone (6 layers, 768 hidden dims, 66.36M params)
    ↓
[CLS] token → 768-dim representation
    +
10-dim RFC header features → Linear(10→32) + ReLU → 32-dim projection
    ↓
Concatenation → 800-dim unified vector
    ↓
Linear(800→256) → ReLU → Dropout(0.3) → Linear(256→2)
    ↓
Softmax → risk_score (phishing probability)
    ↓
Decision corridor:
    risk_score >= 0.25         → PHISHING
    0.08 <= risk_score < 0.25  → SUSPICIOUS
    risk_score < 0.08          → LEGITIMATE
    +
Multi-signal override (typosquatting + header fail + phish URLs)
```

**Tier-1 validated metrics:**
- Accuracy: 99.72%
- Precision: 86.26%
- Phishing Recall: 76.59% ← this is the weak point
- F1: 81.14%
- FPR: 0.10%
- PR-AUC: 85.88%
- Inference: ~180ms

**Key observation:** Recall at 76.59% means roughly 1 in 4 phishing emails
is not being caught by Tier-1. Tier-2 exists to catch these hard cases
without sending every email through a heavier model.

---

## 2. Tier-2 Goal

Tier-2 must:

1. Handle cases Tier-1 escalates — low confidence, borderline scores, conflicting signals
2. Improve phishing recall on the hard cases without degrading FPR
3. Provide token-level XAI (Integrated Gradients) — currently missing from Tier-1
4. Run as a remote service (Google Cloud Run) — not locally

Tier-2 must NOT:
- Process every email by default
- Replace Tier-1
- Be evaluated only on accuracy

---

## 3. Selected Model — RoBERTa-base

**Model:** `roberta-base` (Hugging Face)  
**Parameters:** ~125M  
**Hidden dimensions:** 768  
**Layers:** 12 (vs DistilBERT's 6)  
**Max sequence length:** 512 tokens (vs DistilBERT's 256)

**Why RoBERTa over alternatives:**

| Model | Reason for/against |
|---|---|
| RoBERTa-base | Stronger pretraining (10x more data, dynamic masking), same tokenizer family, validated in literature (Paper 12 in NETRA lit review), natural step up from DistilBERT |
| BERT-base | RoBERTa is strictly better trained — no reason to pick BERT |
| DeBERTa-v3-small | Better benchmarks but more complex, less community support, riskier for deadline |
| SecBERT/PhishBERT | Domain-specific but niche, less maintained, uncertain stability |

---

## 4. Tier-2 Architecture

```
Escalated Email Input
    ↓
Text prompt assembly:
"Subject: {subject}\nBody: {body}\nURLs: {urls}"
    ↓
RoBERTa-base tokenizer
(max_length=512, padding, truncation)
    ↓
RoBERTa-base backbone (12 layers, 768 dims)
    ↓
[CLS] token → 768-dim representation
    +
10-dim RFC header features (reuse existing header_features.py)
    +
1-dim Tier-1 confidence score (passed from escalation payload)
    ↓
Concatenation → 779-dim unified vector
    ↓
Linear(779→256) → ReLU → Dropout(0.3) → Linear(256→3)
    ↓
Softmax → 3-class output
    ↓
LEGITIMATE / SUSPICIOUS / PHISHING
    ↓
Integrated Gradients (token-level XAI)
```

**Why 3-class output:** Unlike Tier-1 which uses binary + threshold corridor,
Tier-2 trains directly on 3 classes since it only sees hard/uncertain cases.

**Why include Tier-1 confidence as input:** It tells the model why the email
was escalated. A score of 0.15 (borderline) is a different signal than 0.09
(near-legitimate). This is a meaningful feature.

---

## 5. Escalation Policy

The escalation threshold is experimentally determined — do not hardcode.

**Candidate escalation conditions:**

```
Tier-1 produces:
    ↓
    ├── risk_score in SUSPICIOUS corridor (0.08–0.25)
    │   → ALWAYS escalate
    │
    ├── risk_score >= 0.25 (PHISHING) BUT confidence < 0.70
    │   → escalate (uncertain phishing)
    │
    ├── LEGITIMATE verdict BUT 1 or more override signals fired
    │   → escalate (conflicting signals)
    │
    └── LEGITIMATE verdict, confidence >= 0.70, no signals
        → do NOT escalate (final result)
```

**What gets passed to Tier-2:**

```json
{
  "subject": "...",
  "body_text": "...",
  "urls": ["..."],
  "sender": "...",
  "reply_to": "...",
  "headers_available": {"spf": "fail", "dkim": "none", "dmarc": "fail"},
  "tier1_risk_score": 0.14,
  "tier1_confidence": 0.61,
  "tier1_verdict": "SUSPICIOUS",
  "tier1_signals": {
    "header_auth_failed": true,
    "urgency_detected": false,
    "suspicious_urls": 1,
    "typosquatting_detected": false
  }
}
```

---

## 6. Training Plan

### 6.1 Training Data Strategy

Do NOT train Tier-2 on the full dataset. Train it on the hard cases only.

**Step 1 — Generate escalation dataset:**
- Run all training records through Tier-1
- Keep only records that would have been escalated
  (SUSPICIOUS verdict OR low-confidence PHISHING OR LEGITIMATE with signals)
- This becomes the Tier-2 training set

**Step 2 — Label mapping:**
- Original label 0 (Legitimate) → class 0
- Original label 1 (Phishing) → class 2
- Escalated uncertain cases with conflicting signals → class 1 (Suspicious)

**Step 3 — Expected dataset size:**
- Roughly 20–40% of total records will be escalated
- From ~10,000–15,000 total records expect ~2,000–6,000 Tier-2 training samples
- If too small, augment with hard negatives (legitimate emails with phishing-like language)

### 6.2 Training Configuration

| Parameter | Value |
|---|---|
| Base model | `roberta-base` (Hugging Face) |
| Max sequence length | 512 tokens |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Batch size | 16 (RoBERTa is larger, reduce from Tier-1's 32) |
| Epochs | 3–5 (early stopping on val F1) |
| Loss | Weighted CrossEntropy (class imbalance expected) |
| Dropout | 0.3 |
| Gradient clipping | 1.0 |
| Warmup steps | 100 |
| Random seed | 42 |

### 6.3 Training Environment

- **Platform:** Google Colab Pro (T4 GPU) or Kaggle Notebooks (free T4)
- **Estimated training time:** 2–4 hours on T4 for 3 epochs
- **Save checkpoints to:** Google Drive after each epoch
- **Final model upload to:** Hugging Face Hub (private repo)

---

## 7. XAI — Integrated Gradients

Tier-1 has only heuristic signal explanation. Tier-2 must implement
real mathematical XAI.

**Method:** Integrated Gradients (token-level attribution)

**Library:** `captum` (PyTorch XAI library by Meta)

**What it produces:**

```json
{
  "verdict": "PHISHING",
  "risk_score": 0.87,
  "confidence": 0.91,
  "top_tokens": [
    {"token": "verify", "attribution": 0.42},
    {"token": "account", "attribution": 0.38},
    {"token": "suspended", "attribution": 0.31},
    {"token": "immediately", "attribution": 0.29},
    {"token": "paypa1.com", "attribution": 0.61}
  ],
  "signals": {
    "header_auth_failed": true,
    "urgency_detected": true,
    "suspicious_urls": 2,
    "typosquatting_detected": true
  },
  "tier1_verdict": "SUSPICIOUS",
  "tier1_risk_score": 0.14
}
```

**XAI latency:** Must be measured separately from inference latency.
Target: XAI adds no more than 300ms on top of inference.

---

## 8. Repository Structure Changes

Add these to the existing NETRA repository:

```
ml/
├── train_roberta_tier2.py        ← NEW: Tier-2 training script
├── generate_tier2_dataset.py     ← NEW: Run Tier-1 on data, extract escalated records
├── calibrate_tier2_threshold.py  ← NEW: Threshold sweep for Tier-2 escalation policy
├── models/
│   ├── roberta_tier2/            ← NEW: Saved Tier-2 model directory
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── tokenizer.json
│   │   └── tier2_metrics.json    ← NEW: Tier-2 evaluation results
│   └── tier2_config.json         ← NEW: Tier-2 thresholds and config

api/
├── main.py                       ← MODIFY: Add escalation logic + Tier-2 client call
├── tier2_client.py               ← NEW: HTTP client to call Cloud Run Tier-2 endpoint
└── tier2_service/                ← NEW: Separate FastAPI app for Tier-2 (deployed to Cloud Run)
    ├── main.py                   ← Tier-2 inference + XAI endpoint
    ├── model.py                  ← RoBERTa model class
    ├── xai.py                    ← Integrated Gradients implementation
    ├── Dockerfile                ← Container definition for Cloud Run
    └── requirements.txt          ← Tier-2 specific dependencies

notebooks/
└── NETRA_RoBERTa_Tier2_Training.ipynb  ← NEW: Colab training notebook
```

---

## 9. API Changes

### 9.1 Modified Tier-1 API flow (api/main.py)

```python
# After Tier-1 inference:
if should_escalate(tier1_result):
    tier2_result = await tier2_client.predict(payload)
    return tier2_result  # Tier-2 result is final
else:
    return tier1_result  # Tier-1 result is final
```

### 9.2 New Tier-2 API endpoint (tier2_service/main.py)

```
POST /predict
Input:  EmailPayload + tier1_risk_score + tier1_confidence + tier1_verdict + tier1_signals
Output: verdict, risk_score, confidence, top_tokens (XAI), signals, processing_time_ms
```

### 9.3 Also fix during this phase (from code audit)

- Fix duplicate subject injection in `api/main.py:455-456`
- Add the missing `tier1_model.pkl` fallback or remove the dead fallback code
- Reconcile `sender_domain_matches_reply_to` vs `sender_domain_match` naming inconsistency

---

## 10. Evaluation Plan

Tier-2 must be evaluated separately from Tier-1 and then as a combined system.

### 10.1 Tier-2 standalone metrics
- Recall, Precision, F1, PR-AUC on escalated test set
- Confusion matrix across 3 classes
- Inference latency (model only)
- XAI latency (separately)
- Total latency (model + XAI)

### 10.2 Combined system metrics (Tier-1 + Tier-2)
- Overall recall on full test set
- Overall FPR on full test set
- Overall F1 on full test set
- Percentage of emails escalated to Tier-2
- Percentage handled locally by Tier-1 only
- End-to-end latency for escalated vs non-escalated emails

### 10.3 Key comparison to report

| Metric | Tier-1 Only | Tier-1 + Tier-2 |
|---|---|---|
| Phishing Recall | 76.59% | TBD |
| FPR | 0.10% | TBD |
| F1 | 81.14% | TBD |
| PR-AUC | 85.88% | TBD |
| Escalation rate | — | TBD |
| Avg latency (non-escalated) | ~180ms | ~180ms |
| Avg latency (escalated) | — | TBD |

> If Tier-2 does not improve recall meaningfully, document that result honestly.
> A null result is a valid research finding.

---

## 11. Implementation Order

Follow this exact order. Do not skip steps.

```
Step 1 — Generate Tier-2 training dataset
    Run Tier-1 on all training records
    Extract escalated subset
    Label the 3 classes
    Save to data/processed/tier2_train.csv

Step 2 — Write training script
    ml/train_roberta_tier2.py
    RoBERTa-base + header fusion + Tier-1 confidence input
    3-class output head

Step 3 — Train on Colab/Kaggle
    Upload notebook
    Train 3–5 epochs
    Save best checkpoint (by val F1)
    Upload to Hugging Face Hub (private)

Step 4 — Calibrate escalation threshold
    Run ml/calibrate_tier2_threshold.py
    Find optimal escalation boundary
    Record in tier2_config.json

Step 5 — Implement Tier-2 service
    tier2_service/main.py (FastAPI)
    tier2_service/model.py (RoBERTa inference)
    tier2_service/xai.py (Integrated Gradients via captum)

Step 6 — Containerize
    Write Dockerfile
    Test locally with docker run
    Push to Google Artifact Registry

Step 7 — Deploy to Cloud Run
    Set memory: 2GB minimum
    Set min instances: 1 (to avoid cold start during demo)
    Test endpoint

Step 8 — Integrate into Tier-1 API
    api/tier2_client.py (async HTTP client)
    Modify api/main.py escalation logic
    Fix code audit issues from documentation

Step 9 — Evaluate combined system
    Run full evaluation pipeline
    Compare Tier-1 only vs Tier-1 + Tier-2
    Record all metrics

Step 10 — Document results in DECISIONS.md
    Final Tier-2 model decision
    Escalation threshold decision
    XAI method decision
    Combined system performance
```

---

## 12. Dependencies to Add

Add to `requirements.txt` for Tier-2 service:

```
transformers>=4.40.0
torch>=2.0.0
captum>=0.7.0
httpx>=0.27.0        # async HTTP client for Tier-2 calls from Tier-1
```

---

## 13. Known Risks

| Risk | Mitigation |
|---|---|
| Tier-2 training set too small (escalated subset only) | Augment with hard negatives; consider synthetic escalated samples |
| RoBERTa cold start on Cloud Run (~10–20s) | Set min-instances=1 during active development and demo |
| XAI latency too high for real-time use | Measure early; if >500ms, restrict token attribution to top-k only |
| Tier-2 does not improve recall meaningfully | This is a valid research result — document and explain |
| Duplicate subject injection bug affecting Tier-2 input | Fix in api/main.py before generating Tier-2 training data |

---

## 14. Success Criteria

Tier-2 implementation is complete when:

- [ ] Tier-2 training script written and tested
- [ ] RoBERTa-base fine-tuned on escalated dataset
- [ ] Tier-2 metrics evaluated and recorded
- [ ] Integrated Gradients XAI producing token attributions
- [ ] XAI latency measured separately
- [ ] Tier-2 FastAPI service containerized
- [ ] Deployed to Google Cloud Run
- [ ] Tier-1 API modified with escalation logic
- [ ] Combined system evaluated on full test set
- [ ] Tier-1-only vs Tier-1+Tier-2 comparison table complete
- [ ] All decisions recorded in DECISIONS.md
- [ ] Code audit issues from documentation fixed
