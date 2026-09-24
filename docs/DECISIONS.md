# NETRA - Architecture Decision Records

## Decision 1: Tier-2 Model Selection

**Date:** 2026-09-23
**Status:** Accepted

### Context
Tier-1 (DistilBERT, 66M params) achieves 99.93% F1 on the full dataset but may miss
hard edge cases. A heavier model is needed as an escalation layer for uncertain cases.

### Decision
Selected **RoBERTa-base** (125M params, 12 layers, 512 max tokens) as the Tier-2 model.

### Rationale
- Stronger pretraining than BERT (10x more data, dynamic masking)
- Same transformer family - consistent architecture
- 512 token context (vs DistilBERT's 256) captures more email content
- Validated in phishing detection literature (Paper 12 in NETRA lit review)
- Community support and stability for production use

### Alternatives Considered
| Model | Why Not |
|---|---|
| BERT-base | RoBERTa is strictly better trained |
| DeBERTa-v3-small | Better benchmarks but more complex, less stable |
| SecBERT/PhishBERT | Domain-specific but niche, uncertain maintenance |

---

## Decision 2: Escalation Policy

**Date:** 2026-09-23
**Status:** Accepted

### Context
Not every email should go through the heavier Tier-2 model. Only uncertain/hard cases
should be escalated.

### Decision
Escalation triggers when ANY of these conditions are met:
1. Tier-1 verdict is SUSPICIOUS (risk score in corridor)
2. Tier-1 verdict is PHISHING but confidence < 0.70
3. Tier-1 verdict is LEGITIMATE but threat signals fired (typosquatting, auth failure)

### Training Data Strategy
For training data generation, the escalation corridor is **widened** (0.03-0.50 vs
production 0.08-0.25) to give Tier-2 enough samples to learn from.

---

## Decision 3: 3-Class Output

**Date:** 2026-09-23
**Status:** Accepted

### Context
Tier-1 uses binary classification + threshold corridor for SUSPICIOUS.
Tier-2 only sees hard/uncertain cases.

### Decision
Tier-2 uses direct **3-class output** (LEGITIMATE/SUSPICIOUS/PHISHING).

### Rationale
Since Tier-2 only processes escalated cases (already uncertain), training directly
on 3 classes is more natural than binary + threshold.

---

## Decision 4: XAI Method

**Date:** 2026-09-23
**Status:** Accepted

### Decision
Use **Integrated Gradients** (via Meta's captum library) for token-level attribution.

### Rationale
- Mathematically grounded (path integral from baseline to input)
- Token-level granularity shows exactly which words triggered the verdict
- captum is the standard PyTorch XAI library
- Target: adds no more than 300ms on top of inference

---

## Decision 5: Deployment Architecture

**Date:** 2026-09-23
**Status:** Planned

### Decision
- Tier-1: Runs locally (user's machine) via uvicorn
- Tier-2: Deployed to Google Cloud Run as a Docker container
- Communication: Tier-1 calls Tier-2 via async HTTP (httpx)
- Fallback: If Tier-2 is unavailable, Tier-1 result is final

### Configuration
- Cloud Run memory: 2GB minimum
- Min instances: 1 (avoid cold start during demo)
- Timeout: 10s per request

---

## Decision 6: Combined System Performance

**Date:** TBD
**Status:** Pending Evaluation

| Metric | Tier-1 Only | Tier-1 + Tier-2 |
|---|---|---|
| Phishing Recall | 99.92% | TBD |
| FPR | TBD | TBD |
| F1 | 99.93% | TBD |
| Escalation Rate | N/A | TBD |
| Avg Latency (non-escalated) | ~180ms | ~180ms |
| Avg Latency (escalated) | N/A | TBD |

> If Tier-2 does not improve recall meaningfully, that result will be
> documented honestly. A null result is a valid research finding.
