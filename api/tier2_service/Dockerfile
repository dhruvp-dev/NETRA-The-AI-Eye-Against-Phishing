# =============================================================================
# NETRA Tier-2 Service - Dockerfile
# =============================================================================
# Containerizes the RoBERTa Tier-2 escalation service for Cloud Run.
# Uses CPU-only PyTorch (Cloud Run doesn't have GPU).
#
# Build:
#   docker build -t netra-tier2 -f api/tier2_service/Dockerfile .
#
# Run locally:
#   docker run -p 8080:8080 netra-tier2
# =============================================================================

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY api/tier2_service/requirements.txt /app/requirements.txt

# Install CPU-only PyTorch first, then other deps
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY api/tier2_service/ /app/api/tier2_service/
COPY ml/features/ /app/ml/features/
COPY ml/__init__.py /app/ml/__init__.py
COPY ml/models/roberta_tier2.pt /app/ml/models/roberta_tier2.pt
COPY ml/models/roberta_tier2_config.json /app/ml/models/roberta_tier2_config.json
COPY ml/models/roberta_tier2_tokenizer/ /app/ml/models/roberta_tier2_tokenizer/

# Create __init__.py files for imports
RUN touch /app/api/__init__.py

# Environment
ENV PORT=8080
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

# Run
CMD ["uvicorn", "api.tier2_service.main:app", "--host", "0.0.0.0", "--port", "8080"]
