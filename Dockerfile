# ==============================================================================
# NETRA - AI Phishing Detection API Dockerfile for Google Cloud Run
# Optimized for CPU inference & Cloud Build
# ==============================================================================
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    PYTHONPATH=/app \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Install minimal OS dependencies needed for fetching weights and running healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only wheel first to keep image lightweight (~180MB vs ~4GB with CUDA)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model configs
COPY api/ ./api/
COPY ml/ ./ml/

# Download the 254MB DistilBERT model weights from GitHub Release (v2.0.0)
# This bakes the weights directly into the container so Cloud Run instances start instantly
RUN echo "Downloading DistilBERT model weights from GitHub Release..." && \
    curl -L -f -sS -o ml/models/distilbert_tier1.pt \
    https://github.com/ramanan-2735/NETRA-The-AI-Eye-Against-Phishing/releases/download/v2.0.0/distilbert_tier1.pt && \
    ls -lh ml/models/distilbert_tier1.pt

# Pre-cache distilbert-base-uncased architecture weights in the image
# This ensures zero network calls on container cold-start
RUN python -c "from transformers import DistilBertModel; DistilBertModel.from_pretrained('distilbert-base-uncased')"

# Create non-root user for security best practices
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose default Cloud Run port
EXPOSE 8080

# Start Uvicorn bound to 0.0.0.0 and dynamic $PORT injected by Cloud Run
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
