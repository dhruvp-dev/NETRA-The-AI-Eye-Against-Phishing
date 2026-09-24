#!/usr/bin/env bash
# ==============================================================================
# NETRA - Deploy FastAPI Phishing Detection Service to Google Cloud Run
# Builds remotely in Cloud Build (Zero local Docker required)
# ==============================================================================

set -euo pipefail

# Configuration defaults
SERVICE_NAME="${SERVICE_NAME:-netra-api}"
REGION="${REGION:-asia-south1}"       # Default to Mumbai (or change to us-central1)
CPU="${CPU:-2}"                       # 2 vCPUs recommended for DistilBERT
MEMORY="${MEMORY:-2Gi}"               # 2 GiB RAM
MIN_INSTANCES="${MIN_INSTANCES:-0}"   # Scale to 0 when idle to save cost
MAX_INSTANCES="${MAX_INSTANCES:-5}"

echo "============================================================"
echo " NETRA: Google Cloud Run Remote Deployment"
echo "============================================================"

# Check gcloud CLI
if ! command -v gcloud &> /dev/null; then
    echo "❌ Error: 'gcloud' CLI is not found in PATH."
    echo "Please run this script inside Google Cloud Shell or install Google Cloud SDK."
    exit 1
fi

# Check active GCP project
PROJECT_ID=$(gcloud config get-value project 2>/dev/null || true)
if [ -z "$PROJECT_ID" ] || [ "$PROJECT_ID" = "(unset)" ]; then
    echo "⚠️  No active Google Cloud project set."
    read -rp "Enter your Google Cloud Project ID: " PROJECT_ID
    gcloud config set project "$PROJECT_ID"
fi

echo "✔ Active GCP Project: $PROJECT_ID"
echo "✔ Target Service:    $SERVICE_NAME"
echo "✔ Target Region:     $REGION"
echo "✔ CPU / Memory:      $CPU vCPU / $MEMORY"
echo ""

# Enable required Google Cloud APIs
echo "⏳ Ensuring required Google Cloud APIs are enabled..."
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com

echo "🚀 Submitting build to Cloud Build and deploying to Cloud Run..."
echo "   (Remote cloud build: no local Docker daemon needed)"
echo ""

gcloud run deploy "$SERVICE_NAME" \
    --source . \
    --region "$REGION" \
    --platform managed \
    --cpu "$CPU" \
    --memory "$MEMORY" \
    --min-instances "$MIN_INSTANCES" \
    --max-instances "$MAX_INSTANCES" \
    --port 8080 \
    --allow-unauthenticated

# Retrieve service URL
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --platform managed --region "$REGION" --format 'value(status.url)')

echo ""
echo "============================================================"
echo "🎉 Deployment Complete!"
echo "Service URL: $SERVICE_URL"
echo "API Docs:    $SERVICE_URL/docs"
echo "Health:      $SERVICE_URL/health"
echo "============================================================"
echo ""
echo "Testing /health endpoint..."
curl -s "$SERVICE_URL/health" | python3 -m json.tool || true
echo ""
