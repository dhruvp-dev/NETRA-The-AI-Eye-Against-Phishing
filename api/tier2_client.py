# -*- coding: utf-8 -*-
"""
NETRA - Tier-2 Async HTTP Client
==================================
Calls the Tier-2 RoBERTa Cloud Run service from the Tier-1 API.
Handles timeouts, retries, and graceful fallback.

Usage:
    from api.tier2_client import Tier2Client
    client = Tier2Client(base_url="https://netra-tier2-xxx.run.app")
    result = await client.predict(payload)
"""

import json
import logging
import os
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

# Default Cloud Run endpoint (set via environment variable)
DEFAULT_TIER2_URL = os.getenv("NETRA_TIER2_URL", "http://localhost:8080")
TIMEOUT_SECONDS   = float(os.getenv("NETRA_TIER2_TIMEOUT", "10.0"))
MAX_RETRIES       = int(os.getenv("NETRA_TIER2_RETRIES", "2"))


class Tier2Client:
    """Async HTTP client for calling the Tier-2 escalation service."""

    def __init__(self, base_url: str = DEFAULT_TIER2_URL):
        self.base_url = base_url.rstrip("/")
        self._client = None

    async def _get_client(self):
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(TIMEOUT_SECONDS),
            )
        return self._client

    async def health_check(self) -> bool:
        """Check if Tier-2 service is healthy."""
        try:
            client = await self._get_client()
            resp = await client.get("/health")
            data = resp.json()
            return data.get("status") == "ok" and data.get("model_loaded", False)
        except Exception as e:
            log.warning(f"Tier-2 health check failed: {e}")
            return False

    async def predict(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Send an escalated email to Tier-2 for analysis.

        Args:
            payload: Dict containing email fields + tier1 metadata:
                - subject, body_text, urls, sender, reply_to, headers_available
                - tier1_risk_score, tier1_confidence, tier1_verdict, tier1_signals

        Returns:
            Tier-2 prediction result dict, or None if service is unavailable.
        """
        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                client = await self._get_client()
                resp = await client.post("/predict", json=payload)

                if resp.status_code == 200:
                    result = resp.json()
                    log.info(
                        f"Tier-2 result: verdict={result.get('verdict')} "
                        f"risk_score={result.get('risk_score')} "
                        f"latency={result.get('processing_time_ms')}ms "
                        f"xai={result.get('xai_latency_ms')}ms"
                    )
                    return result

                elif resp.status_code == 503:
                    log.warning(f"Tier-2 not ready (attempt {attempt}/{MAX_RETRIES})")
                    last_error = f"HTTP 503: {resp.text}"
                else:
                    log.error(f"Tier-2 error {resp.status_code}: {resp.text}")
                    last_error = f"HTTP {resp.status_code}: {resp.text}"
                    break  # Don't retry on 4xx errors

            except Exception as e:
                last_error = str(e)
                log.warning(f"Tier-2 call failed (attempt {attempt}/{MAX_RETRIES}): {e}")

        log.error(f"Tier-2 unavailable after {MAX_RETRIES} attempts: {last_error}")
        return None  # Fallback to Tier-1 result

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
