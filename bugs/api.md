# NETRA API bug hunt (api/, tier2_service/, requirements)

Format: severity | file:line | description | suggested fix. All items verified by reading code (runtime-only caveats noted).

## High

1. **High | api/main.py:576-581** | `elif cls == "LEGITIMATE" and high_threat_signals >= 3` is unreachable: the preceding `if ... >= 2` branch always catches it first. The intended PHISHING escalation (score floor 0.75) never happens. The 0.18 / 0.75 floors are also hard-coded rather than derived from `state` thresholds. | Test `>= 3` first, then `>= 2`; derive floors from state thresholds.

2. **High | api/main.py:720-753** | `POST /admin/recalibrate` has no authentication and no rate limit (limiter only covers `/predict`). Anyone who can reach the service can change live thresholds (e.g. phishing threshold 0.99 turns phishing into LEGITIMATE) and the change is persisted to disk. | Require an admin token or disable in production; include in rate limiter.

3. **High | api/main.py:111-113** | Rate limiter keys on the leftmost `X-Forwarded-For` entry, which is client-controlled. Behind Cloud Run the proxy appends the real client IP (not first), so a caller can send a random XFF per request and bypass the 20/min limit. XFF is also trusted when not behind any proxy. | Use the right-most trusted hop (configurable proxy count) or uvicorn `--proxy-headers --forwarded-allow-ips`; only honor XFF when configured.

4. **High | api/main.py:102,121,136** | Memory leak: `_rate_limits` keys are never evicted; pruning only happens for the IP currently served. With spoofed XFF (#3) this is an unauthenticated memory-exhaustion vector. | Delete key when pruned list is empty; periodic sweep; cap dict size.

5. **High | api/tier2_service/main.py:188-223 (model.py:93, xai.py:121); api/main.py:652,664** | `async def predict` runs synchronous PyTorch inference and Captum Integrated Gradients (50 steps x 512 tokens on CPU, seconds) directly on the event loop. All other requests, including `/health`, block meanwhile; Docker HEALTHCHECK (10s timeout) can fail under load. Same in Tier-1 (`_predict_distilbert` called synchronously). | Make handlers plain `def` or use `asyncio.to_thread`; limit concurrency with a semaphore.

6. **High | api/tier2_service/main.py:45-50,56-66,188** | Tier-2 service is unauthenticated, CORS `allow_origins=["*"]` with all methods/headers, no rate limit and no input-size limits (`body_text`, `urls`, `headers_available` unbounded). On Cloud Run it is a public CPU-expensive endpoint, bypassing the Tier-1 limiter. | Require auth (Cloud Run IAM / bearer token), drop wildcard CORS, add field length limits.

## Medium

7. **Med | api/main.py:560-565** | Classification contradicts the thresholds. `confidence = max(proba)`; `confidence < 0.70` means risk in (0.30, 0.70), which forces SUSPICIOUS even though risk >= phishing_threshold 0.25. A 0.60 score is "SUSPICIOUS" while `_risk_level` says CRITICAL (line 442-447); 0.25-0.30 scores are PHISHING. The escalation branch `verdict == "PHISHING" and confidence < conf_thresh` (line 617) can never fire for DistilBERT. Documented rule (<0.08 / 0.08-0.25 / >=0.25) is not what is implemented. | Classify purely by risk_score vs thresholds; use confidence only for escalation.

8. **Med | api/main.py:635-648** | `/health` hard-codes `model_type="distilbert"`, `model_loaded=state.db_ready`, `status="ok"` always. With RF fallback serving or no model, health misreports. | Use `state.model_type`, `state.model_loaded`; return 503 when nothing loaded.

9. **Med | api/main.py:226-235; api/tier2_service/model.py:63-72** | Auto-download writes directly to the final path via `urlretrieve`. An interrupted download leaves a truncated file that satisfies `.exists()` on next start, permanently breaking load. No checksum, no timeout, blocking at startup. | Download to temp, verify SHA-256, `os.replace`; set timeout.

10. **Med | api/main.py:261 vs model.py:84** | Tier-1 `torch.load(...)` lacks `weights_only=True` (pickle code execution) on a network-downloaded, unverified file. Tier-2 sets it correctly. | Add `weights_only=True` and checksum verification.

11. **Med | api/main.py:307-321,399-401,532-538** | No input bounds on `body_text`, `urls`, `subject`, `headers`. Regex scans, tokenization and per-URL Levenshtein against all brands (`check_typosquatting`) run over arbitrarily large input. | Add `max_length`/`max_items`, body size cap.

12. **Med | api/main.py:532-537** | URLs deduplicated via `set(all_urls)`: string-hash randomization makes ordering nondeterministic across processes, so which URLs survive 256-token truncation (and the score) varies for identical input. | Use order-preserving `dict.fromkeys`; cap count.

13. **Med | api/tier2_client.py:23-24,68-96; api/main.py:695** | Tier-1 awaits Tier-2 inline with 10s timeout x 2 retries and no backoff: up to ~20s added to a request documented at ~180ms+800ms. Timeout retries double load on a slow Tier-2; no circuit breaker. | Shorter timeout, backoff, circuit breaker; retry only connect errors/503.

14. **Med | api/tier2_service/xai.py:42-44,51-66,116-128** | Likely broken attribution: `LayerIntegratedGradients` hooks `word_embeddings`, but `_forward_for_ig` calls `roberta(inputs_embeds=...)`, which bypasses that module, and `attribute()` is given embeddings instead of `input_ids`. The hooked layer is never invoked, so Captum errors or returns meaningless values; the broad `except` (line 168) silently returns `top_tokens=[]`. (Runtime unverified: torch/captum not installed here.) | Use plain `IntegratedGradients` on embeddings, or `LayerIntegratedGradients` with an `input_ids` forward; surface failures.

15. **Med | api/tier2_service/Dockerfile:36-38; .gitignore:21** | Dockerfile `COPY ml/models/roberta_tier2.pt`, but `.pt` is gitignored, so image cannot be built from a fresh clone/CI, and model.py's auto-download is never used in the container. No `.dockerignore`, so the whole repo (incl. ~770MB weights) is sent as build context. | Drop COPY and download at startup/build; add `.dockerignore`.

16. **Med | Dockerfile:50-52; tier2_service/main.py:127-128,180-185** | HEALTHCHECK passes while model not loaded: `/health` returns HTTP 200 with `status:"not_ready"`. If load fails the container keeps running, serving 503 forever. `Tier2Model.load()` can also raise (missing config/tokenizer after download) and crash startup. | Return 503 when not ready; exit on load failure.

17. **Med | api/main.py:203-212,739-743** | Recalibrate mutates state before `_save_thresholds()`; if write fails (read-only FS on Cloud Run) it 500s but new thresholds stay active. Write is non-atomic and drops other keys (val_f1 etc.). Recalibration does not affect Tier-2 escalation policy (lines 603-606) and `suspicious_upper` is unused. | Temp+rename, persist before mutating, merge existing keys.

18. **Med | api/main.py:700-707** | Final response mixes tiers: classification/risk/confidence from Tier-2 (risk = P(PHISHING) of 3-class softmax) but `threshold_used` and `risk_level` use Tier-1 thresholds/scale, so a Tier-2 LEGITIMATE can carry HIGH risk_level. `tier2_config.json` thresholds (0.35/0.65) are unused: model.py:116 uses plain argmax. | Derive risk_level from final verdict; apply or remove configured Tier-2 thresholds.

## Low

19. **Low | api/main.py:88-95** | `"chrome-extension://*"` in `allow_origins` matches nothing (Starlette compares literally); `http://localhost` without port also won't match `localhost:8000/3000`. | Use `allow_origin_regex`.

20. **Low | api/main.py:412-414** | `req.sender.split("@")[-1]` on `Name <a@paypa1.com>` yields `paypa1.com>`, which can defeat typosquat detection. | Use `email.utils.parseaddr`.

21. **Low | api/main.py:569,668,449,603-605** | `_extract_signals` runs twice per request; `_classify` is dead code; `lower`/`upper` in `should_escalate` unused. | Compute once; remove dead code.

22. **Low | api/main.py:715-717** | `Prediction failed: {str(e)}` leaks internal exception text; 503 message references nonexistent `ml/train.py` flow. | Generic message; log details.

23. **Low | api/main.py:218; tier2 main.py:110; tier2_client.py:37** | Deprecated `on_event("startup")`; no shutdown handler, so the httpx.AsyncClient is never closed (resource leak). | Use lifespan and `await tier2_client.close()`.

24. **Low | api/tier2_service/main.py:145** | `sys.path.insert(0, ROOT)` runs on every request, growing `sys.path` unboundedly; only `ImportError` handled. | Do once at import.

25. **Low | api/tier2_service/main.py:150-161** | Fallback extractor treats softfail/neutral as "none" and hard-codes `sender_domain_match=1.0`, silently producing different features than the real extractor. | Fail loudly or share one implementation.

26. **Low | api/tier2_service/model.py:52** | Device selection omits MPS contrary to CLAUDE.md convention. | Add mps branch.

27. **Low | api/tier2_service/xai.py:98,153** | Hard-coded `max_length=512` ignores config `max_length`; `abs()` discards attribution sign; normalization only over top-k. | Pass config max_len; keep sign or document.

28. **Low | api/tier2_service/Dockerfile:44,55** | `ENV PORT=8080` but CMD hard-codes `--port 8080` (Cloud Run injects PORT); runs as root; gcc/g++ unnecessary. CLAUDE.md says Tier-2 is on 8001 while Dockerfile/client default use 8080. | Use `${PORT}`, non-root user, align docs.

29. **Low | requirements.txt, api/tier2_service/requirements.txt** | Tier-2 requirements fully unpinned; root leaves `torch`/`transformers` unpinned (checkpoint drift risk). Root lists `httpx` twice (`==0.27.2` in Dev section and `>=0.27.0`) though it is a runtime dependency of `api/tier2_client.py`. | Pin versions; move httpx to runtime, drop duplicate.

## Summary
High 6, Med 12, Low 11 (29 findings). api/__init__.py: no issues.
