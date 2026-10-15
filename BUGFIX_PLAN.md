# NETRA Bug-Fix Implementation Plan

## Context
Three bug reports in `bugs/` list ~77 findings (many duplicates): `api.md` (A1–A29), `ml.md` (ML1–ML23), `notebooks_tests_docs.md` (H1–H5, M1–M9, L1–L11). Spot-checks of `api/main.py` confirm them (unreachable `>= 3` branch at 576-581, leftmost-XFF limiter at 113, no `weights_only` at 261, `"Phishing URL:"` prompt at 536, etc.).

Decisions from the user:
- **Drop Tier-2 entirely**: remove it from the pipeline, code, configs, notebooks and docs. That makes every Tier-2-only bug moot.
- **Fix the code only.** The user retrains DistilBERT afterwards. The shipped `distilbert_tier1.pt` stays in use until then.
- **Admin auth**: use a bearer token from an env var.
- **This is a plan only.** A different model will implement it, so every step names exact files and functions.

Bug IDs are cited as `A#`, `ML#`, `H#/M#/L#` so the implementer can check each one against `bugs/*.md`.

---

## Phase 0 — Remove Tier-2 (do first; it shrinks everything else)

**Delete:**
- `api/tier2_service/` (whole dir), `api/tier2_client.py`
- `ml/train_roberta_tier2.py`, `ml/generate_tier2_dataset.py`, `ml/calibrate_tier2_threshold.py`
- `ml/evaluation/tier2_evaluation.py`, `ml/evaluation/roberta_tier2_metrics.json`
- `ml/models/tier2_config.json`, `ml/models/roberta_tier2_config.json`, `ml/models/roberta_tier2_tokenizer/`
- `notebooks/NETRA_RoBERTa_Tier2_Training.ipynb`
- `docs/NETRA_TIER2_IMPLEMENTATION_PLAN.md`

**Edit `api/main.py`:**
- Remove `TIER2_CONFIG_PATH`, the `tier2_*` fields on `ModelState`, the Tier-2 block in `load_models` (lines 284-299), `should_escalate` (595-630), and the escalation block in `predict` (678-698).
- Remove the `escalated_to_tier2`, `tier2_verdict`, `tier2_confidence` and `tier2_xai_tokens` fields from `PredictResponse`. The extension only reads `classification` (`extension/popup.js:295`, `service-worker.js:56`), so it is not affected.
- `predict` returns the Tier-1 result directly, with `model_type = state.model_type`.

**Other edits:**
- `requirements.txt`: drop `captum` and other Tier-2-only deps. Keep `httpx` once, in the dev section, for `TestClient`.
- `.gitignore`: drop the roberta entries.
- `tests/stress_test_tier1.py`: remove the `should_escalate` / `is_escalated` usage (also fixes the L7 dead branch).
- Docs (`CLAUDE.md`, `README.md`, `docs/DECISIONS.md`): remove the two-tier diagram, port 8001, Tier-2 commands and the v3.0.0 weight note. Mark DECISIONS "Decision 6" as "Tier-2 dropped" with a one-line reason: degenerate model, phishing recall 2/17.
- Leave the GitHub Release `v3.0.0` alone. Deleting it is the user's call.

**Resolved by removal (no further work):**
- From `api.md`: the Tier-2 parts of A5, plus A6, A13, A14, A15, A16, A18, A24, A25, A26, A27, A28, the Tier-2 half of A9 and A23, and the Tier-2 pinning in A29.
- From `ml.md`: ML3, ML4, ML10, ML11, ML12, ML19.
- From `notebooks_tests_docs.md`: H4, M3, M4, M5, M6, M7, L2, and the port 8080/8001 part of L3.

---

## Phase 1 — Shared helpers (removes train/serve skew at the source)

Create **`ml/features/text_builder.py`**:
- `build_model_text(subject, body, urls) -> str`. It writes `Subject: …\nBody: …\nURL: …`, with a neutral `URL:` prefix (never `Phishing URL:`). URLs are deduplicated in order with `dict.fromkeys` and capped at `MAX_URLS = 20`.
- `extract_urls(subject, body, explicit_urls) -> list[str]`. It uses the single regex now duplicated in `api/main.py:400,534`. It merges explicit URLs with auto-extracted ones, in order and deduplicated. Fixes ML2/H2, A12, ML21 and ML8(b).
- `URGENCY_KEYWORDS`: one constant. Re-export it from `ml/features/text_features.py`, and have `api/main.py:63` import it (ML23).

**Edit `ml/features/header_features.py`:**
- Add `normalize_auth(value)` that maps `softfail→fail` and `neutral|temperror|permerror→none`. Use it in `_one_hot_auth`, and in `ml/data_pipeline.py` `parse_auth_results` / `parse_spf_header` (ML14).
- Parse the domain with `email.utils.parseaddr` (A20).
- Add `extract_from_row(headers_json, sender, reply_to)` for training code.

**Edit `ml/features/url_features.py` `check_typosquatting`** (ML6, ML20):
- Use `tldextract` (add it to requirements) to get the registrable domain.
- Return "legitimate" only when the registrable domain equals the brand's real domain.
- Flag a brand token that appears in a subdomain or a hyphen piece of a non-brand registrable domain (`paypal.evil.com`, `secure-paypal-login.com`).
- Add a small allow-list (e.g. `stream.com`) for false positives at distance ≤ 0.2.
- Compute `_subdomain_depth` from the tldextract `subdomain`.

---

## Phase 2 — Tier-1 API (`api/main.py`)

**Correctness:**
- **A7 + A1 + A21 — classification.** Rewrite `_predict_distilbert`:
  - Classify only by `risk_score` against `state.suspicious_lower` / `state.phishing_threshold`. Reuse the existing `_classify` at line 449 instead of deleting it.
  - Keep `confidence` as an output only.
  - Multi-signal override: test `>= 3` → PHISHING first, then `>= 2` → SUSPICIOUS.
  - Set the floors from thresholds: SUSPICIOUS floor `= state.suspicious_lower`, PHISHING floor `= state.phishing_threshold`.
  - Compute `_extract_signals` once in `predict` and pass it in.
- **`_risk_level`**: derive it from the final classification plus thresholds, not from hard-coded 0.08/0.20/0.50 values.
- **Use `text_builder` in `_predict_distilbert` and `_predict_rf`** (ML2, ML8). The RF path must pass `req.subject` and auto-extracted URLs.
- **Old-weights mitigation (temporary):**
  - Read `distilbert_config.json` key `"header_features_trained"`, default `false`. When it is false, feed a zero header vector, which is what the shipped model saw in training (H1/ML1).
  - Read key `"text_format"`, default `"body_only"`. When it is `"body_only"`, send only the body text. Otherwise use `build_model_text`.
  - The retrain script (Phase 3) writes `header_features_trained: true` and `text_format: "v3"`. Old weights keep behaving as trained, and new weights get the fixed pipeline automatically.
- **A8/L5 — `/health`**: use `state.model_type` and `state.model_loaded`, and return 503 with `status="not_ready"` when no model is loaded.

**Security / robustness:**
- **A2 — admin auth.**
  - Add a `require_admin` dependency that compares `Authorization: Bearer` against `os.getenv("NETRA_ADMIN_TOKEN")` with `hmac.compare_digest`.
  - If the env var is unset, the endpoint returns 404 (disabled).
  - Rate-limit the admin endpoint as well.
- **A3/A4 — rate limiter.**
  - Add env `NETRA_TRUSTED_PROXY_HOPS` (default 0). When it is 0, ignore XFF and use `request.client.host`. When it is N, take the N-th entry from the right of XFF. On Cloud Run use 1.
  - Delete a key whenever its pruned list is empty.
  - Run a full sweep of stale keys every 60s (tracked via a `_last_sweep` timestamp).
  - Hard-cap the dict size (e.g. 100k): over the cap, clear the oldest entries.
  - Keep the proxy-aware pattern that CLAUDE.md requires.
- **A11 — input bounds** in `PredictRequest` (Pydantic v2 `Field`):
  - `body_text` `max_length=100_000`, `subject` 1_000, `sender`/`reply_to` 512.
  - `urls` `max_length=50`, with each URL ≤ 2048 characters (`Annotated[str, StringConstraints(max_length=2048)]`).
  - `headers` dict: at most 50 keys.
  - Replace `class Config` with `model_config = ConfigDict(json_schema_extra=...)`.
- **A5 — blocking inference.** Make `predict` call `await asyncio.to_thread(_predict_distilbert, req, signals)`. Guard it with an `asyncio.Semaphore(int(os.getenv("NETRA_MAX_CONCURRENCY", 2)))`.
- **A9/A10 — weight download.**
  - Add `_download_weights(url, dest, sha256)`: download to `dest.with_suffix(".part")` with `urllib.request.urlopen(url, timeout=60)` in chunks, check SHA-256, then `os.replace`.
  - Store the expected hash in `distilbert_config.json` as `"weights_sha256"`. The implementer computes it from the current v2.0.0 asset.
  - Use `torch.load(..., weights_only=True)`.
- **A17 — recalibrate.**
  - Validate `0 < suspicious_lower < phishing_threshold <= 1` and set `suspicious_upper = phishing_threshold` (A17 says it is otherwise unused).
  - In `_save_thresholds`: read the existing JSON, merge in the new keys, write to a temp file, `os.replace`, then mutate `state`. If the write fails, return 500 and leave `state` untouched.
- **A19 — CORS**: replace the wildcard with `allow_origin_regex=r"^(chrome-extension://[a-p]{32}|http://(localhost|127\.0\.0\.1)(:\d+)?)$"`.
- **A22 — errors**: the 500 response becomes a generic `"Prediction failed"` and the details go to the log only. Fix the 503 text, which references `ml/train.py` incorrectly.
- **A23 — startup**: replace `@app.on_event("startup")` with a `lifespan` context manager.
- **L6 — tracked threshold file**:
  - Commit `ml/models/threshold_config.default.json` and gitignore `threshold_config.json`.
  - `_load_thresholds` falls back to the default file.
  - The implementer must `git rm --cached` the live file.
- **A29/L9 — dependencies**:
  - Pin `torch` and `transformers` in `requirements.txt` to the versions used to train the shipped weights. `transformers==4.40.0` matches the notebook.
  - Remove the duplicate `httpx`, and add `tldextract`.

---

## Phase 3 — ML pipeline & training (code only; user retrains)

**`ml/data_pipeline.py`:**
- **ML2/H2**: lines 390 and 422 store `body_text = ""` and `urls=[url]` for URL-only records instead of `"Phishing URL: …"`. Add a `--include-url-only` flag, off by default. Tag these rows with `source` so evaluation can exclude them.
- **ML1/H1**: write the 10 header columns using `header_features.extract_from_row` and `HEADER_FEATURE_NAMES`, so every consumer gets real values.
- **ML5**:
  - Add `_decode_header(raw)` using `email.header.make_header(decode_header(...))` inside try/except. Use it for subject, from and reply-to.
  - Move the try/except inside the per-message loops of `read_enron_mbox`, `read_nazario_mbox` and `read_spamassassin_mbox`.
- **ML15**: prefer `text/plain`, and fall back to HTML stripped with `html.parser` / a regex. Decode with `part.get_content_charset()` and fall back to utf-8 with `errors="replace"`.
- **ML9**: split stratified by `(label, source)` (sklearn `train_test_split(stratify=…)`). Choose the cross-source holdout from sources that contain both classes. Fix the docstring.
- **ML16**: keep the spam→phishing label, but write a `label_origin` column (`spam`/`phish`/`ham`) and document it in the module docstring.
- **ML17**:
  - Before deduplicating, drop body hashes that appear with conflicting labels.
  - Create `--output` after argparse runs.
  - Raise a clear error when `RAW_DIR` is missing.

**`ml/train_distilbert.py`:**
- **ML1**: read header columns by `HEADER_FEATURE_NAMES` (no more `sender_domain_match`).
- **ML2/M3**: build inputs with `build_model_text(subject, body, urls)`.
- **ML18/M9**:
  - Add `--epochs`, `--batch_size`, `--lr`, `--fp16`, `--seed`, and seed everything with `random`, `np` and `torch`.
  - Use the CLAUDE.md device line, including MPS.
  - Always save a checkpoint on epoch 1; replace the `> 0.0` guard with `>=`.
  - Load the best checkpoint with `weights_only=True`.
- **M2/ML9**: after training, evaluate on `test_std` and `test_cross`, with URL-only sources excluded and reported separately. Write the results to `distilbert_config.json` under `"test_metrics"`. Also write `header_features_trained: true`, `text_format: "v3"`, `max_length`, and `weights_sha256` for the new `.pt`.
- Add a class-balance guard: warn when the class ratio is above 10:1, and log the per-class counts.

**`ml/calibrate_threshold.py`** (ML7, ML22):
- Calibrate **DistilBERT** val probabilities (load the model and score `val`), not the RF.
- Write only `phishing_threshold`, `suspicious_lower`, `suspicious_upper` and `method` with a read-merge-write.
- Optional `--model rf` writes a separate `rf_threshold_config.json`.
- Replace the per-threshold confusion matrix with sorted cumulative sums.

**`ml/train.py`** (RF, ML13, ML22): wrap the handcrafted features in a `StandardScaler`/`log1p` (`ColumnTransformer` fit on train only). Remove `class_weight="balanced"` when SMOTE is applied. Log the SMOTE failure at WARNING instead of swallowing it. Fix the mojibake log strings by replacing them with ASCII `->` and `OK`.

---

## Phase 4 — Notebooks (DistilBERT Colab + Local, IPDS)

Mirror Phase 3 in both `NETRA_DistilBERT_Training*.ipynb`:
- cell 13: use the pipeline's header columns and `HEADER_FEATURE_NAMES` (H1); fix the nested-quote f-string to `df['split']` (M1); add the class-balance guard and the Enron requirement (M2).
- cell 17: use `build_model_text` (H2).
- cells 21/23/26: evaluate the test splits (M2); write the same config keys as `train_distilbert.py`.
- L8 hygiene:
  - Seed cell.
  - Cell 4 git ops run with `check=True`, and before `reset --hard` they check that `.git` exists.
  - `tar.extractall(filter="data")`.
  - The download fallbacks raise instead of writing stub records.
- `NETRA_IPDS_Training.ipynb`:
  - Cells 6 and 8 raise on failure.
  - Cell 10: `legend()` after `axhline`, `set_xticks` before `set_xticklabels`, and plot the calibrated model.

---

## Phase 5 — Tests

- Rename/convert to **`tests/test_api.py`** (pytest, FastAPI `TestClient`, L3). Cover:
  - `/health` 200 and 503
  - 422 on an empty or oversized body
  - 429 after 20 requests, and XFF spoofing that does not bypass it when `NETRA_TRUSTED_PROXY_HOPS=0`
  - `/admin/recalibrate` 404 with no token, 401 with a wrong token, 200 plus persistence (on a tmp threshold file via monkeypatch)
  - the multi-signal override reaching PHISHING when there are ≥ 3 signals (monkeypatch `_predict_distilbert` to return a low score)
  - the classification boundaries 0.079 / 0.08 / 0.249 / 0.25
- **`tests/test_features.py`**:
  - typosquatting cases from ML6/ML20 (`paypal.evil.com`, `secure-paypal-login.com`, `apple.com.evil.ru` flagged; `paypal.com`, `stream.com` not flagged; `example.co.uk` depth 0)
  - `normalize_auth`
  - `parseaddr` sender
  - `build_model_text` determinism and no `"Phishing URL"`
  - `_decode_header` on `Subject: Caf\xe9`
- **`tests/stress_test_tier1.py`** (H3, L7):
  - Collect failures and `sys.exit(1)` when an expected verdict is missed or the p95 latency budget is exceeded.
  - Fix the counts and step labels, and print the device and model type.
  - Mark it so pytest skips it (`if __name__ == "__main__"`, documented as a script).
  - Expect it to fail on the current weights until the user retrains. Note this in the README.

---

## Phase 6 — Docs, repo hygiene

- **`CLAUDE.md`**:
  - One-tier architecture.
  - Thresholds live in `threshold_config.json` (M8).
  - Real training flags (M9).
  - Accurate component descriptions (L3): "post-hoc signal override", not "blends"; data_pipeline does no tokenization.
  - Document `NETRA_ADMIN_TOKEN`, `NETRA_TRUSTED_PROXY_HOPS` and `NETRA_MAX_CONCURRENCY`.
  - Add the env vars to the repo map.
- **`README.md`**:
  - Re-run Test 1–3 and paste the actual outputs (H5), plus a note to re-verify after retraining.
  - One set of metrics, with the split named (L1).
  - Update the training commands.
- **`docs/DECISIONS.md`**: align the metrics (L1); Decision 6 = Tier-2 dropped.
- **`.gitignore`** (L10): `.env*`, `venv/`, `.venv/`, `.ipynb_checkpoints/`, `**/*.pt`, `**/*.bin`, `*.safetensors`, `*.onnx`, `*.ckpt`, `*.zip`, `*.tar*`, `.claude/settings.local.json`, `ml/evaluation/*.png`, `ml/models/threshold_config.json`.
- **L11**: regenerate `skills-lock.json`, or remove the unlocked `.agents/skills/git-commit` and `multi-stage-dockerfile`. Ask the user which.

---

## Suggested commit order
1. Phase 0 (Tier-2 removal)
2. Phase 1 (helpers + tests for helpers)
3. Phase 2 (API)
4. Phase 5 API tests
5. Phase 3 (ML)
6. Phase 4 (notebooks)
7. Phase 6 (docs/hygiene)

Each commit leaves `pytest tests/` green.

## Verification
- `pytest tests/` passes: the new API and feature tests.
- `python -m uvicorn api.main:app --port 8000` starts; `/health` reports the correct `model_type`; `/docs` shows no tier2 fields.
- Manual curl checks:
  - 21 rapid `/predict` calls → 429; a spoofed `X-Forwarded-For` does not reset the limit.
  - `/admin/recalibrate` without a token → 404 or 401.
- `grep -ri "tier2\|roberta\|Phishing URL" api ml tests` → no hits (outside docs history).
- `python ml/data_pipeline.py` runs on raw data. `unified.csv` contains the 10 header columns and no `"Phishing URL:"` text.
- `python ml/train_distilbert.py --epochs 1 --batch_size 8 --seed 42` is a smoke run on a small sample. Its config contains `test_metrics`, `header_features_trained` and `text_format`.
- **Retrain (user):**
  1. `python ml/data_pipeline.py`
  2. `python ml/train_distilbert.py --epochs 3 --batch_size 16 --fp16 --seed 42`
  3. `python ml/calibrate_threshold.py`
  4. `python tests/stress_test_tier1.py` must exit 0
  5. Publish via `gh release create v4.0.0 ml/models/distilbert_tier1.pt …` and update the download URL and SHA-256 in `api/main.py` / `distilbert_config.json`.
