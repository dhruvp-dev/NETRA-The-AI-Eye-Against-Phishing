# Bug hunt: notebooks, tests, docs, .gitignore, skills-lock

Scope: notebooks/*.ipynb (code cells extracted with a Python script; "cell N" = raw notebook cell index), tests/stress_test_tier1.py, .gitignore, README.md, CLAUDE.md, docs/*, skills-lock.json.
No project files were modified. Findings are verified by reading code, or by running the stress test and README examples against the checked-in weights (py -3.14).

Counts: High 5, Med 9, Low 11.

---------------------------------------------------------------------------

## HIGH

### H1. Tier-1 notebooks train the header branch on all-zero features (train/serve skew)
- Files: notebooks/NETRA_DistilBERT_Training.ipynb cell 13 (also *_Local.ipynb cell 13). Cause in ml/data_pipeline.py:623-625.
- unified.csv schema_cols contains no spf_*/dkim_*/dmarc_*/sender_domain_* columns (only the `headers_available` JSON). The notebook does `if col not in df.columns: df[col] = 0`, so `header_proj` is trained only on zero vectors. The notebook never parses `headers_available`.
- At serving time api/main.py:545 builds real vectors (e.g. spf_none=1, dkim_none=1, dmarc_none=1, or spf_pass=1). Those inputs were never seen in training, so the header-fusion claim (README, CLAUDE.md) is unsupported.
- The notebook's HEADER_FEATURE_NAMES uses `sender_domain_match`, while ml/features/header_features.py emits `sender_domain_matches_reply_to`. Even if the columns existed, the name would not match.
- Fix: in the notebook, build the 10 features from `headers_available`/sender/reply_to via `ml.features.header_features.extract`, using that module's HEADER_FEATURE_NAMES. Or have data_pipeline.py write those columns. Then retrain.

### H2. Label-leaking text prefix "Phishing URL:" in training data vs. API prompt builder
- Files: ml/data_pipeline.py:390 and :422 (consumed by notebooks, DistilBERT cell 13/17); api/main.py:535-536.
- Every PhishTank/OpenPhish record (label=1) has body_text = "Phishing URL: <url>". No legitimate record has that prefix. The model can learn the literal prefix as a label shortcut. Source identity (URL-only vs. email) also separates the classes.
- The API (api/main.py:535) appends "Phishing URL: {u}" for every URL found in any email, including legitimate ones. This pushes legitimate emails that contain a link toward PHISHING.
- Notebook cell 17 trains on raw `body_text` only. The API sends "Subject: ...\nBody: ...\nPhishing URL: ...", so there is a second prompt-format skew.
- Fix: strip or neutralise the prefix in the pipeline (use the plain URL, or a neutral "URL:"). Make the API and the notebooks build the same prompt via one shared function. Evaluate per source, or keep URL-only records out of val/test metrics.

### H3. tests/stress_test_tier1.py cannot fail, and it currently hides a broken model
- File: tests/stress_test_tier1.py:165-169, 217.
- No assert, no sys.exit, no nonzero return. `badge = "WARN [CHECK]"` is only printed. The last line always says "All scenarios executed successfully...".
- I ran it. 11/11 cases return PHISHING (scores 0.95-0.9997), including the two LEGITIMATE cases (#1 Target order, #2 calendar invite) and the 50k-char legitimate body (#11). Three WARNs were printed, and the script still reported success and exited 0.
- Because the model returns PHISHING for everything, the "SUSPICIOUS/PHISHING" adversarial cases (5-8) pass trivially, so the suite cannot discriminate.
- Fix: collect failures and `sys.exit(1)` when any expected verdict is missed. Add assertions for the legitimate cases (FPR) and for a latency budget. Fix the underlying false positives (see H2).

### H4. Shipped Tier-2 RoBERTa model is degenerate but is wired in as the final verdict
- Files: ml/evaluation/roberta_tier2_metrics.json and ml/models/tier2_config.json (test_accuracy 0.9408, macro_f1 0.3912). Used by api/main.py:700-705, 712. Notebook: NETRA_RoBERTa_Tier2_Training.ipynb cells 14, 16, 20.
- Confusion matrix: SUSPICIOUS recall 0/5. PHISHING recall 2/17 = 0.118, with 12 of 17 phishing emails predicted LEGITIMATE. Accuracy is high only because 333 of 355 test rows are class 0.
- api/main.py overwrites Tier-1's classification, score and confidence with the Tier-2 values whenever Tier-2 responds. An escalated Tier-1 PHISHING can therefore be downgraded to LEGITIMATE, which lowers recall relative to Tier-1.
- Notebook cell 14 saves the best checkpoint on val macro-F1 with no minimum bar (best_val_macro_f1 was 0.4486). Cell 20 then writes the config and metrics anyway.
- Test support (17 phishing, 5 suspicious) is too small to be meaningful.
- Fix: gate cell 20 on a minimum per-class recall. Do not let Tier-2 override Tier-1 unless it beats Tier-1 on a held-out escalated set (for example only allow upgrades, or blend). Regenerate a larger escalated set. Document the null result in DECISIONS.md (Decision 6 is still TBD).

### H5. README sample cases contradict actual API output
- File: README.md "Test 1" (~line 92) and "Test 3" (~line 118).
- I ran them against the checked-in weights.
  - Test 1 (benign weekly update, sender sarah.connor@cyberdyne.com): README says LEGITIMATE, ~0.0000, LOW. Actual is SUSPICIOUS, 0.1007, MEDIUM.
  - Test 3 (micros0ft, failing SPF/DKIM/DMARC): README says SUSPICIOUS 0.08-0.25 MEDIUM. Actual is PHISHING 0.9867 CRITICAL.
  - Test 2 matches (PHISHING, 0.9997).
- The model output is essentially bimodal (~0.002 or ~0.998), so a "0.08-0.25" example is not reproducible.
- Fix: update the expected verdicts and scores to what the shipped model returns, or choose inputs that land in the corridor. Re-verify after any retraining.

---------------------------------------------------------------------------

## MEDIUM

### M1. Python 3.12-only f-string in the DistilBERT notebooks
- notebooks/NETRA_DistilBERT_Training.ipynb cell 13 (also *_Local cell 13): `print(f"Splits: {df["split"].value_counts().to_dict()}")`.
- Reusing the same quote type inside an f-string is a SyntaxError before Python 3.12. Colab runtimes are typically 3.10/3.11, and the README recommends Colab ("Option A").
- Fix: use `df['split']`.

### M2. Reported Tier-1 metrics come from the split used for checkpoint selection; test splits are never evaluated
- DistilBERT notebooks cells 13, 21, 23, 26. Cell 13 filters to `split in (train, val)`. The best checkpoint is picked on val F1 (cell 21) and the same val is reported as "Final" (cell 23). test_std, test_ext and test_cross are never used.
- The headline numbers in distilbert_config.json (best_val_f1 0.9993, recall 0.9992) are therefore optimistic. The split in data_pipeline.assign_splits is a plain shuffle (not source- or near-duplicate-aware).
- The numbers also imply a very imbalanced set. Precision 0.9994 with FPR 0.0182 requires roughly 30 times more phishing than legitimate rows (derived arithmetic). The Colab notebook skips Enron by default (cell 8), so its legitimate class is only SpamAssassin ham against tens of thousands of URL-only phishing rows. Cell 13 only guards `phishing < 100`.
- Fix: also report test_std and test_cross in the notebook. Add a class-balance guard. Enable or require the Enron legitimate data.

### M3. RoBERTa notebook trains on raw body_text but the Tier-2 service sends "Subject:/Body:/URL:" prompts
- NETRA_RoBERTa_Tier2_Training.ipynb cell 8 lines 162-164 (`train_df["body_text"]`) versus api/tier2_service/main.py:164-174 (build_input_text).
- It also differs from the Tier-1 prompt ("Phishing URL:"). docs/NETRA_TIER2_IMPLEMENTATION_PLAN.md section 4 specifies "Subject/Body/URLs".
- Fix: build the same prompt in training and serving through a shared helper.

### M4. Tier-2 `tier1_confidence` feature leaks: Tier-1 is run on its own training data
- ml/generate_tier2_dataset.py (generate_dataset), consumed by RoBERTa notebook cells 6 and 8 (`tier1_confidence`).
- Tier-1 scores every record, including the ones it was trained on, so confidences on training rows are inflated compared with live traffic. Tier-1 verdicts also define class 1 (SUSPICIOUS) in `map_tier2_label`.
- Fix: use out-of-fold Tier-1 predictions, or restrict the Tier-2 train/val sets to records Tier-1 did not train on.

### M5. Balanced WeightedRandomSampler plus sqrt-class-weighted loss both compensate for the same imbalance
- RoBERTa notebook cell 8 lines 182-187 and cell 12 lines 247-257.
- The sampler already makes batches roughly class-uniform. The loss additionally multiplies minority classes by sqrt-balanced weights (normalised so min=1). This double-corrects the prior, and the loss weights are computed from the unsampled train labels.
- Val loss also uses the weighted criterion although val is not resampled. This is a plausible contributor to H4.
- Fix: use one mechanism (sampler or loss weights). Compute val loss unweighted. Choose the epoch and thresholds on a per-class metric.

### M6. Notebook cell 20 overwrites ml/models/tier2_config.json, wiping `escalation_policy`
- RoBERTa notebook cell 20 lines 531-533. ml/calibrate_tier2_threshold.py:120 writes `escalation_policy` into the same file. api/main.py:603 reads it.
- Re-running the notebook after calibration silently drops the policy and the API falls back to hard-coded defaults. The committed tier2_config.json has no `escalation_policy` at all.
- The notebook also hard-codes `thresholds: {suspicious_lower: 0.35, phishing_lower: 0.65}`, which nothing in the API or service reads.
- Fix: read-merge-write the file and preserve existing keys. Remove or implement the hard-coded thresholds.

### M7. api `should_escalate` ignores the documented corridor (0.08-0.25) and the `lower`/`upper` policy values
- api/main.py:604-606 reads `lower`, `upper` and `conf_thresh`, but `lower` and `upper` are never used. CLAUDE.md (Core Components point 1 and the diagram), docs/DECISIONS.md Decision 2 and the plan doc describe a score corridor.
- Real behavior: SUSPICIOUS verdicts, PHISHING with confidence < 0.70, or LEGITIMATE with any threat signal.
- Fix: either implement the corridor check or fix the docs.

### M8. CLAUDE.md says thresholds live in distilbert_config.json; the code reads threshold_config.json
- CLAUDE.md "Threshold Consistency" and the Repository Map entry for distilbert_config.json ("Metadata & active threshold boundaries").
- api/main.py:48 and 184-212 use ml/models/threshold_config.json. distilbert_config.json has no thresholds. `/admin/recalibrate` rewrites threshold_config.json, which is git-tracked.
- Fix: correct the docs.

### M9. Documented training commands use flags that do not exist
- README.md "Option B" and CLAUDE.md "Model Training & Calibration": `python ml/train_distilbert.py --epochs 3 --batch_size 16 --fp16` and `python ml/train_roberta_tier2.py --epochs 6 --batch_size 16 --fp16`.
- ml/train_distilbert.py:434 and ml/train_roberta_tier2.py:473 define only `--data`. argparse will exit with an error on the other flags.
- Fix: add the flags to the scripts, or correct the docs.

---------------------------------------------------------------------------

## LOW

### L1. Metrics contradict each other across README, DECISIONS and the plan doc
- README.md "Validation F1 0.8114, FPR 0.0010" and docs/NETRA_TIER2_IMPLEMENTATION_PLAN.md sections 1 and 10.3 (recall 76.59%, F1 81.14%, FPR 0.10%).
- docs/DECISIONS.md Decision 1 and 6 say F1 99.93%, recall 99.92%.
- ml/models/distilbert_config.json says best_val_f1 0.9993 and best_val_fpr 0.0182.
- Fix: pick one verified run and update all four places. State which split.

### L2. Plan doc is stale and has internal errors
- docs/NETRA_TIER2_IMPLEMENTATION_PLAN.md line 7: "Status: Planning — Not Yet Implemented". The service, client, notebook and artifacts all exist. The success-criteria checkboxes are all unchecked.
- Line 108: "Concatenation → 779-dim" and "Linear(779→256)". The real model (notebook cell 10, api/tier2_service/model.py:27) is 768+32+1 = 801 with `Linear(801,256)`.
- Line 82: "same tokenizer family" as DistilBERT. RoBERTa uses byte-level BPE (vocab.json/merges.txt), DistilBERT uses WordPiece (vocab.txt).
- Section 6.2 says lr 2e-5, epochs 3-5, warmup 100. The notebook uses 1.5e-5 backbone with 1e-4 head, 6 epochs, 10% warmup ratio.
- Section 8 lists `ml/models/roberta_tier2/` with pytorch_model.bin. The actual files are ml/models/roberta_tier2.pt, roberta_tier2_tokenizer/ and roberta_tier2_config.json.
- Section 5 says the escalation threshold is "experimentally determined — do not hardcode". The corridor is hard-coded in api/main.py and generate_tier2_dataset.py.

### L3. CLAUDE.md descriptions that do not match the code
- "Blends semantic probability with heuristic risk boost": there is only a post-hoc override to SUSPICIOUS/PHISHING (api/main.py:576-581).
- ml/data_pipeline.py is described as doing "tokenization" (it does none). generate_tier2_dataset.py is described as "Synthesizes hard-negative ... datasets" (it only filters escalated Tier-1 records).
- `pytest tests/` is listed as "Run test suite", but tests/ contains only a script. Its filename does not match pytest's `test_*.py` or `*_test.py` patterns, so pytest collects 0 tests (exit code 5). The script also never goes through HTTP or TestClient, so the rate limiter, the 422/503 paths, response_model validation and /admin/recalibrate are never exercised, although it is billed as a stress/adversarial suite.
- Tier2Client default URL is http://localhost:8080 (api/tier2_client.py:22), while the docs say Tier-2 runs on port 8001.

### L4. Unreachable branch in api/main.py multi-signal override
- api/main.py:576-581. `if cls == "LEGITIMATE" and high_threat_signals >= 2` is checked first, so the `elif ... >= 3` (override to PHISHING) can never run. The documented override-to-PHISHING (README, plan doc) never fires.
- Fix: test `>= 3` first.

### L5. /health always reports model_type="distilbert"
- api/main.py:639-640. When DistilBERT fails to load and the RF fallback serves, /health still says `distilbert`, and `model_loaded=state.db_ready` is false even though RF is serving. README says it "Returns active model status".
- Fix: use `state.model_type` and `state.model_loaded`.

### L6. Runtime-mutated file is tracked in git
- ml/models/threshold_config.json is committed, and `_save_thresholds()` (api/main.py:203) rewrites it on every /admin/recalibrate. This leads to a dirty tree and accidental commits of demo thresholds.
- Fix: commit a default file and gitignore the live one.

### L7. stress_test_tier1.py cosmetic/accuracy bugs
- tests/stress_test_tier1.py:30 says "12 Extensive Test Cases"; there are 11. Line 168 hard-codes "/11".
- "[Step N/5]" is printed for steps 1-4 only; Step 5 never exists.
- Line 191 prints "Completed 50 full transformer forward passes on CPU" even when the RF fallback or CUDA is used.
- `should_escalate` (lines 160-163) is called with Tier-2 disabled (NETRA_TIER2_URL unset), so `is_escalated` is always False and the `(is_escalated and "SUSPICIOUS" in expected)` branch is dead.
- The script calls `load_models()`, which downloads a ~266 MB checkpoint on first run if missing.
- `asyncio.run()` per call includes event-loop startup in the latency numbers.

### L8. Notebook hygiene and robustness
- DistilBERT (Colab) cell 4 lines 19-30: if DRIVE_ROOT exists but is not a git repo (for example the folder the IPDS notebook creates with mkdir), `git fetch`/`reset --hard` fail silently (no check=True) and the notebook continues with stale or absent code. `reset --hard` also discards uncommitted edits in Drive.
- Colab cell 6: `tar.extractall` without `filter=` (deprecated on 3.12+, unsafe on untrusted archives). Download fallbacks write stub records (for example phishing.example.com) into the training data with only a print.
- No notebook seeds torch or numpy (plan doc says "seed 42"). Only `random_state=42` in train_test_split (RoBERTa cell 8), so runs are not reproducible.
- RoBERTa cell 18 (IG demo): `torch.tensor(hf, ...)` on a tensor and `torch.tensor([[test_t1_conf[i]]])` on a tensor element produce UserWarnings. Only ImportError is caught, so any other captum error aborts the cell.
- RoBERTa cell 16: `classification_report(..., target_names=[3 names])` without `labels=[0,1,2]` raises ValueError if a class is absent from both targets and predictions.
- NETRA_IPDS_Training cells 6/8: pipeline/training failures print errors but execution continues, and cell 10 then loads stale artifacts. Cell 10 adds `axhline(..., label=...)` after `legend()`, so those labels never show, and `set_xticklabels` is called without `set_xticks`. It also plots the uncalibrated tier1_model.pkl while the API prefers tier1_model_calibrated.pkl.

### L9. Duplicate dependency lines and unpinned ML libs
- requirements.txt lists `httpx==0.27.2` and later `httpx>=0.27.0`. torch and transformers are unpinned, while the Colab notebook pins transformers==4.40.0, so training and serving versions differ.

### L10. .gitignore gaps
- No entries for `.env`/`.env.*`, `venv/`/`.venv/`, `.ipynb_checkpoints/`, `*.safetensors`/`*.onnx`/`*.ckpt`, `*.zip`/`*.tar*`, `.claude/settings.local.json`, or `ml/evaluation/*.png`. `git check-ignore` confirmed `.env`, `.venv/x`, `notebooks/.ipynb_checkpoints/x` and `ml/models/a.safetensors` are NOT ignored.
- `*.pt`/`*.bin` are only ignored under ml/models/, so a checkpoint saved elsewhere (for example a Colab download placed at the repo root) would be committed.
- No secrets and no large binaries in tracked content. The scan for API keys and HF/GitHub/AWS tokens over tracked files found nothing, and the largest tracked file is a 2.1 MB tokenizer.json. 205 vendored third-party skill files are tracked under .agents/ (bloat only).

### L11. skills-lock.json is out of sync with .agents/skills
- 29 entries in the lock, 31 directories in .agents/skills. `git-commit` and `multi-stage-dockerfile` are installed (git-commit is tracked) but have no lock entry and so no integrity hash.
- Fix: re-run the skills installer so the lock covers every vendored skill, or remove the unlocked directories.

---------------------------------------------------------------------------

## Checked, no issue found
- No hard-coded secrets in tracked non-.agents files; weights, pkl and data/ are ignored.
- DistilBERT notebooks: architecture (768+32 -> 256 -> 2), `header_proj` name and tokenizer settings match api/main.py's PhishingClassifier. Train/val are disjoint, class weights come from train labels only, and `scaler.unscale_` precedes clipping in the Local notebook.
- RoBERTa notebook: architecture (768+32+1 -> 256 -> 3) matches api/tier2_service/model.py; train/val/test use different split labels; clip-after-unscale is correct.
- Release tags in CLAUDE.md (v2.0.0, v3.0.0) match api/main.py:231 and api/tier2_service/model.py:67.
