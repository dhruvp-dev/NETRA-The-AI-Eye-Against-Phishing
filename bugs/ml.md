# ML bug hunt: NETRA/ml (read-only review, no project files modified)

Scope: data_pipeline.py, train.py, train_distilbert.py, train_roberta_tier2.py, generate_tier2_dataset.py,
calibrate_threshold.py, calibrate_tier2_threshold.py, features/*, evaluation/tier2_evaluation.py, ml/models/*.json,
cross-checked against api/main.py and api/tier2_service/{main,model}.py.
Note: data/processed/ is empty, so data-dependent claims are from code/config only. Items were verified by reading code
(and, where noted, by running Python).

## High

1. **High | ml/train_distilbert.py:68-76,171-175 (also generate_tier2_dataset.py:47-52,211-214, train_roberta_tier2.py:64-69,157-160, evaluation/tier2_evaluation.py:63-72)**
   Header features are never populated in training. The code looks for columns `spf_pass ... sender_domain_match` in unified.csv, but data_pipeline.py only writes a JSON `headers_available` column. The missing columns are filled with 0, so every training row has an all-zero header vector. At inference api/main.py:545 (and tier2_service) feeds real values from `header_features.extract`. For example `{}` gives `spf_none=dkim_none=dmarc_none=1, match=1`.
   Because the inputs were always 0, `header_projection` weights got zero gradient and stay at random init. At inference they inject random noise from out-of-distribution inputs. Also, the name `sender_domain_match` differs from `header_features.HEADER_FEATURE_NAMES[-1] = sender_domain_matches_reply_to`.
   Fix: in `load_and_prepare_data` (and in generate_tier2 / roberta / eval) build the 10 features with `header_features.extract(json.loads(headers_available), sender, reply_to)` and use `header_features.HEADER_FEATURE_NAMES`. Retrain Tier-1 and Tier-2.

2. **High | ml/data_pipeline.py:390,422 vs api/main.py:535-536; ml/train_distilbert.py:179**
   Label leakage and train/inference text mismatch. PhishTank/OpenPhish records have `body_text = "Phishing URL: <url>"` and label 1, and that string occurs only in phishing rows. DistilBERT is trained on `body_text` only, with no subject and no `Body:` prefix. At inference `_predict_distilbert` builds `Subject: ...\nBody: ...\nPhishing URL: <u>` and appends `Phishing URL:` for every URL in every email, including legitimate ones. The model has learned that this literal prefix means phishing. This probably explains the suspicious val numbers (F1 0.9993, PR-AUC 1.0 in distilbert_config.json) and pushes legit emails with links toward PHISHING/SUSPICIOUS.
   Fix: do not prefix URL-only records with a label-revealing string. Use one shared `build_text(subject, body, urls)` function (neutral `URL:` prefix) in training, generate_tier2, and api/main.py. Drop or down-weight URL-only sources, which are not emails.

3. **High | ml/train_roberta_tier2.py:278**
   `torch.cuda.get_device_properties(0).total_mem` is not a valid attribute (it is `total_memory`). This raises AttributeError at the start of every CUDA run, the documented use case (RTX 5050 / Colab), before any training. Fix: use `.total_memory`. (Not run: torch is not installed in this environment.)

4. **High | ml/generate_tier2_dataset.py:292-300,263-265 with train_roberta_tier2.py:152,339**
   Tier-2 label leakage via `tier1_confidence`. Class 1 (SUSPICIOUS) is defined as "non-phishing and Tier-1 verdict == SUSPICIOUS". The verdict is a deterministic function of the Tier-1 score and confidence (`confidence < 0.7 or 0.08 <= p < 0.25`). Tier-2 receives `tier1_confidence` as an input feature, so it can read the label off that feature. Separately, Tier-1 is run on train rows it was trained on, so those confidences are overfit (near 1.0). Train and val/production distributions differ.
   The committed metrics show the consequence: SUSPICIOUS F1 = 0 (support 5) and macro-F1 0.39 (roberta_tier2_metrics.json).
   Fix: derive class 1 from something independent of the input features, or drop `tier1_confidence` as a feature. Generate Tier-1 scores out-of-fold or only from held-out splits.

## Medium

5. **Med | ml/data_pipeline.py:176,183,151-152 (readers at 198-215, 260-286)**
   `msg.get("subject")` returns an `email.header.Header` object, not a str, when the header has raw 8-bit bytes (confirmed by running Python on `Subject: Caf\xe9`). `extract_urls(subject)` and `subject.strip()` then raise TypeError.
   - `read_spamassassin_eml_folder` silently drops those emails (`except Exception`, debug log only). Spam with non-ASCII subjects is lost, which biases the data.
   - `read_enron_mbox`, `read_nazario_mbox`, and `read_spamassassin_mbox` wrap the whole loop in one try. The first bad message aborts the file and discards all remaining records.
   Fix: `subject = str(email.header.make_header(email.header.decode_header(raw)))` with a try/except. Do the same for from/reply-to, and put try/except inside the per-message loop.

6. **Med | ml/features/url_features.py:79-88 (verified by running)**
   `check_typosquatting` treats any exact match (distance 0) of any candidate label as "legitimate brand". Candidates include subdomain labels and hyphen-split pieces. So `paypal.evil.com`, `paypal-secure.tk`, `secure-paypal-login.com`, and `apple.com.evil.ru` all return `(0, 0.0)` and are NOT flagged. Those are the classic brand-impersonation patterns. Fix: only treat as legitimate when the registrable domain (the last two labels) equals the brand (`brand.com`). Flag brand tokens appearing in any other position. Use a public-suffix list.

7. **Med | ml/calibrate_threshold.py:31,94-105,137-152 vs api/main.py:45-48,272**
   This script calibrates the uncalibrated RF (`tier1_model.pkl`), but the API prefers `tier1_model_calibrated.pkl` when present, so RF thresholds do not match the RF served. More importantly, it writes `threshold_config.json`, which the API applies to DistilBERT scores (`phishing_threshold`, `suspicious_lower`). Running `python ml/calibrate_threshold.py` (the documented recalibration command in CLAUDE.md) overwrites the hand-tuned 0.25/0.08 DistilBERT corridor with RF-derived values, with `suspicious_lower = 0.6 * threshold` and method "pr_curve_f1".
   Fix: calibrate the model that is served. Separate RF and DistilBERT threshold files, or have the script calibrate DistilBERT val probabilities.

8. **Med | api/main.py:469-486 (RF path) vs ml/train.py:94-160, features/text_features.py**
   RF train/inference skew. (a) TF-IDF and handcrafted features are trained on `subject + " " + body`, but `_predict_rf` hard-codes `"subject": ""` and ignores `req.subject`. (b) URL features are trained on all URLs regex-extracted from body and subject. `_predict_rf` uses only `req.urls` and does not auto-extract from the body, unlike `_extract_signals`. (c) `www.` URLs without a scheme give hostname None, so `domain_length=0` and `uses_https=False`, which wrongly sets `any_http_url`. Fix: pass subject, auto-extract URLs the same way as the pipeline, and normalise scheme-less URLs.

9. **Med | ml/data_pipeline.py:440-483**
   Split issues. (a) The docstring and the log say "stratified random split", but it is a plain shuffle with no stratification by label or source. (b) The "cross-source" holdout is the smallest source. With PhishTank/OpenPhish present this is likely a single-class (all-phishing) set. FPR and precision are undefined on it, and that source is entirely absent from training. (c) The `test_std`, `test_ext`, and `test_cross` splits are never used by any script (grep found no consumers). Every reported metric (distilbert_config.json, tier2 metrics) is on `val`, which is also used for checkpoint selection. Fix: stratify by (label, source), pick the cross-source holdout so it contains both classes, and add a test-evaluation script.

10. **Med | ml/evaluation/tier2_evaluation.py:59,82-92,171-205 and train_roberta_tier2.py:454-457**
    "Standalone evaluation" runs on `split == "val"`, the same split used for early stopping and checkpoint selection, so it is optimistic and not a test. It builds text without the `URL:` lines that training (train_roberta_tier2.py:173-178) and the service (tier2_service/main.py:170-173) use. `evaluate_combined_system` always returns None, and the comparison table prints hard-coded Tier-1 numbers ("99.92%"). The training script writes metrics to `ml/models/roberta_tier2_metrics.json` while the committed file lives in `ml/evaluation/` with a different schema (`per_class`, `test_accuracy`), so the committed file was not produced by this script. Fix: evaluate on a held-out test split with identical text construction, and align the output path and schema.

11. **Med | ml/calibrate_tier2_threshold.py:63-99,119-140**
    (a) `total = len(df)` (all unified records, all splits) is the denominator for escalation rate, but `t2` contains only already-escalated rows, so the rate is wrong. (b) "phishing_recall" is the share of phishing inside `t2` captured by the window. It is monotonic in window width, so the sweep just picks the widest window allowed by the 30% penalty and measures nothing about Tier-1 recall. (c) It uses all splits including val and test. (d) It writes `tier2_config.json` with an `escalation_policy` schema. The committed `tier2_config.json` is a byte-identical copy of `roberta_tier2_config.json` (`thresholds`, `metrics`) with no `escalation_policy`. So `api/main.should_escalate` always falls back to defaults, and `lower`/`upper` are read there but never used. Fix: compute escalation rate over the full evaluation split with Tier-1 scores for all rows, optimise on val only, and keep model config and escalation policy in separate files.

12. **Med | ml/generate_tier2_dataset.py:243-250,104 vs api/main.py:557-565**
    The Tier-1 verdict logic hard-codes `pt = 0.25` and `0.08`. It does not read `threshold_config.json`, so it silently diverges after any recalibration (`/admin/recalibrate`, calibrate_threshold.py). `load_state_dict(..., strict=False)` (line 104) hides any key mismatch, so a failed load leaves random weights with no error. Fix: load thresholds from config and use `strict=True` after renaming keys. The tier1 text is also body-only here, while serving uses a `Subject/Body/URL` prompt (see item 2).

13. **Med | ml/train.py:231-247 with features/text_features.py:70-89; ml/models/training_metrics.json**
    Unscaled features: `hc_text_length` (up to 1e5+) sits next to TF-IDF values in [0,1] and goes into LogisticRegression(saga) and LinearSVC. training_metrics.json shows LR collapsed (precision 0, recall 0, accuracy 0.7643). RF is unaffected, so this is only latent, but the LR candidate is useless. Fix: scale or log1p the handcrafted columns (fit on train only). Also SMOTE (line 330) is combined with `class_weight="balanced"`, and the broad `except` silently skips SMOTE (e.g. if `sampling_strategy=0.3` is below the current minority ratio). Fix: log loudly or fail.

## Low

14. **Low | ml/data_pipeline.py:97-100 vs 104-113 and header_features.py:30-33**
    `parse_auth_results` maps `softfail`/`neutral`/`temperror` to "none", but `parse_spf_header` maps `softfail` to "fail". Results differ depending on which header exists. At inference `_extract_signals` treats `softfail` as a failure while `_one_hot_auth` maps it to "none". Fix: one shared normaliser (`softfail` -> `fail`).

15. **Low | ml/data_pipeline.py:158-165**
    For multipart mail, both `text/plain` and `text/html` parts are concatenated, so content is duplicated and raw HTML tags are fed to the models. Inference receives plain text from the extension. `safe_decode` ignores the declared charset, and `cp1252` is unreachable because latin-1 never fails. Fix: prefer text/plain, strip HTML, use `part.get_content_charset()`.

16. **Low | ml/data_pipeline.py:581,586 (SpamAssassin spam)**
    SpamAssassin spam is labelled phishing (`label=1`). This is a label-semantics mismatch (spam is not phishing) and makes "phishing" partly a "style of source" classifier (Enron = all legit). Fix: document it or keep a separate `spam` label and source-aware evaluation.

17. **Low | ml/data_pipeline.py:608,640-650**
    `drop_duplicates(subset=["_body_hash"])` keeps the first record even when identical bodies carry different labels (ham vs spam). Cross-label duplicates should be removed or flagged. The `--output` and `--raw-dir` flags are applied after `PROCESSED_DIR.mkdir` ran at import, so a custom output directory is never created and `to_csv` fails. A missing `RAW_DIR` raises in `discover_files`.

18. **Low | ml/train_distilbert.py:265-300,349-352,363 and train_roberta_tier2.py:271-272**
    No seeds in train_distilbert.py (no `torch.manual_seed` / `np.random.seed`), so runs are not reproducible. If val F1 never exceeds 0.0, no checkpoint is saved and the later `torch.load(best_model_path)` crashes with FileNotFoundError. The scripts have no `--epochs/--batch_size/--fp16` flags (train_distilbert only has `--data`), but CLAUDE.md documents `python ml/train_distilbert.py --epochs 3 --batch_size 16 --fp16` and the same for `train_roberta_tier2.py`, so argparse errors out. The device falls back to CPU only; MPS from CLAUDE.md is not handled.

19. **Low | ml/train_roberta_tier2.py:352-360,152**
    `scheduler.step()` runs even when `GradScaler` skips `optimizer.step()` (inf grads), which gives a warning and slight LR drift. A NaN `tier1_confidence` is filled with 0.5, a value that does not mean "missing".

20. **Low | ml/features/url_features.py:79-88,96-100**
    Over-flagging and wrong depth. `stream.com` returns flag 1 (distance 0.2 to "steam", verified). `_subdomain_depth` has no public-suffix list, so `example.co.uk` gets depth 1. Fix: add a PSL (`tldextract`) and an allow-list.

21. **Low | api/main.py:535-538,576-581 (consumers of ml features, flagged for completeness)**
    `for u in set(all_urls)` gives non-deterministic URL order and duplicates (req.urls plus regex hits), so truncated 256-token input varies between processes. The `elif cls == "LEGITIMATE" and high_threat_signals >= 3` branch is unreachable because `>= 2` matches first.

22. **Low | ml/train.py:187,212,350,357 (and calibrate scripts)**
    The source file contains mojibake (`â†`, `âœ“`) inside log strings, which print as garbage on Windows consoles. `calibrate_threshold.sweep_thresholds` recomputes a full confusion matrix per threshold (O(n_thresholds * n_val)), which is very slow on a large val set. Fix: use cumulative sums.

23. **Low | features/text_features.py:25, api/main.py:63, generate_tier2_dataset.py:130**
    Three different urgency keyword sets. generate_tier2's `extract_signals` claims to mirror api/main.py but does not (the signal is informational only today). Fix: one shared constant.

## Config observations (not code bugs, but inconsistencies)
- ml/models/distilbert_config.json has no threshold keys, though CLAUDE.md says thresholds live there; they actually live in threshold_config.json.
- ml/models/tier2_config.json and roberta_tier2_config.json are identical (`thresholds` 0.35/0.65 and `metrics`) and neither matches what train_roberta_tier2.py (`model`, `final_metrics`, `confidence_threshold`) or calibrate_tier2_threshold.py (`escalation_policy`) would write.
- roberta_tier2_config.json `thresholds` (0.35/0.65) are unused: tier2_service/model.py uses argmax.
- The committed roberta_tier2_metrics.json shows SUSPICIOUS support 5 and F1 0, PHISHING recall 0.118 (2/17): the Tier-2 model is effectively a LEGITIMATE classifier.
