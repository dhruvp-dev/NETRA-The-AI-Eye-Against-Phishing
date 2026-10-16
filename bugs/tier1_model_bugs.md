# Tier-1 model audit: does it separate legitimate from phishing?

Date: 2026-10-17. Checkpoint: `ml/models/distilbert_tier1.pt` (sha256 `653ebaf3...` matches `distilbert_config.json`),
`text_format=v3`, `header_features_trained=true`. Live thresholds: `ml/models/threshold_config.json`.
No project code was changed. The old reports in `bugs/` (api / ml / notebooks_tests_docs, mostly about the dropped Tier-2) were removed
and replaced by this file; they are still in git history.

## Verdict

**No. The model does not reliably separate legitimate email from phishing on real-world mail, and the thresholds are
not the main cause.** The threshold is broken too (B2), but fixing it alone would not change the result:
the model's scores are almost all near 0 or near 1.

| Test | Result |
|---|---|
| Held-out benchmark splits (val, test_std, test_cross = SpamAssassin, test_ext), email rows only, 4,368-row stratified sample | AUC 0.998–1.000. Recall 0.97–1.00, FPR 0.0–0.4% at any threshold from 0.0248 to 0.5 |
| `python tests/stress_test_tier1.py` | **FAILED**: legitimate e-commerce confirmation scored 0.8856, verdict PHISHING |
| 20 hand-written ordinary legitimate emails (Amazon, Walmart, Netflix, Stripe, Spotify, Google, LinkedIn, Chase, calendar invite, newsletter...) | **11/20 flagged PHISHING at the default 0.25 threshold, 12/20 at the live 0.0248**. Scores 0.87–0.9999 |
| 15 hand-written phishing emails | 13/15 caught at 0.25 (14/15 at 0.0248). Missed: a fake invoice with a bit.ly link (0.035), a no-link CEO/wire-transfer BEC (0.0) |
| Same phishing text placed after 40 benign sentences | 0.9997 alone, **0.0004** when padded |
| `pytest tests/` | 108 passed, 1 skipped (the tests do not cover any of the below) |

The benchmark numbers are high because the benchmark separates *corpora*, not *intent* (B1). The reported
99.9% F1 in `distilbert_config.json` describes that, not real-world accuracy.

---

## Critical

### B1. Training data teaches "has a URL / looks like a brand or automated message" = phishing
- Where: `data/processed/unified.csv`, `ml/data_pipeline.py`, `ml/train_distilbert.py`.
- Train split: **every legitimate row is Enron** (22,108 rows: internal person-to-person mail, only 15% contain a URL). Phishing is
  PhishTank + OpenPhish (6,091 rows, **100% have a URL and an empty body, no sender**) plus Nazario (63% have a URL).
  So 82% of phishing training rows are just a URL, and URL presence is a near-perfect shortcut.
  There are no legitimate transactional, marketing, notification, calendar or bank-statement emails in training.
- Observed: any legitimate email that is formal, brand-sent or contains a link scores about 1.0.

  | Legit email | Score |
  |---|---|
  | Google "Security alert" (with myaccount.google.com link) | 0.9998 |
  | Chase "Your statement is ready" | 0.9999 |
  | Weekly newsletter (healthline) | 0.9997 |
  | Spotify welcome | 0.9996 |
  | LinkedIn reset password | 0.9996 |
  | Amazon shipped | 0.9973 |
  | Netflix invoice | 0.9955 |
  | Stripe payment received | 0.9918 |
  | Target order placed (stress test case 1) | 0.8856 |
  | Uber Eats order / Etsy order confirmation | 0.8688 / 0.9436 |
  | Google Calendar invite with Meet link | 0.9983 |

  Plain person-to-person mail scores about 0.0, as in Enron.
- Why the benchmark hides it: val/test_std legit = Enron again; test_cross legit = SpamAssassin ham, which is also mostly
  mailing-list/person mail. Phishing and legit differ by corpus style, so the model learns corpus style.
- Fix: add real legitimate automated/transactional/marketing mail (e.g. a held-out personal inbox export, newsletters, bank/shop
  notifications, other ham corpora with URLs) to train/val/test; balance URL presence across classes; add a
  "with URL" slice and a "brand/transactional" slice to the reported metrics. Retrain, then re-run `stress_test_tier1.py`.

### B2. Live threshold file is degenerate; SUSPICIOUS band is 0.0002 wide
- Where: `ml/models/threshold_config.json` (gitignored, written by `ml/calibrate_threshold.py` at 2026-10-16 16:17).
- Values: `phishing_threshold=0.02480626`, `suspicious_lower=0.02456119`. The band `[0.02456, 0.02481)` is 0.00025 wide. In 4,368 scored rows,
  **0 landed inside it**. SUSPICIOUS is therefore unreachable from the model score; it only appears through the signal-override floor
  (the stress test "Ultra-Short Body" case got SUSPICIOUS only because the override floors the score to `suspicious_lower` = 0.0246).
  The documented defaults are 0.08 / 0.25 (`threshold_config.default.json`, CLAUDE.md).
- Cause (`ml/calibrate_threshold.py:83-95` `pick_thresholds`): the model is saturated, so the F1 curve is flat from about 0.02 to 0.99 and
  `idxmax` picks an arbitrary point on the plateau. `suspicious_lower` is "lowest threshold with recall >= target below the
  phishing threshold", which lands 0.0002 away. `_valid_thresholds` (`api/main.py:248`) accepts any `0 < lower < phishing`, so no guard rejects it.
  Calibration also uses only the in-distribution `val` split (Enron vs Nazario), so it cannot see B1.
- Effect: this threshold is not what causes the false positives (they score 0.87+). It does flip borderline cases the wrong way in both directions:
  the fake invoice (0.035) is PHISHING at 0.0248 but LEGITIMATE at 0.25, while a legitimate Walmart order mail (0.098) is PHISHING at 0.0248 but SUSPICIOUS at 0.08/0.25.
  The effective classifier is binary; "SUSPICIOUS" does not work as a gray zone.
- Fix: enforce a minimum band width in `pick_thresholds` and in `_valid_thresholds`/`/admin/recalibrate`; calibrate on a mixed set that
  includes legitimate automated mail (B1) and cross-corpus data; consider probability calibration (temperature scaling) so scores are not saturated.
  Do not rely on max-F1 alone.

### B3. Padding evasion: anything after about 256 tokens is ignored; URLs are appended last so they are dropped first
- Where: `api/main.py:569-582` (`max_length=256`, head truncation) and `ml/features/text_builder.py` (`Subject` + `Body` + `URL:` lines in that order, URLs last).
- Observed: the same phishing paragraph scores 0.9997 alone and **0.0004** after 40 benign sentences; passing the malicious URL
  explicitly in `urls` does not help (0.0004), because `URL:` lines come after the body and are truncated. Any real phishing email with a long
  benign preamble or a long quoted thread, and any long email in which the link is near the end, bypasses the model.
- The stress test's 50,000-char "Massive Body" case "passes" (0.0011, LEGITIMATE) for the same reason: nothing after the first 256 tokens is read.
- Defense in depth does not cover it: the override needs >= 2 signals, and benign-looking text with one bad link yields at most 1.
- Fix: put URLs (and a capped list of distinct domains) before the body, or use head+tail / chunked scoring with max over chunks; test with padded phishing in the stress suite.

---

## High

### B4. Header features are dead: all auth values are "none" in all 44,497 training rows
- Where: `data/processed/unified.csv`, `ml/models/distilbert_config.json` (`header_features_trained: true`).
- `spf_pass`=1 in 1 row; `spf_fail` never; `dkim_none` and `dmarc_none` = 1 in every row; `sender_domain_matches_reply_to`=0 in 832 rows (1.9%).
  The corpora carry no SPF/DKIM/DMARC results. So the header branch can learn nothing about authentication, though config says it was trained.
- Observed: changing headers from `{}` to pass/pass/pass or fail/fail/fail moves the score by about 0.001–0.01
  (e.g. 0.8779 / 0.8856 / 0.8823). A genuinely failing-auth email gets no model credit; the only use of auth is the
  override (`header_auth_failed` counts as one signal).
- Docs say the model uses "10 SPF / DKIM / DMARC / sender-match features"; in effect it uses none.
- Fix: either build training data that has real auth results (e.g. captured `Authentication-Results`) or drop the branch and
  say so in CLAUDE.md/README; correct `header_features_trained`.

### B5. Missed phishing with no obvious link or brand wording
- "Re: wire transfer" CFO/BEC email: score 0.0. Fake invoice with a `bit.ly` link: 0.035 (only caught by the abnormal 0.0248 threshold).
  CEO gift-card scam scored 0.994 (caught). BEC and short-link phishing are underrepresented in training (no BEC corpus; PhishTank has no body text).
- Combined with B1, the model separates "formal/link" from "informal/no link", not "malicious" from "benign".
- Fix: add BEC/short-link/invoice-lure data; evaluate per-category recall.

### B6. Reported metrics overstate quality; stress test is red but documented as passing
- `distilbert_config.json` shows F1 0.99+, FPR 0.4% (val/test_std). `test_std` legit is Enron-only; URL-only rows are excluded from email
  metrics; test_cross is mostly list mail. None measure transactional/brand mail (B1) or long/padded mail (B3).
- `python tests/stress_test_tier1.py` currently exits 1 (item 1 of 11). README/CLAUDE.md should not describe the model as validated until this is fixed.

---

## Medium

### B7. Score saturation: probabilities are not calibrated
- Legit scores cluster at 0.0000-0.0001, phishing at 0.996-0.9999, and false positives also sit at 0.87-0.9999. In the 4,368-row sample the score is
  essentially 0/1. `risk_score`, `confidence = max(proba)` (`api/main.py:569-582`) and `risk_level` (CRITICAL at >= 0.5)
  carry no usable uncertainty. The extension shows a "risk score" that is effectively a class flag.
- Fix: temperature scaling on a mixed validation set (B1); label-smoothing or fewer epochs; report ECE.

### B8. Signal override raises severity from weak, common signals
- `_extract_signals` (`api/main.py:467-491`): `suspicious_urls` counts `any_http_url` (a plain `http://` link) and `any_phishing_tld`, so a
  harmless link already gives 1 point; `_signal_count` adds `min(suspicious_urls, 2)`. Two such points or one plus a typosquat/auth-fail make
  a model-LEGITIMATE email SUSPICIOUS, and 3 points PHISHING (floored at `phishing_threshold`). With a tiny band (B2) the floored score
  sits on the boundary. `urgency_detected` is returned but not counted at all, contrary to the multi-signal idea in CLAUDE.md
  ("threat signals").
- Checked: a legitimate email with one `http://` link gives `suspicious_urls=1`, count 1 (no override). Two plain-http links to different
  hosts give count 2 → SUSPICIOUS, likely noisy on real newsletters.

### B9. Subject sensitivity: one line can flip the verdict
- "Your Target.com order #84921 has been placed" (subject) → 0.8779-0.8856; the same body with an empty subject → 0.0009.
  The model keys on subject phrasing seen in phishing lures ("order", "account", "verify", "has been placed") rather than on context.
  Part of B1; listed separately because it is easy to reproduce and shows in the stress test.

---

## Low

### B10. Config advertises training that did not happen
`header_features_trained: true` (B4); `text_format: v3` is fine. Also `confidence_threshold: 0.7` in `distilbert_config.json` is not read by
`api/main.py` any more (dead key).

### B11. Stale / unused files and edge handling
- `ml/models/roberta_tier2.pt` (499 MB) and `roberta_tier2_tokenizer/` are still on disk although Tier-2 was dropped (untracked, gitignored).
- `/predict` returns 422 for an email with empty body but a subject/URL (HTML-only or image-only mail), so the extension cannot score it
  (`api/main.py:656`). Scoring an empty body with a subject gave 0.39 in the raw model, so this guard hides a poorly defined input
  instead of handling it.
- `PredictRequest` limits (50 URLs, 512-char addresses, 2,048-char URLs) reject real emails with 422 instead of truncating
  (hit while scoring SpamAssassin ham: "List should have at most 50 items", "String should have at most 512 characters").
  The extension should truncate before sending, or the API should accept and clip.
- `threshold_config.json` is gitignored: the broken 0.0248 value exists only on this machine. A fresh checkout/deploy silently uses the 0.08/0.25 defaults,
  so behaviour differs between environments. Document which file is authoritative.

---

## Suggested order of work
1. B1 + B6: build a representative legitimate set (transactional, marketing, notifications, bank, calendar) and a held-out realistic
   test; add it to `stress_test_tier1.py` (at least the 20 legit / 15 phish probes used here).
2. B3: reorder `build_model_text` (URLs before body) or chunk long bodies; retrain.
3. Retrain, then B7 calibration and B2 threshold selection on the new mixed validation set, with a minimum SUSPICIOUS width.
4. B4: real auth data or remove the header branch.
5. B8/B5/B9/B11 clean-ups.

## Reproduction
- Interpreter with torch/fastapi: `C:\Users\Ramanan\AppData\Local\Programs\Python\Python314\python.exe` (the default `python` on PATH lacks them).
- Held-out scoring and the probe scripts were written in the session scratchpad (not in the repo). The probe emails are listed in B1/B3 and can be sent to
  `/predict` directly.

---

## Status after the serving-side fixes (2026-10-17, no retraining)

The weights and the model input format are unchanged; short emails get exactly the same model score as before.
Everything below works around the model. It does not fix it.

| Bug | Status | What changed |
|---|---|---|
| B1 brand/transactional false positives | **Mitigated** | `api/trust.py`: a known brand sender (curated `KNOWN_ORGS`) or an authenticated sender, with every link on its own / known domains, has its verdict lowered (`adjustment: trusted_sender`). Unknown senders still get the biased model verdict. |
| B2 degenerate thresholds | **Fixed** | Live file reset to 0.08 / 0.25 (old file in `threshold_config.audit_backup.json`). `MIN_SUSPICIOUS_BAND = 0.03` in the API and calibration; calibration refuses a flat F1 plateau unless `--force`. |
| B3 padding evasion | **Mitigated** | Head window + up to 4 body windows of 400 chars, subject and URL lines repeated, max score. Phishing placed in the middle of a very long email can still fall between windows. |
| B4 header branch dead | **Documented** | `header_auth_learned: false`; `/explain` no longer reports header attributions. Auth is used by the rules only. |
| B5 BEC / short-link misses | **Mitigated** | `bec_detected` (payment/credential request + pressure, 2 units), shortener, IP link and program-file link signals. |
| B6 stress test red, metrics overstated | **Fixed / documented** | Stress suite has 46 cases (core, 20 legit, 15 phishing incl. padded) and passes. README has a Known limitations section. |
| B7 saturation | **Open** | Needs a mixed validation set to fit temperature scaling. |
| B8 weak signals counted | **Fixed** | Plain `http://` is no longer counted; urgency counts only next to another unit. |
| B9 subject sensitivity | **Mitigated** (via B1 rule for trusted senders) | |
| B10 config | **Fixed** | Dead `confidence_threshold` removed from `distilbert_config.json`; `known_limitations` added. |
| B11 input handling | **Fixed** | Empty body accepted with subject or URLs; URL list, long URLs, addresses and subject are clipped instead of 422. |

Still required: retrain on representative legitimate mail (B1), real auth data (B4), calibration (B7).
