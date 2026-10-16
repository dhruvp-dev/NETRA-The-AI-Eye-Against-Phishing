# NETRA Local Training Plan (RTX 3060 12 GB, Windows)

> **First step for the implementer:** copy this file to the repo root as `LOCAL_TRAINING_PLAN.md`.
> Work on a new branch (`feat/local-training`). Commit locally, do not push.
> Python: `C:\Users\Ramanan\AppData\Local\Programs\Python\Python314\python.exe` (has torch, transformers, sklearn).

## Context

The bug-fix work is merged. The code now trains without label leakage. The local notebook
(`notebooks/NETRA_DistilBERT_Training_Local.ipynb`) still has problems that would give misleading or
poor results on a local 3060:

- **Data imbalance.** Legitimate: ~517k Enron + 2,752 SpamAssassin ham. Malicious *emails*: 2,279
  Nazario + 1,899 SpamAssassin spam. Plus 73,855 PhishTank and 300 OpenPhish **URL-only** rows. Without
  caps, Enron dominates training time, and URL-only rows dominate the malicious class. The model would then learn
  "URL-only line = phishing" instead of phishing content.
- **Model selection and calibration.** Both include URL-only rows, but the API only ever sees emails.
- **Near-duplicates.** Phishing campaigns send near-identical emails. Exact-hash dedup leaves them,
  so they leak across train and test, and test metrics come out inflated.
- **Notebook flow.**
  - It silently reuses a stale `unified.csv` built by the old pipeline.
  - The pipeline cell shows no output for long runs.
  - There are no calibration, stress-test or data-sanity cells.
  - The Enron cell is Colab-only, and the markdown is outdated (the 0.70 SUSPICIOUS rule, section numbers jump from 7 to 10).
  - Training writes straight over the shipped weights, with no backup.
- **Training script.**
  - Loads the CSV 4 times.
  - No bf16 (Ampere supports it), no gradient accumulation, no output-dir option, so smoke runs would overwrite real weights.
  - No per-source test breakdown.

**Outcome:** one notebook run on the 3060 does the following:
1. Builds a balanced, deduplicated dataset.
2. Trains with bf16.
3. Selects the checkpoint and calibrates thresholds on **email** rows.
4. Reports honest held-out metrics by source.
5. Passes `tests/stress_test_tier1.py`.

---

## Should you download more data? Yes, more real phishing EMAILS

Phishing email content is the bottleneck: about 4k malicious emails against hundreds of thousands of legitimate ones.
Do **not** add more legitimate mail; Enron gets capped below. Add the following, then let the new readers (Phase A3) pick them up:

| Dataset | Where to put it | Use |
|---|---|---|
| Kaggle "Phishing Email Dataset" (`naserabdullahalam/phishing-email-dataset`) | `data/raw/phishing_email_dataset/*.csv` | Use `CEAS_08.csv`, `Nigerian_Fraud.csv`, `Ling.csv`. **Skip** `Enron.csv`, `SpamAssasin.csv`, `Nazario.csv`, `phishing_email.csv` (duplicates of what you already have). |
| phishing_pot (`github.com/rf-peixoto/phishing_pot`), recent real phishing `.eml` | `data/raw/phishing_pot/` (any depth) | All `.eml` files, label 1 |
| More Nazario years (`monkey.org/~jose/phishing/`) | `data/raw/` | Rename each to `phishing-YYYY.mbox` (discovery needs `.mbox` and "phishing" in the name) |

- Confirm the file names and columns after download. The listings may have changed. The readers must log and skip unknown files instead of crashing.
- **Defender:** the folder exclusion for `C:\Users\Ramanan\Desktop\NETRA\data\raw` must stay active
  before you download (verify with `Get-MpPreference | Select-Object -ExpandProperty ExclusionPath` in an
  admin PowerShell). The preflight cell (Phase D) reports any file Windows blocks.

---

## Phase A: `ml/data_pipeline.py`

**A1. Per-source caps (memory-safe).**
- Add a repeatable CLI flag `--cap SOURCE=N`.
- Defaults: `enron=30000`, `phishtank=8000`; other sources uncapped.
- For `read_enron_csv`, apply the cap **while reading**, using reservoir sampling (Algorithm R, `random.Random(seed)`, keep `cap * 1.2` for dedup headroom). That way 517k records are never held in memory.
- Do **not** take the first N rows: Enron is ordered by mailbox, so the first N would be biased.
- For other sources, sample after dedup with `df.groupby("source").sample(n=min(cap, len), random_state=seed)`.
- Log the before and after counts per source.

**A2. Near-duplicate hashing.**
- Change `body_hash()` to normalise before hashing: lowercase, replace URLs with `<url>` (use `text_builder.URL_PATTERN`), replace digit runs with `0`, collapse whitespace.
- Hash `subject + "\n" + body` for emails.
- Keep `_url_only_record` hashing the URL itself, so distinct URLs stay distinct.
- The existing conflicting-label drop and `drop_duplicates` then catch near-duplicates too.

**A3. New readers. Reuse the existing helpers (`_safe_record`, `_url_only_record`, `extract_urls`).**
- Generalise `read_spamassassin_eml_folder` into `read_eml_folder(folder, source, label, recursive=False)`.
  - The SpamAssassin readers call it with `recursive=False`.
  - phishing_pot uses `recursive=True` (rglob `*.eml`), `source="phishing_pot"`, `label=1`.
- Add `read_labeled_csv(path, source)` for the Kaggle CSVs:
  - Flexible columns: body from the first of `body`, `text_combined`, `text`; optional `subject`, `sender`, `receiver`/`reply_to`, `urls`. The label comes from the `label` column (1 = malicious).
  - Use `pd.read_csv(..., on_bad_lines="skip", dtype=str)`.
  - Build records with the same keys as `read_nazario_csv`.
  - Source name = file stem lowercased (`ceas_08`, `nigerian_fraud`, `ling`).
- `discover_files()`:
  - Add `labeled_csv` (only files in `data/raw/phishing_email_dataset/` whose stem is in an allow-list `{"CEAS_08","Nigerian_Fraud","Ling"}`; log skipped files).
  - Add `phishing_pot` (folder exists).
  - Nazario discovery already matches `*phishing*.mbox`.
- `add_label_origin()`:
  - Phish sources: `{"nazario","phishtank","openphish","phishing_pot","nigerian_fraud"}`.
  - Every other label-1 row gets `spam`; label 0 rows get `ham`.

**A4. Cross-source holdout override.** Add `--cross-source NAME` to force the holdout source. The default keeps the current `_pick_cross_source` rule.

**A5. Pipeline version marker.**
- Add a `pipeline_version` column (constant `PIPELINE_VERSION = 4`).
- The notebook uses it to detect a stale `unified.csv`.

---

## Phase B: `ml/train_distilbert.py`

- **B1.** Read the CSV **once**. Refactor `load_split_frame(df, splits)` to take a DataFrame, and call `pd.read_csv` once in `train()`.
- **B2.** Add `--precision {fp32,fp16,bf16}`.
  - Default: `bf16` if `torch.cuda.is_bf16_supported()`, else `fp16`, and `fp32` on CPU.
  - bf16 means `torch.autocast(dtype=torch.bfloat16)` and `GradScaler(enabled=False)`.
  - Keep `--fp16` as an alias for `--precision fp16`.
- **B3.** Add `--grad_accum N` (default 1). Divide the loss by N, and step the optimizer and scheduler every N batches. Compute scheduler steps from the effective step count.
- **B4.** Checkpoint selection on **val email rows only** (`~url_only`), using val PR-AUC. Ties go to the higher F1. Log both.
- **B5.** Add `--patience` (default 1). Stop early when val PR-AUC has not improved for that many epochs.
- **B6.** Per-group test metrics.
  - For `test_std` and `test_cross`, email rows only, add `by_source` and `by_label_origin` dicts using `classification_metrics`.
  - Skip groups under 20 rows, and use `pr_auc=None` for single-class groups.
  - Also report `url_only_recall` separately.
  - Store everything under `test_metrics` in the config.
- **B7.** Add `--max_train_rows N` (stratified sample of train for smoke runs) and `--output_dir PATH` (default `ml/models`). `WEIGHTS_PATH`, `TOKENIZER_DIR` and `CONFIG_PATH` derive from it.
- **B8.** Log the token-length p50/p90/p99 of train texts, throughput (samples/s) per epoch, and `torch.cuda.max_memory_allocated()`.
- Keep `num_workers=0` (Windows-safe), the seeding and the existing config keys (`header_features_trained: true`, `text_format: "v3"`, `weights_sha256`).

## Phase C: `ml/calibrate_threshold.py`

- Exclude `url_only` rows from the val set by default. Add `--include-url-only` to opt back in.
- Add `--models_dir` (default `ml/models`) so it can calibrate a smoke-run directory.

---

## Phase D: rewrite `notebooks/NETRA_DistilBERT_Training_Local.ipynb`

Keep the notebook thin. Every heavy step is a streamed subprocess. Reuse the `run()` / `must_run()` helpers
from `notebooks/NETRA_DistilBERT_Training_Kaggle.ipynb` (cell 1). Cells, in order:

0. **Title and instructions** for RTX 3060 12 GB on Windows. Run from VS Code or JupyterLab with the Python 3.14 env above.
1. **Environment check.** Torch version, `cuda.is_available()` (raise if False, with the hint "install the CUDA build of torch"), GPU name, VRAM, `is_bf16_supported()`.
2. **Paths.** Keep the current cell 4 logic (`PROJECT_ROOT`, `DATA_CSV`, `RAW_DIR`, `MODELS_DIR`).
3. **Defender / readability preflight.**
   - Try `open(p, "rb").read(1)` on every file under `data/raw` (sample at most 200 files per folder).
   - Print the blocked files.
   - Raise if `phishing3.mbox` or `emails.csv` is unreadable. The message tells the user to check the exclusion.
4. **Downloads.**
   - Keep the SpamAssassin, Nazario and OpenPhish downloads (they already raise instead of writing stubs).
   - Replace the Colab Enron cell with a check-and-message: "place Kaggle `emails.csv` in data/raw".
   - A markdown cell lists the optional extra datasets and folder names from the table above.
5. **Pipeline.**
   - Rebuild when `unified.csv` is missing, lacks `pipeline_version == 4`, or `REBUILD = True`.
   - Streams output.
   - Passes `--cap enron=30000 --cap phishtank=8000`.
6. **Data sanity.**
   - Rows per split×label, per source, and per label_origin.
   - Assert no `"Phishing URL"` text.
   - Assert train has both labels, and that email-malicious rows ≥ 2,000.
   - Warn when the email ratio legit:malicious exceeds 10:1.
7. **Backup.** Copy the current `ml/models/distilbert_tier1.pt`, `distilbert_config.json`, `distilbert_tokenizer/` and `threshold_config.json` to `ml/models/backup_v2/`, unless the backup already exists.
8. **Smoke run.**
   - Command: `train_distilbert.py --max_train_rows 2000 --epochs 1 --output_dir ml/models/_smoke`
   - Then `calibrate_threshold.py --models_dir ml/models/_smoke`.
   - Must finish in a few minutes. It catches errors before the long run.
9. **Full training.**
   - Command: `train_distilbert.py --epochs 3 --batch_size 32 --grad_accum 1 --lr 2e-5 --max_length 256 --seed 42 --precision bf16 --patience 1`.
   - If you hit out-of-memory: `--batch_size 16 --grad_accum 2`.
10. **Calibrate.** Run `calibrate_threshold.py` (email val rows) and print `threshold_config.json`.
11. **Results.**
    - A table of val and test_std/test_cross metrics, by source and label_origin.
    - Print a **leakage warning** if any test F1 ≥ 0.999 or FPR == 0 on more than 500 legitimate rows.
12. **Stress test gate.** Run `tests/stress_test_tier1.py`, print the exit code, and say clearly whether it passed. If it failed, say how to roll back (copy `backup_v2/` back).
13. **Artifacts and next steps.** List the files and their sizes, and how to start the API.

Remove the outdated "Why DistilBERT / 0.70 SUSPICIOUS" text, and number the sections 1..13.

---

## Phase E: tests (keep `pytest tests/` green)

Add to `tests/test_features.py`, using tiny synthetic files in `tmp_path`:
- **Reservoir cap:** exactly N kept, deterministic for a given seed.
- **Near-duplicate hash:** same email with a different URL or number gives the same hash; different text gives a different hash.
- **`read_labeled_csv`:** flexible columns, labels, source names; unknown file skipped.
- **`read_eml_folder(recursive=True)`:** finds nested `.eml` files.
- **`add_label_origin`:** new sources map correctly.
- **`calibrate_threshold` email-only filter:** URL-only rows excluded.

## Phase F: docs

- `CLAUDE.md` and `README.md`: local training commands (new flags), the extra dataset table, and the Defender note.
- After the real run, replace the stale metrics (README `0.8114`/`0.0010`, DECISIONS `99.93%`) with the new `test_std` email metrics, and name the split.

---

## Verification (acceptance)

1. `pytest tests/` passes, including the new tests.
2. The pipeline log shows per-source counts after the caps, no stale-CSV reuse, and `pipeline_version` 4.
3. The smoke run (cell 8) completes and writes only to `ml/models/_smoke/`. The `ml/models/distilbert_tier1.pt` hash is unchanged.
4. Full run:
   - Peak VRAM under ~11 GB.
   - Epoch time is logged. Rough expectation: well under an hour per epoch with the caps (estimate, not measured).
5. Metrics, email rows:
   - `test_std` F1 ≥ 0.95 and FPR ≤ 0.02 (targets, not guarantees).
   - `test_cross` reported.
   - Any F1 ≥ 0.999 means leakage, so investigate before using the model.
6. `python tests/stress_test_tier1.py` exits 0. The two legitimate emails and the 50k-character email must come out LEGITIMATE.
7. Start the API and re-run the three README samples. Update the README with the real outputs.
8. If 5 or 6 fails, restore `ml/models/backup_v2/` and report the numbers. Don't ship the new weights.
