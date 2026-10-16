"""
NETRA /explain sanity, faithfulness and latency suite (script, not a unit test; needs the weights).

    python tests/xai_sanity.py                           # default p95 budget 20 s per /explain on CPU
    python tests/xai_sanity.py --p95-budget-ms 5000      # e.g. on a GPU host

Checks, on the stress-suite emails (tests/stress_test_tier1.py), without changing any weight on disk:
    1. Deletion: replacing the top-k evidence words with the baseline token moves F toward the
       baseline more than replacing k random content words (mean over seeded random draws).
    2. Model randomisation (Adebayo et al., 2018): re-initialising the classifier head in memory
       must change the word attributions (mean Spearman rho below a bound).
    3. Completeness: median |delta| / |F(x) - F(x')| below a bound.
    4. Latency: p95 of /explain end to end.

Exit code:
    0  all checks passed
    1  at least one check failed
"""

import argparse
import asyncio
import copy
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.stats import spearmanr

from api.main import ExplainRequest, _prepare_distilbert_inputs, explain, load_models, state
from ml.explain import integrated_gradients
from ml.features.text_builder import extract_urls
from tests.stress_test_tier1 import TEST_CASES

K_VALUES = (3, 5, 10)
RANDOM_DRAWS = 20
MAX_RANDOMISATION_RHO = 0.5
MAX_MEDIAN_REL_DELTA = 0.10


def _ig(model, req, n_steps=32):
    urls = extract_urls(req.subject, req.body_text, req.urls)
    text, spans, hv, hb = _prepare_distilbert_inputs(req, urls)
    r = integrated_gradients(model, state.db_tokenizer, text, spans, hv, hb,
                             max_length=int(state.db_config.get("max_length", 256)), n_steps=n_steps)
    return r, text, spans, hv


def _margin_without(text, spans, hv, words):
    """F with the tokens of `words` replaced by the baseline token ([SEP])."""
    import torch
    tok = state.db_tokenizer
    enc = tok(text, max_length=int(state.db_config.get("max_length", 256)), truncation=True,
              return_offsets_mapping=True, return_tensors="pt")
    ids = enc["input_ids"].clone()
    span_of = {(sp.field, sp.index): sp for sp in spans}
    ranges = []
    for w in words:
        sp = span_of[(w.segment, w.url_index)]
        ranges.append((w.start - sp.field_offset + sp.start, w.end - sp.field_offset + sp.start))
    for t, (s, e) in enumerate(enc["offset_mapping"][0].tolist()):
        if e > s and any(lo <= s and e <= hi for lo, hi in ranges):
            ids[0, t] = tok.sep_token_id
    with torch.no_grad():
        logits = state.db_model(ids, enc["attention_mask"], torch.tensor([hv], dtype=torch.float32))
    return (logits[0, 1] - logits[0, 0]).item()


def check_deletion(cases, rng):
    print("\n[Check 1/4] Deletion: top-k evidence vs random words")
    ok = True
    for k in K_VALUES:
        top_moves, rand_moves = [], []
        for name, req in cases:
            r, text, spans, hv = _ig(state.db_model, req)
            direction = np.sign(r.f_input - r.f_baseline) or 1.0
            words = [w for w in r.words if any(ch.isalnum() for ch in w.text)]
            if len(words) <= k:
                continue
            top = sorted(words, key=lambda w: -direction * w.score)[:k]
            # Movement toward the baseline, in the direction the evidence pushed.
            top_moves.append(direction * (r.f_input - _margin_without(text, spans, hv, top)))
            draws = [direction * (r.f_input - _margin_without(text, spans, hv, rng.sample(words, k)))
                     for _ in range(RANDOM_DRAWS)]
            rand_moves.append(float(np.mean(draws)))
        passed = np.mean(top_moves) > np.mean(rand_moves)
        ok &= bool(passed)
        print(f"  k={k:2d}: mean move top-k {np.mean(top_moves):6.2f} vs random {np.mean(rand_moves):6.2f} "
              f"logits over {len(top_moves)} emails -> {'PASS' if passed else 'FAIL'}")
    return ok


def check_randomisation(cases):
    import torch
    print("\n[Check 2/4] Model randomisation (classifier head re-initialised in memory)")
    rand_model = copy.deepcopy(state.db_model)
    torch.manual_seed(0)
    for layer in rand_model.classifier:
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    rand_model.header_proj.reset_parameters()
    for p in rand_model.parameters():
        p.requires_grad_(False)

    rhos = []
    for name, req in cases:
        a, *_ = _ig(state.db_model, req, n_steps=16)
        b, *_ = _ig(rand_model, req, n_steps=16)
        if len(a.words) >= 5:
            rhos.append(spearmanr([w.score for w in a.words], [w.score for w in b.words]).statistic)
    mean_rho = float(np.nanmean(rhos))
    passed = mean_rho < MAX_RANDOMISATION_RHO
    print(f"  mean Spearman rho {mean_rho:.3f} over {len(rhos)} emails (bound {MAX_RANDOMISATION_RHO}) "
          f"-> {'PASS' if passed else 'FAIL'}")
    return passed


def check_completeness(cases):
    print("\n[Check 3/4] Completeness")
    rels = []
    for name, req in cases:
        r, *_ = _ig(state.db_model, req)
        rel = abs(r.delta) / max(abs(r.f_input - r.f_baseline), 1e-9)
        rels.append(rel)
        print(f"  {name[:60]:60s} F={r.f_input:7.2f} F'={r.f_baseline:5.2f} |delta|/|dF|={rel:.3f}")
    med = float(np.median(rels))
    passed = med <= MAX_MEDIAN_REL_DELTA
    print(f"  median {med:.3f}, max {max(rels):.3f} (bound on median {MAX_MEDIAN_REL_DELTA}) "
          f"-> {'PASS' if passed else 'FAIL'}")
    return passed


def check_latency(cases, budget_ms):
    print("\n[Check 4/4] /explain latency (end to end)")
    times = []
    for name, req in cases:
        body = ExplainRequest(**req.model_dump())
        t0 = time.perf_counter()
        resp = asyncio.run(explain(body))
        ms = (time.perf_counter() - t0) * 1000
        times.append(ms)
        print(f"  {name[:60]:60s} {resp.explanation.tokens_explained:3d} tokens {ms:8.0f} ms")
    p50, p95 = np.percentile(times, 50), np.percentile(times, 95)
    passed = p95 <= budget_ms
    print(f"  p50 {p50:.0f} ms | p95 {p95:.0f} ms | budget p95 <= {budget_ms} ms -> {'PASS' if passed else 'FAIL'}")
    return passed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p95-budget-ms", type=float, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    load_models()
    if not state.db_ready:
        print("DistilBERT weights are not loaded; this suite needs ml/models/distilbert_tier1.pt.")
        return 1

    cases = [(c["name"], c["req"]) for c in TEST_CASES]
    rng = random.Random(args.seed)
    results = {
        "deletion": check_deletion(cases, rng),
        "randomisation": check_randomisation(cases),
        "completeness": check_completeness(cases),
        "latency": check_latency(cases, args.p95_budget_ms),
    }
    failed = [k for k, v in results.items() if not v]
    print("=" * 75)
    print("RESULT: PASSED" if not failed else f"RESULT: FAILED ({', '.join(failed)})")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
