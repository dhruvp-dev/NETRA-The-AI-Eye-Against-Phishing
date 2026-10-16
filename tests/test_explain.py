"""
Explanation tests. The IG engine runs on a tiny randomly initialised DistilBERT (one layer,
768 wide so PhishingClassifier's head fits) with the committed tokenizer: no weights, no downloads.
Fusion tests use hand-built IGResults and need no model at all.
"""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import ml.distilbert_model as dm
import ml.explain as X
from api.xai import fuse, override_units, aggregate_tokens
from ml.features.header_features import HEADER_FEATURE_NAMES, extract as header_extract
from ml.features.text_builder import build_model_text_with_spans

TOKENIZER_DIR = Path(__file__).resolve().parent.parent / "ml" / "models" / "distilbert_tokenizer"
NO_HEADERS = [float(header_extract({}, "", "")[k]) for k in HEADER_FEATURE_NAMES]
FAILED = [float(header_extract({"spf": "fail", "dkim": "fail", "dmarc": "fail"}, "a@x.com", "b@y.com")[k])
          for k in HEADER_FEATURE_NAMES]


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))


@pytest.fixture(scope="module")
def model():
    from transformers import DistilBertConfig, DistilBertModel
    torch.manual_seed(0)
    cfg = DistilBertConfig(vocab_size=30522, dim=768, n_layers=1, n_heads=12, hidden_dim=256)
    mp = pytest.MonkeyPatch()
    mp.setattr(dm.DistilBertModel, "from_pretrained", classmethod(lambda cls, *a, **k: DistilBertModel(cfg)))
    try:
        m = dm.PhishingClassifier()
    finally:
        mp.undo()
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


SUBJECT = "Account suspended"
BODY = "  URGENT: your account is suspended. Verify immediately at the portal."
URLS = ["http://paypa1-secure.example.tk/login"]


def _inputs():
    return build_model_text_with_spans(SUBJECT, BODY, URLS)


def _run(model, tok, n_steps=32, hdr=FAILED, **kw):
    text, spans = _inputs()
    return X.integrated_gradients(model, tok, text, spans, hdr, NO_HEADERS, n_steps=n_steps, **kw)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
def test_forward_embeds_matches_forward_and_padding_is_neutral(model, tok):
    text, _ = _inputs()
    hdr = torch.tensor([FAILED])
    enc = tok(text, return_tensors="pt")
    pad = tok(text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        a = model(enc["input_ids"], enc["attention_mask"], hdr)
        b = model.forward_embeds(model.distilbert.embeddings.word_embeddings(enc["input_ids"]),
                                 enc["attention_mask"], hdr)
        c = model(pad["input_ids"], pad["attention_mask"], hdr)
    assert torch.allclose(a, b, atol=1e-5)
    assert torch.allclose(a, c, atol=1e-4)


def test_ig_margin_equals_served_margin(model, tok):
    text, _ = _inputs()
    enc = tok(text, max_length=256, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"], torch.tensor([FAILED]))
    r = _run(model, tok, n_steps=8)
    assert r.f_input == pytest.approx((logits[0, 1] - logits[0, 0]).item(), abs=1e-4)


def test_completeness(model, tok):
    r = _run(model, tok, n_steps=64)
    total = sum(w.score for w in r.words) + sum(r.header)
    assert total - (r.f_input - r.f_baseline) == pytest.approx(r.delta, abs=1e-5)
    # Relative bound, with an absolute floor: the random tiny model has a small F(x) - F(x').
    assert abs(r.delta) <= max(0.02 * abs(r.f_input - r.f_baseline), 1e-3)


def test_no_content_and_equal_headers_gives_zero_attribution(model, tok):
    text, _ = _inputs()
    r = X.integrated_gradients(model, tok, text, [], NO_HEADERS, NO_HEADERS, n_steps=8)
    assert r.words == []
    assert r.header == [0.0] * len(HEADER_FEATURE_NAMES)
    assert r.f_input == pytest.approx(r.f_baseline, abs=1e-6)


def test_header_attribution_zero_when_headers_match_baseline(model, tok):
    r = _run(model, tok, n_steps=8, hdr=NO_HEADERS)
    assert r.header == [0.0] * len(HEADER_FEATURE_NAMES)


def test_words_map_back_to_original_fields_and_skip_scaffolding(model, tok):
    r = _run(model, tok, n_steps=8)
    fields = {"subject": SUBJECT, "body": BODY}
    for w in r.words:
        source = URLS[w.url_index] if w.segment == "url" else fields[w.segment]
        assert source[w.start:w.end] == w.text
    texts = [w.text for w in r.words]
    assert "Subject" not in texts and "Body" not in texts and "URL" not in texts
    assert {w.segment for w in r.words} == {"subject", "body", "url"}
    # WordPieces are merged into words: "immediately" is several pieces but one word.
    assert "immediately" in texts


def test_truncation_is_reported(model, tok):
    text, spans = build_model_text_with_spans("s", "word " * 200, [])
    r = X.integrated_gradients(model, tok, text, spans, NO_HEADERS, NO_HEADERS, max_length=32, n_steps=8)
    assert r.truncated is True
    assert r.n_tokens == 32
    assert max(w.end for w in r.words) < len("word " * 200)
    assert _run(model, tok, n_steps=8).truncated is False


def test_deterministic(model, tok):
    a, b = _run(model, tok, n_steps=8), _run(model, tok, n_steps=8)
    assert a == b


def test_step_allocation_finds_a_jump():
    # F is flat, then jumps between alpha 0.80 and 0.82: the refined scan must put most nodes there.
    f = lambda pts, _b: [0.0 if a < 0.80 else (10.0 if a > 0.82 else 500 * (a - 0.80)) for a in pts]
    pts, vals = X._scan_path(f, batch=10)
    alphas, weights = X._allocate_steps(pts, vals, 32)
    assert len(alphas) == 32
    assert float(weights.sum()) == pytest.approx(1.0, abs=1e-6)
    inside = ((alphas > 0.78) & (alphas < 0.84)).sum().item()
    assert inside >= 16
    # The weighted midpoint rule integrates the derivative of f to f(1) - f(0).
    grad = lambda a: 500.0 if 0.80 <= a <= 0.82 else 0.0
    assert sum(w * grad(a) for a, w in zip(alphas.tolist(), weights.tolist())) == pytest.approx(10.0, rel=0.25)


def test_matches_captum(model, tok):
    captum = pytest.importorskip("captum.attr")
    text, spans = _inputs()
    r = X.integrated_gradients(model, tok, text, spans, FAILED, NO_HEADERS, n_steps=64)
    enc = tok(text, return_tensors="pt")
    ids, mask = enc["input_ids"], enc["attention_mask"]
    emb = model.distilbert.embeddings.word_embeddings
    content = torch.tensor([[any(sp.start <= s and e <= sp.end for sp in spans) and e > s
                             for s, e in enc["offset_mapping"][0].tolist()]])
    base = torch.where(content, torch.full_like(ids, tok.sep_token_id), ids)
    fwd = lambda e, h: (lambda l: l[:, 1] - l[:, 0])(model.forward_embeds(e, mask.expand(len(e), -1), h))
    ig = captum.IntegratedGradients(fwd)
    ae, ah = ig.attribute((emb(ids), torch.tensor([FAILED])), baselines=(emb(base), torch.tensor([NO_HEADERS])),
                          n_steps=256)
    assert sum(r.header) == pytest.approx(ah.sum().item(), abs=0.05)
    assert sum(w.score for w in r.words) == pytest.approx(ae.sum().item(), abs=0.05)


# ---------------------------------------------------------------------------
# Fusion (pure)
# ---------------------------------------------------------------------------
def _ig(words, header=None, f_input=3.0, f_baseline=0.4, delta=0.0):
    return X.IGResult(words=words, header=header or [0.0] * 10, f_input=f_input, f_baseline=f_baseline,
                      delta=delta, n_tokens=20, n_steps=32, truncated=False)


def _w(text, seg, score, idx=None, start=0):
    return X.WordAttribution(text, seg, idx, start, start + len(text), score)


SIG_NONE = {"header_auth_failed": False, "urgency_detected": False, "suspicious_urls": 0,
            "typosquatting_detected": False}
URL_NONE = {"any_http_url": 0, "any_phishing_tld": 0, "any_typosquatting": 0}


def _fuse(ig, signals=SIG_NONE, url_feats=URL_NONE, decided=None, model_cls="PHISHING", p_model=None,
          headers_used=True, urls=("http://x.tk/a",), auth=None):
    p_model = p_model if p_model is not None else 1 / (1 + np.exp(-ig.f_input))
    decided = decided or {"classification": model_cls, "risk_score": round(p_model, 4)}
    return fuse(ig, signals, url_feats, list(urls), auth or {}, True, decided, model_cls, p_model,
                [0.0] * 10, headers_used, top_k=5)


def test_fusion_parts_add_up_to_final_score():
    header = [0.0, 0.3, 0.0, 0.0, 0.2, 0.0, 0.0, 0.1, 0.0, -0.05]
    words = [_w("verify", "body", 1.2), _w("team", "body", -0.4), _w("paypa1", "url", 1.0, 0),
             _w("Urgent", "subject", 0.25)]
    ig = _ig(words, header, f_input=0.4 + sum(w.score for w in words) + sum(header))
    e = _fuse(ig)
    assert e.baseline_score + sum(f.contribution for f in e.threat_factors) == pytest.approx(e.final_score, abs=1e-3)
    assert sum(f.share for f in e.threat_factors if f.share) == pytest.approx(100.0, abs=0.2)
    assert e.threat_factors[0].contribution >= e.threat_factors[-1].contribution
    assert [t.text for t in e.top_phishing_tokens] == ["verify", "paypa1", "Urgent"]
    assert [t.text for t in e.top_legitimate_tokens] == ["team"]
    assert e.converged


def test_override_uplift_is_split_equally_per_signal_unit():
    signals = {"header_auth_failed": True, "urgency_detected": False, "suspicious_urls": 3,
               "typosquatting_detected": True}
    url_feats = {"any_http_url": 1, "any_phishing_tld": 1, "any_typosquatting": 1}
    units = override_units(signals, url_feats)
    # typosquatting is counted directly and again inside suspicious_urls (capped at 2 units).
    assert [u[1] for u in units] == ["typosquatting_detected", "header_auth_failed", "non_https_link",
                                     "suspicious_tld"]
    ig = _ig([_w("hello", "body", -4.0)], f_input=-3.6, f_baseline=0.4)
    p_model = 1 / (1 + np.exp(3.6))
    e = _fuse(ig, signals, url_feats, decided={"classification": "PHISHING", "risk_score": 0.25},
              model_cls="LEGITIMATE", p_model=p_model)
    assert e.override == "phishing_floor"
    rule = [c for f in e.threat_factors for c in f.components if c.source == "rule"]
    assert len(rule) == 4
    assert all(c.contribution == pytest.approx((0.25 - p_model) / 4, abs=1e-4) for c in rule)
    links = next(f for f in e.threat_factors if f.key == "links")
    assert sum(c.contribution for c in links.components if c.source == "rule") == pytest.approx(
        3 * (0.25 - p_model) / 4, abs=1e-3)
    assert e.baseline_score + sum(f.contribution for f in e.threat_factors) == pytest.approx(0.25, abs=1e-3)
    assert "raised the verdict to PHISHING" in e.summary


def test_fired_signals_without_override_carry_no_weight():
    signals = dict(SIG_NONE, header_auth_failed=True, urgency_detected=True)
    ig = _ig([_w("verify", "body", 2.6)])
    e = _fuse(ig, signals, auth={"spf": "fail"})
    assert e.override is None
    rule = [c for f in e.threat_factors for c in f.components if c.source == "rule"]
    assert [c.name for c in rule] == ["header_auth_failed"] and rule[0].contribution == 0.0
    wording = next(f for f in e.threat_factors if f.key == "wording")
    assert "urgent language" in wording.evidence


def test_headers_unused_are_left_out():
    e = _fuse(_ig([_w("verify", "body", 2.6)]), headers_used=False)
    assert e.header_attributions == [] and e.headers_used_by_model is False
    assert all(c.name not in ("spf", "dkim", "dmarc", "reply_to")
               for f in e.threat_factors for c in f.components)


def test_signals_only_without_model():
    signals = {"header_auth_failed": True, "urgency_detected": False, "suspicious_urls": 1,
               "typosquatting_detected": False}
    e = fuse(None, signals, dict(URL_NONE, any_http_url=1), ["http://a.com"], {"spf": "fail"}, True,
             {"classification": "SUSPICIOUS", "risk_score": 0.08}, "LEGITIMATE", 0.02, [], False, top_k=5)
    assert e.method == "signals_only" and e.top_phishing_tokens == [] and e.completeness_delta is None
    assert e.baseline_score == 0.02
    assert e.baseline_score + sum(f.contribution for f in e.threat_factors) == pytest.approx(0.08, abs=1e-3)


def test_not_converged_is_flagged():
    e = _fuse(_ig([_w("verify", "body", 1.0)], f_input=3.0, f_baseline=0.4, delta=-1.6))
    assert e.converged is False


def test_repeated_words_are_merged_and_punctuation_dropped():
    words = [_w("Verify", "body", 0.5, start=0), _w(".", "body", 0.9, start=6), _w("verify", "body", 0.7, start=8)]
    pos, neg = aggregate_tokens(words, 5)
    assert len(pos) == 1 and neg == []
    assert pos[0].occurrences == 2 and pos[0].attribution == pytest.approx(1.2)
    assert pos[0].start == 8   # strongest occurrence


def test_legitimate_summary_names_what_lowered_risk():
    ig = _ig([_w("roadmap", "body", -9.0), _w("hi", "body", 0.05)], f_input=-8.55, f_baseline=0.4)
    e = _fuse(ig, model_cls="LEGITIMATE", urls=())
    assert e.summary.startswith("LEGITIMATE")
    assert "roadmap" in e.summary
