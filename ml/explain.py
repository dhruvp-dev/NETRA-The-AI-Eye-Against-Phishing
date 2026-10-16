"""
NETRA - Integrated Gradients for PhishingClassifier
===================================================
Post-hoc explanation of the frozen Tier-1 model. Weights are never modified;
gradients are taken with respect to the inputs only.

Target:   F = logit[1] - logit[0]  (log-odds of phishing; sigmoid(F) is the served score)
Inputs:   word embeddings e (before position embeddings / LayerNorm) and the 10 header features h,
          integrated together along one straight path from the baseline (e', h') to (e, h).
Baseline: same tokens and mask, with every content token replaced by [SEP]. [CLS], [SEP] and the
          "Subject: / Body: / URL:" scaffolding are kept, so they get exactly zero attribution.
          [SEP] is used because the checkpoint scores it near-neutral (F ~ 0.3-0.5 on phishing and
          legitimate emails alike); [PAD], [MASK], [UNK] and zero vectors all score F ~ 4-9 (phishing),
          which pushes credit onto stopwords. Header baseline is supplied by the caller ("no headers").

    A_tok_i = sum_d (e_id - e'_id) * sum_k w_k dF/de_id(path(alpha_k))   Riemann sum, sum_k w_k = 1
    A_hdr_j =       (h_j  - h'_j)  * sum_k w_k dF/dh_j (path(alpha_k))
    delta   = sum A_tok + sum A_hdr - (F(e, h) - F(e', h'))             (completeness error)

Step placement: on this checkpoint F stays flat along most of the path and then jumps by ~10
logits within ~5% of it. Uniform or Gauss-Legendre nodes miss the jump (completeness error of
60-100%). A forward-only scan over SCAN_INTERVALS equal intervals measures |dF| per interval, and
the gradient nodes are spread in proportion to it (with a floor so flat intervals stay covered),
midpoint rule inside each interval. It is the same straight-line integral, sampled where it changes.

Usage:
    from ml.explain import integrated_gradients
    result = integrated_gradients(model, tokenizer, text, spans, hdr_values, hdr_baseline)
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch

from ml.features.text_builder import Span


@dataclass
class WordAttribution:
    text: str               # surface text as it appears in the model input
    segment: str            # "subject" | "body" | "url"
    url_index: Optional[int]
    start: int              # char offsets inside the original request field
    end: int
    score: float            # logit-margin attribution (sum of its WordPieces)


@dataclass
class IGResult:
    words: List[WordAttribution]
    header: List[float]     # 10 values, HEADER_FEATURE_NAMES order
    f_input: float          # F(x)
    f_baseline: float       # F(x')
    delta: float            # completeness error
    n_tokens: int
    n_steps: int
    truncated: bool


SCAN_INTERVALS = 8   # initial equal intervals of the forward-only scan
REFINE_ROUNDS = 4    # each round splits the steepest intervals into REFINE_SPLIT parts
REFINE_TOP = 1
REFINE_SPLIT = 4
FLAT_SHARE = 0.25    # share of the gradient budget spread by interval width, whatever the scan shows


def _scan_path(f_at, batch: int):
    """
    Forward-only scan of F along alpha in [0, 1]. Starts from SCAN_INTERVALS equal intervals and
    repeatedly splits the REFINE_TOP intervals with the largest |dF|, so a sharp jump ends up in a
    narrow interval. Returns sorted breakpoints and F at each.
    """
    pts = np.linspace(0.0, 1.0, SCAN_INTERVALS + 1)
    vals = np.asarray(f_at(pts, batch))
    for _ in range(REFINE_ROUNDS):
        steep = np.argsort(-np.abs(np.diff(vals)))[:REFINE_TOP]
        new = np.concatenate([np.linspace(pts[i], pts[i + 1], REFINE_SPLIT + 1)[1:-1] for i in steep])
        pts = np.concatenate([pts, new])
        vals = np.concatenate([vals, f_at(new, batch)])
        order = np.argsort(pts)
        pts, vals = pts[order], vals[order]
    return pts, vals


def _allocate_steps(pts: np.ndarray, vals: np.ndarray, n_steps: int):
    """
    Place n_steps midpoint nodes over the scan intervals, in proportion to
    FLAT_SHARE * width_k + (1 - FLAT_SHARE) * |dF_k| / sum |dF| (largest-remainder rounding).
    Returns (alphas, weights); weights are interval width / nodes in it, renormalised to sum to 1.
    """
    width = np.diff(pts)
    change = np.abs(np.diff(vals))
    total = change.sum()
    share = width if total <= 1e-9 else FLAT_SHARE * width + (1 - FLAT_SHARE) * change / total
    raw = share * n_steps
    counts = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - counts))[: n_steps - counts.sum()]:
        counts[i] += 1

    alphas, weights = [], []
    for lo, wd, c in zip(pts[:-1], width, counts):
        for j in range(c):
            alphas.append(lo + (j + 0.5) / c * wd)
            weights.append(wd / c)
    w = np.asarray(weights)
    w = w / w.sum()   # intervals with no node are dropped; delta reports what that costs
    return torch.tensor(alphas, dtype=torch.float32), torch.tensor(w, dtype=torch.float32)


def _margin(model, embeds, mask, hdr) -> torch.Tensor:
    logits = model.forward_embeds(embeds, mask, hdr)
    return logits[:, 1] - logits[:, 0]


def _span_of(start: int, end: int, spans: Sequence[Span]) -> Optional[int]:
    for i, sp in enumerate(spans):
        if sp.start <= start and end <= sp.end:
            return i
    return None


def _aggregate_words(text, offsets, word_ids, owner, tok_attr, spans) -> List[WordAttribution]:
    """Sum WordPiece scores per word and map each word's char range back into its original field."""
    groups = {}
    for t, w in enumerate(word_ids):
        if w is None or owner[t] is None:
            continue
        s, e = offsets[t]
        g = groups.setdefault((owner[t], w), [s, e, 0.0])
        g[0], g[1], g[2] = min(g[0], s), max(g[1], e), g[2] + tok_attr[t]

    words = []
    for (si, _), (s, e, score) in sorted(groups.items(), key=lambda kv: kv[1][0]):
        sp = spans[si]
        shift = sp.field_offset - sp.start
        words.append(WordAttribution(text[s:e], sp.field, sp.index, s + shift, e + shift, score))
    return words


def integrated_gradients(
    model,
    tokenizer,
    text: str,
    spans: Sequence[Span],
    hdr_values: Sequence[float],
    hdr_baseline: Sequence[float],
    max_length: int = 256,
    n_steps: int = 32,
    batch: int = 10,
) -> IGResult:
    """
    Joint IG over word embeddings and header features. The input is not padded: padding is
    masked out, so the logits equal the served (padded) ones and the cost scales with length.
    """
    device = next(model.parameters()).device
    enc = tokenizer(text, max_length=max_length, truncation=True,
                    return_offsets_mapping=True, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    offsets = [tuple(o) for o in enc["offset_mapping"][0].tolist()]
    word_ids = enc.word_ids(0)

    # Content tokens are those inside a field span; specials have empty (0, 0) offsets.
    owner = [_span_of(s, e, spans) if e > s else None for s, e in offsets]
    content = torch.tensor([[o is not None for o in owner]], device=device)
    base_ids = torch.where(content, torch.full_like(ids, tokenizer.sep_token_id), ids)

    emb = model.distilbert.embeddings.word_embeddings
    x_h = torch.tensor([list(hdr_values)], dtype=torch.float32, device=device)
    b_h = torch.tensor([list(hdr_baseline)], dtype=torch.float32, device=device)
    with torch.no_grad():
        x_e, b_e = emb(ids), emb(base_ids)

    def f_at(alpha_values, chunk):
        out = []
        with torch.no_grad():
            for a in torch.tensor(alpha_values, dtype=torch.float32, device=device).split(chunk):
                out += _margin(model, b_e + a.view(-1, 1, 1) * (x_e - b_e), mask.expand(len(a), -1),
                               b_h + a.view(-1, 1) * (x_h - b_h)).tolist()
        return out

    pts, vals = _scan_path(f_at, batch)
    f_b, f_x = float(vals[0]), float(vals[-1])

    g_e = torch.zeros_like(x_e[0])
    g_h = torch.zeros_like(x_h[0])
    alphas, weights = (t.to(device) for t in _allocate_steps(pts, vals, n_steps))
    with torch.enable_grad():
        for a, w in zip(alphas.split(batch), weights.split(batch)):
            e = (b_e + a.view(-1, 1, 1) * (x_e - b_e)).requires_grad_(True)
            h = (b_h + a.view(-1, 1) * (x_h - b_h)).requires_grad_(True)
            m = _margin(model, e, mask.expand(len(a), -1), h)
            ge, gh = torch.autograd.grad(m.sum(), (e, h))
            g_e += (w.view(-1, 1, 1) * ge).sum(0)
            g_h += (w.view(-1, 1) * gh).sum(0)

    tok_attr = ((x_e[0] - b_e[0]) * g_e).sum(-1).tolist()
    hdr_attr = ((x_h[0] - b_h[0]) * g_h).tolist()
    delta = sum(tok_attr) + sum(hdr_attr) - (f_x - f_b)

    last_char = max((e for (s, e), o in zip(offsets, owner) if o is not None), default=0)
    truncated = bool(spans) and last_char < max(sp.end for sp in spans)

    return IGResult(
        words=_aggregate_words(text, offsets, word_ids, owner, tok_attr, spans),
        header=hdr_attr,
        f_input=f_x,
        f_baseline=f_b,
        delta=delta,
        n_tokens=int(mask.sum()),
        n_steps=n_steps,
        truncated=truncated,
    )
