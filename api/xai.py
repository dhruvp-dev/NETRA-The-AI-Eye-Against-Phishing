# -*- coding: utf-8 -*-
"""
NETRA - Explanation fusion for /explain
=======================================
Turns the model attribution (ml/explain.py, Integrated Gradients in logit space) and the
multi-signal override (api/main.py: _decide / _signal_count) into one breakdown of the final
risk score, in probability points:

    p0 = sigmoid(F(baseline))      p_m = sigmoid(F(x)) = model score      p_f = final risk_score

    model part:    p_m - p0, split over attribution groups in proportion to their IG totals
    override part: p_f - p_m, split equally over the counted signal units. The override is a
                   threshold on a count of interchangeable units, so this is their exact Shapley value.
    trust part:    when the trusted-sender rule (api/trust.py) lowered the verdict, p_f - p_m (negative)
                   is one rule component, 'trusted_sender', under sender authentication.

    p0 + sum(factor contributions) == p_f

Everything here is a pure function of its inputs, so it is tested without a model.
"""

import math
from typing import Dict, List, Literal, Optional, Sequence, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from ml.features.header_features import HEADER_FEATURE_NAMES

CONVERGENCE_TOLERANCE = 0.10   # |delta| / |F(x) - F(x')| above this is reported as not converged
EVIDENCE_WORDS = 3
MIN_MENTION = 0.01             # factors below 1 probability point are not named in the summary

FACTOR_LABELS = {
    "wording": "Message wording",
    "links": "Links & look-alike domains",
    "sender_auth": "Sender authentication",
    "reply_to": "Reply-To mismatch",
}

# Header feature index ranges per model component (HEADER_FEATURE_NAMES order).
_HEADER_GROUPS = {"spf": range(0, 3), "dkim": range(3, 6), "dmarc": range(6, 9), "reply_to": range(9, 10)}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class TokenAttribution(BaseModel):
    text: str
    segment: Literal["subject", "body", "url"]
    url_index: Optional[int] = Field(None, description="Index in the extracted URL list (segment 'url')")
    start: int = Field(..., ge=0, description="Char offset of the strongest occurrence in the original field")
    end: int = Field(..., ge=0)
    occurrences: int = Field(1, ge=1)
    attribution: float = Field(..., description="Logit-margin contribution, summed over occurrences "
                                                "(+ toward phishing, - toward legitimate)")


class FeatureAttribution(BaseModel):
    name: str
    value: float
    attribution: float = Field(..., description="Logit margin relative to 'no headers provided'")


class FactorComponent(BaseModel):
    source: Literal["model", "rule"]
    name: str
    contribution: float = Field(..., description="Probability points; 0 for a rule signal that fired "
                                                 "but did not change the score")


class ThreatFactor(BaseModel):
    key: Literal["wording", "links", "sender_auth", "reply_to"]
    label: str
    contribution: float = Field(..., description="Probability points, signed")
    share: Optional[float] = Field(None, ge=0, le=100, description="% of the risk-raising total")
    components: List[FactorComponent]
    evidence: List[str]


class Explanation(BaseModel):
    method: Literal["integrated_gradients", "signals_only"]
    target: Literal["phishing_logit_margin"] = "phishing_logit_margin"
    baseline_score: float
    model_score: float
    final_score: float
    override: Optional[Literal["suspicious_floor", "phishing_floor", "trusted_sender"]] = None
    top_phishing_tokens: List[TokenAttribution]
    top_legitimate_tokens: List[TokenAttribution]
    header_attributions: List[FeatureAttribution]
    headers_used_by_model: bool
    threat_factors: List[ThreatFactor]
    summary: str
    completeness_delta: Optional[float] = None
    converged: bool = True
    truncated: bool = False
    tokens_explained: int = 0
    n_steps: int = 0
    explain_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _is_wordlike(text: str) -> bool:
    return any(ch.isalnum() for ch in text)


def _host(url: str) -> str:
    try:
        return urlparse(url if "://" in url else f"http://{url}").hostname or url
    except ValueError:
        return url


def aggregate_tokens(words, top_k: int) -> Tuple[List[TokenAttribution], List[TokenAttribution]]:
    """
    Merge repeated words (case-insensitive, per segment) into one entry: summed attribution,
    offsets of the strongest occurrence. Punctuation-only words are left out of the lists
    (their attribution still counts in the factor totals).
    """
    merged: Dict[Tuple[str, str], dict] = {}
    for w in words:
        if not _is_wordlike(w.text):
            continue
        key = (w.segment, w.text.lower())
        m = merged.get(key)
        if m is None:
            merged[key] = {"best": w, "sum": w.score, "n": 1}
        else:
            m["sum"] += w.score
            m["n"] += 1
            if abs(w.score) > abs(m["best"].score):
                m["best"] = w

    def entry(m):
        b = m["best"]
        return TokenAttribution(text=b.text, segment=b.segment, url_index=b.url_index, start=b.start,
                                end=b.end, occurrences=m["n"], attribution=round(m["sum"], 4))

    pos = sorted((m for m in merged.values() if m["sum"] > 0), key=lambda m: -m["sum"])[:top_k]
    neg = sorted((m for m in merged.values() if m["sum"] < 0), key=lambda m: m["sum"])[:top_k]
    return [entry(m) for m in pos], [entry(m) for m in neg]


def override_units(signals: dict, url_feats: dict) -> List[Tuple[str, str, str]]:
    """
    The threat units the multi-signal override counts (api/main.py: _signal_count uses this list),
    as (factor key, component name, display label). A plain http:// link is reported but not counted.
    A BEC lure (payment/credential request plus pressure) and a link to a program file each count
    as two units, so either alone gives SUSPICIOUS. Urgent wording only
    counts next to another unit, because ordinary account mail uses the same words.
    """
    units = []
    if signals.get("typosquatting_detected"):
        units.append(("links", "typosquatting_detected", "look-alike domain"))
    if signals.get("header_auth_failed"):
        units.append(("sender_auth", "header_auth_failed", "failed sender authentication"))
    if url_feats.get("any_phishing_tld"):
        units.append(("links", "suspicious_tld", "high-risk TLD"))
    if signals.get("ip_url_detected"):
        units.append(("links", "ip_url", "link to an IP address"))
    if signals.get("shortened_url_detected"):
        units.append(("links", "shortened_url", "shortened link"))
    if signals.get("risky_file_link"):
        units.append(("links", "risky_file_link", "link to an archive or HTML file"))
    if signals.get("executable_link"):
        # A program download link is enough for SUSPICIOUS on its own.
        units.append(("links", "executable_link", "link to a program file"))
        units.append(("links", "executable_download", "direct program download"))
    if signals.get("bec_detected"):
        units.append(("wording", "bec_request", "payment or credential request"))
        units.append(("wording", "bec_pressure", "pressure or secrecy"))
    if units and signals.get("urgency_detected"):
        units.append(("wording", "urgency", "urgent language"))
    return units


def _model_groups(ig) -> Dict[str, float]:
    """Logit attribution per model component."""
    groups = {"subject_words": 0.0, "body_words": 0.0, "url_words": 0.0}
    for w in ig.words:
        groups[f"{w.segment}_words"] += w.score
    for name, idx in _HEADER_GROUPS.items():
        groups[name] = sum(ig.header[i] for i in idx)
    return groups


_COMPONENT_FACTOR = {"subject_words": "wording", "body_words": "wording", "url_words": "links",
                     "spf": "sender_auth", "dkim": "sender_auth", "dmarc": "sender_auth",
                     "reply_to": "reply_to"}


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------
def fuse(
    ig,
    signals: dict,
    url_feats: dict,
    urls: Sequence[str],
    auth: Dict[str, str],
    reply_to_matches: bool,
    decided: dict,
    model_classification: str,
    p_model: float,
    hdr_values: Sequence[float],
    headers_used: bool,
    top_k: int,
    trust_reasons: Sequence[str] = (),
) -> Explanation:
    """
    Build the explanation. `decided` is the output of _decide on p_model; `model_classification`
    is the verdict from the score alone (_classify(p_model)), so an override is visible as a difference.
    `auth` holds normalised spf/dkim/dmarc results ('pass' | 'fail' | 'none').
    """
    final = float(decided["risk_score"])
    cls = decided["classification"]
    override = None
    trusted = decided.get("adjustment") == "trusted_sender"
    if trusted:
        override = "trusted_sender"
    elif cls != model_classification:
        override = "phishing_floor" if cls == "PHISHING" else "suspicious_floor"

    components: Dict[str, List[FactorComponent]] = {k: [] for k in FACTOR_LABELS}

    # Model part: p_m - p0 over attribution groups.
    if ig is not None:
        p0 = sigmoid(ig.f_baseline)
        groups = _model_groups(ig)
        total_attr = sum(groups.values())
        f_diff = ig.f_input - ig.f_baseline
        # Normalise by the attribution total so the parts add up exactly; fall back to the exact
        # function difference when IG is far from converged (total near 0 or of the wrong sign).
        denom = total_attr if total_attr * f_diff > 0 and abs(total_attr) >= 0.5 * abs(f_diff) else f_diff
        scale = (p_model - p0) / denom if abs(denom) > 1e-9 else p_model * (1 - p_model)
        for name, a in groups.items():
            if name in ("spf", "dkim", "dmarc", "reply_to") and not headers_used:
                continue
            if name == "subject_words" and a == 0.0 and not any(w.segment == "subject" for w in ig.words):
                continue
            if name == "url_words" and not urls:
                continue
            components[_COMPONENT_FACTOR[name]].append(
                FactorComponent(source="model", name=name, contribution=round(a * scale, 4)))
        rel = abs(ig.delta) / max(abs(f_diff), 1e-9)
    else:
        p0 = p_model
        rel = 0.0

    # Override part: p_f - p_m over the counted units (zero without an override).
    units = override_units(signals, url_feats)
    per_unit = (final - p_model) / len(units) if override and not trusted and units else 0.0
    for key, name, _label in units:
        components[key].append(FactorComponent(source="rule", name=name, contribution=round(per_unit, 4)))
    if trusted:
        components["sender_auth"].append(FactorComponent(source="rule", name="trusted_sender",
                                                         contribution=round(final - p_model, 4)))

    evidence = _evidence(ig, signals, units, urls, auth, reply_to_matches)
    if trusted:
        evidence["sender_auth"].extend(r for r in trust_reasons if r not in evidence["sender_auth"])

    factors = []
    for key, comps in components.items():
        if not comps and not evidence[key]:
            continue
        factors.append(ThreatFactor(key=key, label=FACTOR_LABELS[key],
                                    contribution=round(sum(c.contribution for c in comps), 4),
                                    components=comps, evidence=evidence[key]))
    positive_total = sum(f.contribution for f in factors if f.contribution > 0)
    for f in factors:
        if f.contribution > 0 and positive_total > 0:
            f.share = round(100 * f.contribution / positive_total, 1)
    factors.sort(key=lambda f: -f.contribution)

    top_pos, top_neg = aggregate_tokens(ig.words, top_k) if ig is not None else ([], [])
    header_attr = []
    if ig is not None and headers_used:
        header_attr = [FeatureAttribution(name=n, value=v, attribution=round(a, 4))
                       for n, v, a in zip(HEADER_FEATURE_NAMES, hdr_values, ig.header)]

    return Explanation(
        method="integrated_gradients" if ig is not None else "signals_only",
        baseline_score=round(p0, 4),
        model_score=round(p_model, 4),
        final_score=round(final, 4),
        override=override,
        top_phishing_tokens=top_pos,
        top_legitimate_tokens=top_neg,
        header_attributions=header_attr,
        headers_used_by_model=bool(ig is not None and headers_used),
        threat_factors=factors,
        summary=summarize(cls, final, factors, override, p_model, [u[2] for u in units], trust_reasons),
        completeness_delta=round(ig.delta, 4) if ig is not None else None,
        converged=rel <= CONVERGENCE_TOLERANCE,
        truncated=bool(ig.truncated) if ig is not None else False,
        tokens_explained=ig.n_tokens if ig is not None else 0,
        n_steps=ig.n_steps if ig is not None else 0,
    )


def _evidence(ig, signals, units, urls, auth, reply_to_matches) -> Dict[str, List[str]]:
    ev: Dict[str, List[str]] = {k: [] for k in FACTOR_LABELS}

    if ig is not None:
        # Evidence follows the sign of the factor: words that raised risk for a risk-raising
        # factor, words that lowered it for a risk-lowering one.
        text_words = [w for w in ig.words if w.segment in ("subject", "body")]
        pos, neg = aggregate_tokens(text_words, EVIDENCE_WORDS)
        raising = sum(w.score for w in text_words) >= 0
        ev["wording"] = [f'"{t.text}"' for t in ((pos or neg) if raising else (neg or pos))]
        url_scores: Dict[int, float] = {}
        for w in ig.words:
            if w.segment == "url" and w.url_index is not None:
                url_scores[w.url_index] = url_scores.get(w.url_index, 0.0) + w.score
        sign = 1 if sum(url_scores.values()) >= 0 else -1
        for i, _ in sorted(url_scores.items(), key=lambda kv: -sign * kv[1])[:2]:
            if i < len(urls):
                ev["links"].append(_host(urls[i]))
    if signals.get("urgency_detected"):
        ev["wording"].append("urgent language")

    for key, _name, label in units:
        if label not in ev[key]:
            ev[key].append(label)

    failed = [k.upper() for k in ("spf", "dkim", "dmarc") if auth.get(k) == "fail"]
    passed = [k.upper() for k in ("spf", "dkim", "dmarc") if auth.get(k) == "pass"]
    if failed:
        ev["sender_auth"].append(f"{', '.join(failed)} failed")
    if passed:
        ev["sender_auth"].append(f"{', '.join(passed)} passed")
    if not reply_to_matches:
        ev["reply_to"].append("Reply-To domain differs from sender")
    return ev


def _join(items: List[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def summarize(classification: str, final: float, factors: List[ThreatFactor], override: Optional[str],
              p_model: float, unit_labels: List[str], trust_reasons: Sequence[str] = ()) -> str:
    """One deterministic sentence (or three) explaining the verdict. No generated text."""
    head = f"{classification} (risk {final:.0%})."

    def describe(f: ThreatFactor, pct: Optional[str]) -> str:
        ev = f", ".join(f.evidence[:EVIDENCE_WORDS])
        detail = f"{ev}: {pct}" if ev and pct else (ev or pct or "")
        return f"{f.label.lower()} ({detail})" if detail else f.label.lower()

    raising = [f for f in factors if f.share]
    notable = [f for f in raising if f.contribution >= MIN_MENTION]
    lowering = sorted((f for f in factors if f.contribution <= -MIN_MENTION), key=lambda f: f.contribution)
    parts = [head]
    if override == "trusted_sender":
        why = "; ".join(trust_reasons) if trust_reasons else "trusted sender"
        parts.append(f"The model alone scored {p_model:.0%}, but the sender check passed ({why}), so the verdict was lowered to "
                     f"{classification}. The model over-flags brand and notification mail; this rule "
                     "checks the sender and where the links go.")
    elif override:
        # The rule signals decided this verdict, not the model score, so they lead.
        parts.append(f"The model alone scored {p_model:.0%}, but {len(unit_labels)} independent threat signals "
                     f"({_join(unit_labels)}) raised the verdict to {classification}.")
    elif classification == "LEGITIMATE" or not raising:
        if lowering:
            parts.append(f"Risk was lowered mainly by the {describe(lowering[0], None)}.")
        if notable and classification == "LEGITIMATE":
            parts.append(f"The strongest risk factor was the {describe(notable[0], None)}, "
                         "which was not enough to flag it.")
        elif not notable:
            parts.append("No strong phishing indicators were found.")
    else:
        top = [describe(f, f"{f.share:.0f}%") for f in (notable or raising)[:2]]
        parts.append(f"Main drivers: {_join(top)}.")
    return " ".join(parts)
