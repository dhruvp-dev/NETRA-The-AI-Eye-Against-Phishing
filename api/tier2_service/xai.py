# -*- coding: utf-8 -*-
"""
NETRA - Tier-2 XAI Module (Integrated Gradients)
==================================================
Implements token-level attribution using Captum's Integrated Gradients.
Explains WHY the model flagged an email as phishing by attributing
importance scores to individual tokens.

Target: XAI adds no more than 300ms on top of inference.
"""

import logging
import time
from typing import List, Dict, Optional

import torch
import numpy as np

log = logging.getLogger(__name__)


class TokenAttributor:
    """
    Wraps Captum Integrated Gradients for token-level explanation.
    """

    def __init__(self, model, tokenizer, device, top_k=10):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.top_k = top_k
        self._ig = None  # Lazy init

    def _init_ig(self):
        """Lazily initialize Integrated Gradients to avoid import overhead."""
        if self._ig is not None:
            return

        try:
            from captum.attr import LayerIntegratedGradients
            # Attribution on the word embedding layer
            self._ig = LayerIntegratedGradients(
                self._forward_for_ig,
                self.model.roberta.embeddings.word_embeddings,
            )
            log.info("Integrated Gradients initialized")
        except ImportError:
            log.warning("captum not installed - XAI disabled")
            self._ig = None

    def _forward_for_ig(self, input_embeds, attention_mask, header_features, tier1_confidence):
        """
        Forward function for Integrated Gradients.
        Takes embeddings directly instead of input_ids.
        """
        # Replace the embedding lookup with direct embedding input
        outputs = self.model.roberta(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        header_proj = torch.relu(self.model.header_projection(header_features))
        tier1_conf = tier1_confidence.unsqueeze(1) if tier1_confidence.dim() == 1 else tier1_confidence
        combined = torch.cat([cls_output, header_proj, tier1_conf], dim=1)
        logits = self.model.classifier(combined)
        return logits

    def explain(
        self,
        text: str,
        header_features: list,
        tier1_confidence: float,
        target_class: Optional[int] = None,
        n_steps: int = 50,
    ) -> Dict:
        """
        Run Integrated Gradients and return top-k token attributions.

        Args:
            text: Input email text
            header_features: 10-dim header feature vector
            tier1_confidence: Tier-1 confidence score
            target_class: Class to explain (default: predicted class)
            n_steps: Number of interpolation steps for IG

        Returns:
            dict with top_tokens, xai_latency_ms
        """
        self._init_ig()
        t0 = time.perf_counter()

        if self._ig is None:
            return {"top_tokens": [], "xai_latency_ms": 0, "error": "captum not available"}

        # Tokenize
        enc = self.tokenizer(
            text, max_length=512, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        hdr_tensor = torch.tensor([header_features], dtype=torch.float32).to(self.device)
        conf_tensor = torch.tensor([tier1_confidence], dtype=torch.float32).to(self.device)

        # Get prediction if no target class specified
        if target_class is None:
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask, hdr_tensor, conf_tensor)
                target_class = int(logits.argmax(dim=1)[0])

        # Generate baseline (all padding tokens)
        baseline_ids = torch.zeros_like(input_ids)
        baseline_ids[:] = self.tokenizer.pad_token_id

        # Get embeddings for input and baseline
        input_embeds = self.model.roberta.embeddings.word_embeddings(input_ids)
        baseline_embeds = self.model.roberta.embeddings.word_embeddings(baseline_ids)

        # Compute attributions
        try:
            attributions = self._ig.attribute(
                inputs=input_embeds,
                baselines=baseline_embeds,
                additional_forward_args=(attention_mask, hdr_tensor, conf_tensor),
                target=target_class,
                n_steps=n_steps,
                return_convergence_delta=False,
            )

            # Sum attributions across embedding dimensions
            attr_scores = attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()

            # Get tokens and their attributions
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
            token_attrs = []

            for i, (token, score) in enumerate(zip(tokens, attr_scores)):
                # Skip padding and special tokens
                if token in [self.tokenizer.pad_token, self.tokenizer.cls_token,
                             self.tokenizer.sep_token, self.tokenizer.bos_token,
                             self.tokenizer.eos_token]:
                    continue
                if attention_mask[0][i] == 0:
                    continue

                # Clean RoBERTa's byte-level BPE token prefix
                clean_token = token.replace("\u0120", "").replace("\u00c4\u00a0", "")
                if not clean_token.strip():
                    continue

                token_attrs.append({
                    "token": clean_token,
                    "attribution": round(float(abs(score)), 4),
                    "position": i,
                })

            # Sort by absolute attribution and take top-k
            token_attrs.sort(key=lambda x: x["attribution"], reverse=True)
            top_tokens = token_attrs[:self.top_k]

            # Normalize attributions to [0, 1]
            if top_tokens:
                max_attr = max(t["attribution"] for t in top_tokens)
                if max_attr > 0:
                    for t in top_tokens:
                        t["attribution"] = round(t["attribution"] / max_attr, 4)

        except Exception as e:
            log.error(f"XAI failed: {e}")
            top_tokens = []

        xai_latency = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "top_tokens":     top_tokens,
            "target_class":   target_class,
            "xai_latency_ms": xai_latency,
        }
