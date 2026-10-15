"""
NETRA - Tier-1 PhishingClassifier (shared architecture)
=======================================================
Single definition used by training (ml/train_distilbert.py) and serving
(api/main.py). Module and attribute names are part of the state_dict keys,
so changing them breaks loading of existing checkpoints.

    DistilBERT [CLS] (768) ----------------------------+
                                                       +--> Linear(800->256) -> ReLU -> Dropout(0.3) -> Linear(256->2)
    10 header features -> Linear(10->32) -> ReLU ------+
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel

HEADER_DIM = 10
PRETRAINED_NAME = "distilbert-base-uncased"


class PhishingClassifier(nn.Module):
    def __init__(self, pretrained_name: str = PRETRAINED_NAME):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(pretrained_name)
        self.header_proj = nn.Linear(HEADER_DIM, 32)
        self.classifier = nn.Sequential(
            nn.Linear(768 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),  # 0 = LEGITIMATE, 1 = PHISHING
        )

    def forward(self, input_ids, attention_mask, header_features):
        cls = self.distilbert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        hdr = torch.relu(self.header_proj(header_features))
        return self.classifier(torch.cat([cls, hdr], dim=1))
