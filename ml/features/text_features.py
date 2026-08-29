"""
NETRA - Text Feature Engineering
=================================
Provides TF-IDF + handcrafted text signal features for phishing detection.

Usage:
    from ml.features.text_features import TextFeatureExtractor
    extractor = TextFeatureExtractor()
    X_train = extractor.fit_transform(df_train)
    X_val   = extractor.transform(df_val)
"""

import re
import logging
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Urgency / phishing keyword list
# ---------------------------------------------------------------------------
URGENCY_KEYWORDS = [
    "verify", "account", "suspended", "click", "login", "confirm",
    "immediately", "unusual", "billing", "update", "password", "security",
    "urgent", "warning", "limited", "expire", "validate", "authorize",
    "access", "alert",
]

# Pre-compile for performance
_URGENCY_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(kw) for kw in URGENCY_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Handcrafted text feature extraction
# ---------------------------------------------------------------------------

def _count_urgency_keywords(text: str) -> int:
    """Count occurrences of urgency/phishing keywords."""
    if not text:
        return 0
    return len(_URGENCY_PATTERN.findall(text))


def _text_length(text: str) -> int:
    """Number of characters in the text."""
    return len(text) if text else 0


def _exclamation_count(text: str) -> int:
    """Count exclamation marks — common in phishing emails."""
    return text.count("!") if text else 0


def _capital_letter_ratio(text: str) -> float:
    """Ratio of uppercase letters to all alphabetic characters."""
    if not text:
        return 0.0
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c.isupper()) / len(alpha)


def extract_handcrafted(df: pd.DataFrame) -> np.ndarray:
    """
    Extract handcrafted text features from a DataFrame with 'body_text' and
    optionally 'subject' columns.

    Returns np.ndarray of shape (n_samples, 4):
        [urgency_count, text_length, exclamation_count, capital_ratio]
    """
    texts = (
        (df.get("subject", pd.Series([""] * len(df))).fillna("") + " " +
         df["body_text"].fillna(""))
        .str.strip()
    )

    urgency = texts.apply(_count_urgency_keywords).values.reshape(-1, 1)
    length = texts.apply(_text_length).values.reshape(-1, 1)
    exclamations = texts.apply(_exclamation_count).values.reshape(-1, 1)
    cap_ratio = texts.apply(_capital_letter_ratio).values.reshape(-1, 1)

    return np.hstack([urgency, length, exclamations, cap_ratio]).astype(np.float32)


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class TextFeatureExtractor:
    """
    Combines TF-IDF on body_text with handcrafted urgency/length features.

    The TF-IDF vectorizer is fit only on training data to prevent leakage.
    """

    def __init__(self, max_features: int = 5000, sublinear_tf: bool = True):
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            sublinear_tf=sublinear_tf,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\b[a-zA-Z]{2,}\b",
            ngram_range=(1, 2),
            min_df=2,
        )
        self._fitted = False

    def _get_text(self, df: pd.DataFrame) -> pd.Series:
        """Concatenate subject + body_text as the corpus for TF-IDF."""
        subject = df.get("subject", pd.Series([""] * len(df))).fillna("")
        body = df["body_text"].fillna("")
        return (subject + " " + body).str.strip()

    def fit_transform(self, df: pd.DataFrame) -> csr_matrix:
        """
        Fit on training data and transform.
        Call this ONLY on the training split.

        Returns:
            sparse matrix of shape (n_samples, max_features + 4)
        """
        text_corpus = self._get_text(df)
        tfidf_matrix = self.tfidf.fit_transform(text_corpus)
        self._fitted = True
        log.info(f"TF-IDF fitted: vocab size = {len(self.tfidf.vocabulary_)}")

        handcrafted = csr_matrix(extract_handcrafted(df))
        return hstack([tfidf_matrix, handcrafted])

    def transform(self, df: pd.DataFrame) -> csr_matrix:
        """
        Transform new data using the already-fitted TF-IDF vectorizer.
        Call this on val/test splits.
        """
        if not self._fitted:
            raise RuntimeError("TextFeatureExtractor must be fit first via fit_transform().")
        text_corpus = self._get_text(df)
        tfidf_matrix = self.tfidf.transform(text_corpus)
        handcrafted = csr_matrix(extract_handcrafted(df))
        return hstack([tfidf_matrix, handcrafted])

    @property
    def feature_names(self) -> list:
        """All feature names (TF-IDF vocab + handcrafted)."""
        tfidf_names = self.tfidf.get_feature_names_out().tolist()
        return tfidf_names + [
            "hc_urgency_count",
            "hc_text_length",
            "hc_exclamation_count",
            "hc_capital_ratio",
        ]


# ---------------------------------------------------------------------------
# Convenience module-level functions (for pipeline compatibility)
# ---------------------------------------------------------------------------

# Module-level singleton (used by train.py via Pipeline)
_extractor = TextFeatureExtractor()


def fit_transform(df: pd.DataFrame) -> csr_matrix:
    """Module-level fit_transform — fits the singleton extractor."""
    return _extractor.fit_transform(df)


def transform(df: pd.DataFrame) -> csr_matrix:
    """Module-level transform — uses the singleton extractor."""
    return _extractor.transform(df)


def get_extractor() -> TextFeatureExtractor:
    """Return the singleton extractor (e.g., to save to disk)."""
    return _extractor
