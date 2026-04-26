"""
Text preprocessing utilities for the Spoiler Detection project.
Covers cleaning, tokenization, stopword removal, and TF-IDF vectorization.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE

STOP_WORDS = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    """
    Lowercase, strip HTML, URLs, and non-alphabetic characters.
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_and_remove_stopwords(text: str) -> list[str]:
    """
    Tokenize and remove English stopwords.
    """

    tokens = word_tokenize(text)
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def preprocess(text: str) -> str:
    """
    Full preprocessing pipeline: clean → tokenize → remove stopwords → rejoin.
    """

    clean_txt = clean_text(text)
    tokens = tokenize_and_remove_stopwords(clean_txt)
    return " ".join(tokens)


def build_tfidf(train_texts, val_texts=None, test_texts=None,
                max_features=TFIDF_MAX_FEATURES,
                ngram_range=TFIDF_NGRAM_RANGE):
    """
    Fit a TF-IDF vectorizer on training texts and transform all splits.

    Returns:
    dict
        Keys: 'vectorizer', 'X_train', and optionally 'X_val', 'X_test'.
    """

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
    )

    result = {
        "vectorizer": vectorizer,
        "X_train": vectorizer.fit_transform(train_texts),
    }

    if val_texts is not None:
        result["X_val"] = vectorizer.transform(val_texts)
    if test_texts is not None:
        result["X_test"] = vectorizer.transform(test_texts)

    return result
