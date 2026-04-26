"""
Data loading and splitting utilities for the IMDB Spoiler Dataset.
Handles JSONL parsing, stratified sampling, and train/val/test splitting.
"""

import json
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import (
    REVIEWS_PATH, MOVIES_PATH, RANDOM_SEED,
    TEST_SIZE, VAL_SIZE, LABEL_MAP,
)


def load_reviews(filepath: str = REVIEWS_PATH,
                 sample_size: int | None = None) -> pd.DataFrame:
    """
    Load IMDB reviews from a JSONL file.
    """

    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)

    # Map boolean label to int
    df["label"] = df["is_spoiler"].map(LABEL_MAP)

    if sample_size is not None and sample_size < len(df):
        df, _ = train_test_split(
            df, train_size=sample_size,
            stratify=df["label"],
            random_state=RANDOM_SEED,
        )
        df = df.reset_index(drop=True)

    return df


def load_movie_details(filepath: str = MOVIES_PATH) -> pd.DataFrame:
    """
    Load IMDB movie details from a JSONL file.
    """
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def create_splits(df: pd.DataFrame,
                  test_size: float = TEST_SIZE,
                  val_size: float = VAL_SIZE):
    """
    Split a DataFrame into train / validation / test sets (stratified).

    Returns:
    train_df, val_df, test_df : pd.DataFrame
    """
    train_val, test_df = train_test_split(
        df, test_size=test_size,
        stratify=df["label"], random_state=RANDOM_SEED,
    )
    relative_val = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val, test_size=relative_val,
        stratify=train_val["label"], random_state=RANDOM_SEED,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
