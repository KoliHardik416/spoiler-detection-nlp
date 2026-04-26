"""
Central configuration for the Spoiler Detection project.
All hyperparameters, file paths, and constants live here.
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

REVIEWS_PATH = os.path.join(DATA_DIR, "IMDB_reviews.json")
MOVIES_PATH = os.path.join(DATA_DIR, "IMDB_movie_details.json")

# Create output directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── Sampling & Splits ──────────────────────────────────────────────────────
RANDOM_SEED = 42
BASELINE_SAMPLE_SIZE = 200_000       
BERT_SAMPLE_SIZE = 100_000        
TEST_SIZE = 0.20                    
VAL_SIZE = 0.10                     

# ─── TF-IDF ─────────────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 50_000
TFIDF_NGRAM_RANGE = (1, 2)

# ─── DistilBERT ─────────────────────────────────────────────────────────────
BERT_MODEL_NAME = "distilbert-base-uncased"
BERT_MAX_LENGTH = 512
BERT_BATCH_SIZE = 32
BERT_LEARNING_RATE = 2e-5
BERT_EPOCHS = 3
BERT_WARMUP_STEPS = 500
BERT_WEIGHT_DECAY = 0.01

# ─── Label mapping ──────────────────────────────────────────────────────────
LABEL_MAP = {False: 0, True: 1}
LABEL_NAMES = ["Non-Spoiler", "Spoiler"]
