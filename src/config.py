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
os.makedirs(os.path.join(RESULTS_DIR, "model_results"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "model_charts"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "evaluation_charts"), exist_ok=True)

# ─── Sampling & Splits ──────────────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE = 0.20                    
VAL_SIZE = 0.10                     

# ─── TF-IDF ─────────────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 50_000
TFIDF_NGRAM_RANGE = (1, 2)

# ─── Label mapping ──────────────────────────────────────────────────────────
LABEL_MAP = {False: 0, True: 1}
LABEL_NAMES = ["Non-Spoiler", "Spoiler"]
