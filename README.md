# 🎬 Detecting Spoilers in Movie Reviews

**Team:** Hardik Kolisetty, Aryan Chokshi, Preyas Joshi, Akanksh Bandaru

## About the Project

Online movie reviews are a primary resource for audiences deciding whether to watch a new movie. However, we have realized that manly reviews contain plot spoilers that can ruin the viewing experience. We are developing a system that can automatically detect spoilers in the reviews and warn the users before they accidentally reach surprise endings, character deaths, or major plot twistsDetecting spoilers is challenging because reviews containing spoilers often appear structurally identical to normal reviews, differing only in subtle semantic content. This project will explore how NLP techniques can be used to classify reviews as spoiler or non-spoiler based on linguistic features and contextual patterns in the text.

## Dataset

[IMDB Spoiler Dataset](https://www.kaggle.com/datasets/rmisra/imdb-spoiler-dataset) — 573,913 user-generated reviews with binary spoiler labels.

| Statistic     | Value   |
| ------------- | ------- |
| Total Reviews | 573,913 |
| Spoiler       | 150,924 |
| Non-Spoiler   | 422,989 |
| Movies        | 1,572   |

## Models

| Model                   | Description                                                      |
| ----------------------- | ---------------------------------------------------------------- |
| **Majority Class**      | Always predicts non-spoiler                                      |
| **Naive Bayes**         | Multinomial NB with TF-IDF features                              |
| **Logistic Regression** | LR with TF-IDF, balanced class weights                           |
| **DistilBERT**          | Fine-tuned `distilbert-base-uncased` for sequence classification |

TF-IDF = Term Frequency-Inverse Document Frequency. It is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus.

## Setup

```bash
# 1. Download the dataset and place the dataset files in the Data folder
# 2. Create a virtual environment
python3 -m venv .venv

# 3. Activate the environment
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download NLTK data (required for text processing)
python download_nltk_resources.py

# 6. Run notebooks in order
cd notebooks
jupyter notebook
```

## Results

Each model saves its evaluation metrics to `results/<model_name>.json` containing accuracy, precision, recall, F1, and ROC-AUC. Notebook 5 produces a comparative summary across all models.
