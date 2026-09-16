# Sentiment & Emotion Classification of Amazon Reviews

**Author:** Daniela Reyes · University of Colorado

Classifies Amazon reviews (2023 **Gift Cards** category) into **sentiment**
(negative / neutral / positive) from the star rating, detects the **primary
emotion**, and checks the model's predictions against the star-rating ground
truth. Includes an interactive, self-contained HTML dashboard.

## Data

Amazon 2023 Gift Cards reviews (each line = one review):
`https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Gift_Cards.jsonl.gz`

Fields include `rating`, `title`, `text`, `verified_purchase`, `helpful_vote`,
`timestamp`, `images`, `asin`, `parent_asin`, `user_id`.

### Ground-truth labels (from star rating)

| Stars | Sentiment | Label |
|-------|-----------|-------|
| 1–2 | negative | 0 |
| 3 | neutral | 1 |
| 4–5 | positive | 2 |

## Models

- **Sentiment:** `cardiffnlp/twitter-roberta-base-sentiment-latest` (pre-trained
  transformer, outputs negative/neutral/positive).
- **Emotion:** `bhadresh-savani/bert-base-uncased-emotion` (6 Ekman emotions +
  neutral).

## Pipeline

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `01_prepare_data.py` | Load JSONL, map stars → sentiment, clean text, profile data |
| 2 | `02_classify.py` | RoBERTa sentiment prediction + check vs. star rating (metrics, confusion matrix) |
| 3 | `03_emotion.py` | BERT primary-emotion detection + emotion × rating cross-tab |
| 4 | `04_make_dashboard.py` | Bakes predictions + stats into a single self-contained HTML dashboard |

## Dashboard

`Sentiment_Emotion_Dashboard.html` — a fully self-contained page (data embedded
as JSON, no server or network needed). Open it by double-clicking, or view the
hosted version (GitHub Pages). Features:

- KPI cards (accuracy, matched/mismatched counts)
- **Correct vs. mismatched filter** plus true/predicted sentiment, emotion, and
  free-text search
- Sortable, explorable review table
- Emotion-mix chart and "where it goes wrong" confusion pairs

## Setup

```bash
uv venv .venv --python 3.11
uv pip install pandas pyarrow transformers torch scikit-learn matplotlib seaborn
```

```bash
# 1. Prepare data (downloads Gift_Cards.jsonl.gz if needed)
python 01_prepare_data.py

# 2. Classify sentiment (--n rows; stratified)
python 02_classify.py --n 6000

# 3. Detect emotion
python 03_emotion.py --n 6000

# 4. Build the dashboard
python 04_make_dashboard.py
```

> Note: pre-trained transformer inference runs on CPU at a few docs/sec, so the
> full 152k-review dataset takes several hours. A stratified sample is used for
> evaluation and the dashboard.
