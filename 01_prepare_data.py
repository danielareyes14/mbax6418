"""
Daniela Reyes — Assignment 1: Sentiment & Emotion Classification of Amazon Reviews
Step 1: Load raw Amazon 2023 Gift Cards reviews, build star-rating ground-truth
labels, clean text, and profile the data.
"""
import gzip
import html as html_lib
import json
import os
import re

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_PATH = os.path.join(DATA_DIR, "Gift_Cards.jsonl.gz")
OUT_PATH = os.path.join(DATA_DIR, "reviews_clean.parquet")

# ---- Star rating -> sentiment ground truth -------------------------------
# 1-2 stars = negative(0) | 3 stars = neutral(1) | 4-5 stars = positive(2)
RATING_BINS = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
SENTIMENT_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


def load_reviews(path: str) -> pd.DataFrame:
    """Read a .jsonl.gz Amazon review file into a DataFrame."""
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    print(f"Loaded {len(df):,} review records from {os.path.basename(path)}")
    return df


def clean_text(s: str) -> str:
    """Minimal text cleaning shared by all downstream steps."""
    if not isinstance(s, str):
        return ""
    s = s.strip()
    # Strip HTML tags and unescape entities (e.g. <br />, "&amp;")
    s = re.sub(r"<[^>]+>", " ", s)
    s = html_lib.unescape(s)
    # Collapse leftover whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main() -> None:
    df = load_reviews(RAW_PATH)

    # --- build ground-truth sentiment label from the star rating ----------
    df["rating_int"] = df["rating"].round().astype(int)
    df["sentiment"] = df["rating_int"].map(RATING_BINS)
    df["sentiment_name"] = df["sentiment"].map(SENTIMENT_NAMES)

    # --- combine title + text into one document ---------------------------
    df["text"] = df["text"].fillna("")
    df["title"] = df["title"].fillna("")
    # Avoid duplication when title and text are identical (common in short reviews)
    df["document"] = df.apply(
        lambda r: r["title"] if r["title"] == r["text"] or r["title"] in r["text"]
        else (r["title"] + ". " + r["text"]) if r["title"] else r["text"],
        axis=1,
    )
    df["document"] = df["document"].apply(clean_text)

    # --- sanity / data-quality profile -------------------------------------
    print("\n=== Columns present ===")
    print(list(df.columns))

    print("\n=== Star-rating distribution ===")
    print(df["rating_int"].value_counts().sort_index())

    print("\n=== Ground-truth sentiment distribution ===")
    print(df["sentiment_name"].value_counts())

    print("\n=== Empty-document count ===")
    n_empty = (df["document"] == "").sum()
    print(f"{n_empty:,} reviews with neither title nor text ({100*n_empty/len(df):.2f}%)")

    print("\n=== Verified purchase rate ===")
    print(df["verified_purchase"].value_counts(normalize=True).mul(100).round(1))

    print("\n=== Sample documents per sentiment ===")
    for name in ["negative", "neutral", "positive"]:
        sample = df[df["sentiment_name"] == name]["document"].dropna().iloc[0]
        print(f"\n[{name}] (rating {df[df['sentiment_name']==name]['rating_int'].iloc[0]}★)")
        print(f"  {sample[:200]}")

    # --- persist cleaned frame --------------------------------------------
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved cleaned frame ({len(df):,} rows) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
