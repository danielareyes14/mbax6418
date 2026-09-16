"""
Daniela Reyes — Assignment 1: Sentiment & Emotion Classification of Amazon Reviews
Step 4: Detect the PRIMARY (dominant) emotion per review.

Model: bhadresh-savani/bert-base-uncased-emotion
  BERT fine-tuned on 6 Ekman emotion classes + neutral:
     0=sadness, 1=joy, 2=love, 3=anger, 4=fear, 5=surprise
  We take the class with the highest probability as the primary emotion.

Output: adds 'emotion' (label) and 'emotion_score' (probability) to the
predicted reviews frame and persists it.
"""
import argparse
import os
import time

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "bhadresh-savani/bert-base-uncased-emotion"
ID2EMO = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=6000,
                   help="Number of reviews to classify. 0 = all (rows in parquet).")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--maxlen", type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(os.cpu_count())

    data_path = os.path.join(os.path.dirname(__file__), "data", "reviews_predicted.parquet")
    if not os.path.exists(data_path):
        print("No reviews_predicted.parquet. Run 02_classify.py first.")
        return
    df = pd.read_parquet(data_path)
    if 0 < args.n < len(df):
        df = df.groupby("sentiment_name", group_keys=False).apply(
            lambda g: g.sample(min(len(g), args.n // 3), random_state=42)
        ).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Loaded {len(df):,} rows with predictions")

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    print(f"Emotion model loaded in {time.time()-t0:.1f}s")

    docs = df["document"].tolist()
    emo_labels, emo_scores = [], []
    t_start = time.time()
    for i in range(0, len(docs), args.batch):
        chunk = docs[i:i + args.batch]
        enc = tok(chunk, padding=True, truncation=True, max_length=args.maxlen,
                  return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1)
        score, idx = probs.max(dim=-1)
        emo_scores.extend(score.tolist())
        emo_labels.extend(idx.tolist())
        if (i // args.batch + 1) % 20 == 0:
            el = time.time() - t_start
            print(f"  {i + len(chunk):>{len(str(len(docs)))}}/{len(docs)} in {el:.0f}s")
    print(f"Done. {len(docs)/(time.time()-t_start):.1f} docs/sec")

    df["emotion_int"] = emo_labels
    df["emotion"] = df["emotion_int"].map(ID2EMO)
    df["emotion_score"] = emo_scores

    print("\n--- Primary emotion distribution ---")
    print(df["emotion"].value_counts())

    print("\n--- Emotion x star-rating ground truth (emotion -> share of each) ---")
    ct = pd.crosstab(df["emotion"], df["rating_int"])
    print(ct.to_string())

    df.to_parquet(data_path, index=False)
    print(f"\nSaved (emotion added) -> {data_path}")


if __name__ == "__main__":
    main()
