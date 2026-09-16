"""
Daniela Reyes — Assignment 1: Sentiment & Emotion Classification of Amazon Reviews
Step 3 + 5: Classify review sentiment with a pre-trained RoBERTa transformer,
then CHECK the predicted sentiment against the star-rating ground truth.

Model: cardiffnlp/twitter-roberta-base-sentiment-latest
  Sentiment transformer pre-trained to output exactly:
     0 = negative, 1 = neutral, 2 = positive
  This matches our star-derived label space, so the check is apples-to-apples.

Design:
  - Load the cleaned parquet produced by 01_prepare_data.py
  - Run the model over the review documents in batches (CPU)
  - Evaluate predicted vs. star-rating ground-truth via: accuracy,
    macro F1, per-class precision/recall/F1, and a confusion matrix.
  - Persist predictions for the dashboard step.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}  # model output label order
# Inverse of the star-derived sentiment labels ({name: int})
SENTIMENT_NAMES_INV = {"negative": 0, "neutral": 1, "positive": 2}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=6000,
                   help="Number of reviews to classify. 0 = all.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--maxlen", type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(os.cpu_count())

    data_path = os.path.join(os.path.dirname(__file__), "data", "reviews_clean.parquet")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} rows from reviews_clean.parquet")

    # Deterministic stratified sample so the check reflects all three classes.
    # (pandas >=3: groupby().apply() DROPS the group column; groupby().sample()
    #  keeps it, so use it and cap per-class counts instead.)
    df = df.groupby("sentiment", group_keys=False).apply(
        lambda g: g.sample(min(len(g), args.n // 3), random_state=args.seed)
    ).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    # pandas >=3 groupby().apply() drops the group-by key column, but keeps
    # 'sentiment_name' — recover the integer label from it (1:1 mapping).
    df["sentiment"] = df["sentiment_name"].map(SENTIMENT_NAMES_INV).astype(int)
    print(f"Classifying {len(df):,} reviews "
          f"(distribution:\n{df['sentiment_name'].value_counts().to_dict()})")

    # --- load model -----------------------------------------------------------
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    # --- run inference in batches ---------------------------------------------
    docs = df["document"].tolist()
    preds, confs = [], []
    t_start = time.time()
    for i in range(0, len(docs), args.batch):
        chunk = docs[i:i + args.batch]
        enc = tok(chunk, padding=True, truncation=True, max_length=args.maxlen,
                  return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        conf, idx = probs.max(dim=-1)
        preds.extend(idx.tolist())
        confs.extend(conf.tolist())
        if (i // args.batch + 1) % 10 == 0:
            el = time.time() - t_start
            print(f"  {i + len(chunk):>{len(str(len(docs)))}}/{len(docs)} "
                  f"in {el:.0f}s")
    rate = len(docs) / (time.time() - t_start)
    print(f"Done. {rate:.1f} docs/sec")

    df["pred_int"] = preds
    df["pred_confidence"] = confs
    df["pred_sentiment"] = df["pred_int"].map(ID2LABEL)

    # --- CHECK predicted sentiment against star-rating ground truth ----------
    y_true = df["sentiment"].astype(int).to_numpy()
    y_pred = np.array(preds)

    print("\n" + "=" * 62)
    print("STEP 5 — CHECK: predicted sentiment vs. star-rating ground truth")
    print("=" * 62)
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1:  {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print("\n--- Classification report ---")
    print(classification_report(
        y_true, y_pred, labels=[0, 1, 2],
        target_names=["negative(1-2★)", "neutral(3★)", "positive(4-5★)"],
        zero_division=0))

    print("\n--- Confusion matrix (rows=true from stars, cols=model predicted) ---")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    tcv = "true \\ pred"
    labels = ["negative", "neutral", "positive"]
    print(f"{tcv:<22}" + "".join(f"{l:>10}" for l in labels))
    for row, lab in zip(cm, labels):
        print(f"{lab:<22}" + "".join(f"{v:>10}" for v in row))
    pct = cm / cm.sum(axis=1, keepdims=True)
    print("\n--- Row-normalized (each true class sums to 100%) ---")
    print(f"{tcv:<22}" + "".join(f"{l:>10}" for l in labels))
    for row, lab in zip(pct, labels):
        print(f"{lab:<22}" + "".join(f"{v*100:>9.1f}%" for v in row))

    # --- why does the model get things wrong? (slice by rating) ----------------
    print("\n--- Accuracy by star rating ---")
    acc_by_rating = df.groupby("rating_int").apply(
        lambda g: (g["pred_int"].astype(int) == g["sentiment"].astype(int)).mean(),
        include_groups=False)
    for r, a in acc_by_rating.items():
        print(f"  {int(r)}★: {a:.3f}  (n={len(df[df['rating_int']==r])})")

    # --- persist ----------------------------------------------------------------
    df.to_parquet(os.path.join(os.path.dirname(__file__), "data",
                               "reviews_predicted.parquet"), index=False)
    print(f"\nSaved predictions ({len(df):,} rows) -> data/reviews_predicted.parquet")


if __name__ == "__main__":
    main()
