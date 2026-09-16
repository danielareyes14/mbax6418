"""
MBAX 6418 Assignment 1 — NRC word-list emotion (Step 5, part 2).

Scores each review's words against the NRC emotion lexicon, sums the raw
term-loadings per emotion, and takes the highest as the word-list answer. It
runs over existing predictions (no model calls) and then compares the
word-list emotion against the LLM emotion.

Lexicon source: NRC Word-Emotion Association Lexicon (EmoLex), bundled with the
`nrclex` package (nrc_en.json) — a public list linking ~6,468 English words to
the 8 basic emotions plus positive/negative valence.

Usage:
  python nrc_emotion.py            # adds NRC emotion to results_*.csv/parquet
  python nrc_emotion.py --mode binary
"""
import argparse
import json
import os
import re

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(ROOT, "outputs")
NRC_PATH = (r"C:/Users/dr721/amazon-sentiment/.venv/Lib/site-packages/"
            r"nrclex/data/nrc_en.json")

# The eight basic emotions — EXCLUDING positive/negative valence, which are
# polarity, not emotions. Exactly the assignment's list.
EMOTIONS = ["anger", "anticipation", "disgust", "fear",
            "joy", "sadness", "surprise", "trust"]


def load_nrc(path=NRC_PATH):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    # data: {word: {"anger": 1, ...}}  or  {word: ["anger", ...]}
    lex = {}
    for word, val in data.items():
        if isinstance(val, dict):
            emos = [e for e, flag in val.items() if flag and e in EMOTIONS]
        elif isinstance(val, list):
            emos = [e for e in val if e in EMOTIONS]
        else:
            emos = []
        if emos:
            lex[word.lower()] = emos
    return lex


def word_list_emotion(text, lex):
    """Highest-scoring NRC emotion for a review, or None if no words match."""
    tokens = re.findall(r"[A-Za-z']+", (text or "").lower())
    scores = {e: 0 for e in EMOTIONS}
    for tok in tokens:
        for e in lex.get(tok, []):
            scores[e] += 1
    top = max(scores, key=lambda e: scores[e])
    return (top, scores[top]) if scores[top] > 0 else (None, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["binary", "three"], default="three")
    args = ap.parse_args()

    res_path = os.path.join(OUTDIR, f"results_{args.mode}.csv")
    if not os.path.exists(res_path):
        print(f"No {res_path} — run llm_classify.py first.")
        return

    df = pd.read_csv(res_path)
    print(f"Loaded {len(df)} results ({args.mode})")

    lex = load_nrc()
    print(f"NRC lexicon: {len(lex)} words, {len(EMOTIONS)} emotions")

    emos, scores = [], []
    for text in df["text"].fillna(""):
        e, s = word_list_emotion(text, lex)
        emos.append(e)
        scores.append(s)
    df["nrc_emotion"] = emos
    df["nrc_score"] = scores

    # Valid LLM emotions are the NRC 8; drop label-variant blanks.
    m1 = df["emotion"].notna()
    m2 = df["nrc_emotion"].notna()
    both = df[m1 & m2]
    agree = int(((both["emotion"] == both["nrc_emotion"])).sum())

    print(f"\nNRC vs LLM emotion agreement (where both have an emotion): "
          f"{agree}/{len(both)} = {agree/len(both):.1%}"
          if len(both) else "no matched pairs")

    print("\n--- Word-list (NRC) emotion distribution ---")
    print(df["nrc_emotion"].value_counts(dropna=False).to_string())

    print("\n--- LLM emotion distribution ---")
    print(df["emotion"].value_counts(dropna=False).to_string())

    print("\n--- Agreement table (rows=NRC, cols=LLM) ---")
    ct = pd.crosstab(df["nrc_emotion"], df["emotion"], dropna=False)
    print(ct.to_string())

    # Persist.
    keep = [c for c in df.columns if c not in ("raw",)]
    df[keep].to_parquet(os.path.join(OUTDIR, f"results_{args.mode}.parquet"), index=False)
    df.to_csv(res_path, index=False)
    print(f"\nSaved (NRC added) -> {res_path} / results_{args.mode}.parquet")


if __name__ == "__main__":
    main()
