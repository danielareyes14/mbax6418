"""
MBAX 6418 Assignment 1 — LLM classification and scoring (Steps 1, 2, 6).

Loads the cleaned Amazon Gift-Cards reviews, classifies a chosen set with the
LLM via an OpenAI-compatible endpoint, and scores the predictions against the
"correct answer" derived from the star rating AFTERWARDS. The model never sees
the rating.

Modes:
  --mode binary   (Step 2) 100-row first batch; correct = POSITIVE if rating>=4
                           else NEGATIVE.  (raw order, reflects the lopsided data)
  --mode three    (Step 6) balanced ~50/class from the whole file, fixed seed;
                           correct = POSITIVE 4-5, NEUTRAL 3, NEGATIVE 1-2.

Save:
  - one JSONL line per review (raw LLM answer + parses + label)
  - a CSV/parquet of results for the dashboard
  - printed agreement, per-class breakdown and confusion.

Usage:
  python llm_classify.py --mode binary
  python llm_classify.py --mode three --per 50 --seed 42
"""
import argparse
import json
import os
import time
from datetime import datetime

import openai
import pandas as pd

from prompt import SYSTEM_PROMPT, build_user_prompt, parse_llm_response, NRC_EMOTIONS

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "reviews_clean.parquet")
OUTDIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTDIR, exist_ok=True)

# OpenAI-compatible endpoint (Hermes Agent provider config).
BASE_URL = "http://dobolyi.com:9000/v1"
API_KEY = "6418"
MODEL = "DeepSeek-V4-Flash-0731"

STAR_CLASS = {1: "NEGATIVE", 2: "NEGATIVE", 3: "NEUTRAL",
              4: "POSITIVE", 5: "POSITIVE"}


def true_from_rating(rating, mode):
    """Correct answer derived ONLY from the rating (never shown to the model)."""
    if mode == "binary":
        return "POSITIVE" if int(round(rating)) >= 4 else "NEGATIVE"
    return STAR_CLASS[int(round(rating))]


def call_llm(client, title, text, mode):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(title, text, mode)},
        ],
        temperature=0,
        max_tokens=1024,
    )
    msg = resp.choices[0].message
    content = (msg.content or "").strip()
    # The server sometimes puts the answer in a 'reasoning' field but leaves
    # content empty when the token budget is tight — fall back to it.
    if not content:
        reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) or ""
        content = reasoning.strip()
    return content


def classify_batch(df, mode, out_path, max_rows=None):
    """Classify rows with the LLM, appending JSONL as it goes (resumable)."""
    df = df.reset_index(drop=True)
    if max_rows:
        df = df.head(max_rows)
    client = openai.OpenAI(base_url=BASE_URL, api_key=API_KEY)

    # Resume: load already-done rows.
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["idx"])
                except Exception:
                    pass
    print(f"{len(done)} rows already done; classifying remaining "
          f"{len(df)-len(done)}")

    recs = []
    t0 = time.time()
    fh = open(out_path, "a", encoding="utf-8")
    try:
        for i, row in df.iterrows():
            if i in done:
                continue
            raw = call_llm(client, row["title"], row["text"], mode)
            sent, emo = parse_llm_response(raw, mode, valid_emotions=NRC_EMOTIONS)
            rec = {
                "idx": int(i),
                "mode": mode,
                "title": row["title"], "text": row["text"],
                "rating": float(row["rating"]),
                "raw": raw, "sentiment": sent, "emotion": emo,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            recs.append(rec)
            if (len(done) + len(recs)) % 10 == 0:
                el = time.time() - t0
                print(f"  {len(done)+len(recs)}/{len(df)} in {el:.0f}s"
                      f" (~{el/(len(done)+len(recs)):.2f}s/call)", flush=True)
    finally:
        fh.close()
    return recs


def score(out_path, mode):
    """Score the saved JSONL: agreement vs rating, per-class, confusion."""
    recs = []
    with open(out_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                recs.append(json.loads(line))
    print(f"\nScoring {len(recs)} records ({mode} mode)")

    rows = []
    for r in recs:
        true_lbl = true_from_rating(r["rating"], mode)
        pred = r["sentiment"]
        rows.append({
            "idx": r["idx"], "title": r["title"], "text": r["text"],
            "rating": r["rating"], "true": true_lbl, "pred": pred,
            "correct": bool(pred and pred == true_lbl),
            "emotion": r["emotion"], "raw": r["raw"] or "",
        })
    sdf = pd.DataFrame(rows)

    n = len(sdf)
    agree = int(sdf["correct"].sum()) if n else 0
    print(f"\nAGREEMENT with rating: {agree}/{n} = {agree/n:.1%}")

    print("\nPer-class (rows = 'true' from rating, cols = model prediction):")
    true_list = [r["true"] for r in rows]
    pred_list = [r["pred"] for r in rows]
    classes = {"binary": ["NEGATIVE", "POSITIVE"], "three": ["NEGATIVE", "NEUTRAL", "POSITIVE"]}[mode]
    hdr = "true\\pred"
    print(f"{hdr:<12}" + "".join(f"{c:>12}" for c in classes) + f"{'hit%':>8}")
    from collections import OrderedDict
    for t in classes:
        idxs = [k for k, r in enumerate(rows) if r["true"] == t]
        row_counts = OrderedDict((c, 0) for c in classes)
        for k in idxs:
            p = pred_list[k]
            if p in row_counts:
                row_counts[p] += 1
        hit = row_counts[t] / len(idxs) * 100 if idxs else float("nan")
        cells = "".join(f"{row_counts[c]:>12}" for c in classes)
        print(f"{t:<12}" + cells + f"{hit:>7.1f}%")

    # Save results (parquet + csv) for the dashboard.
    cols = [c for c in ["idx", "title", "text", "rating", "true", "pred",
                        "correct", "emotion", "raw"] if c in sdf.columns]
    res_df = sdf[cols].copy()
    parquet_path = os.path.join(OUTDIR, f"results_{mode}.parquet")
    res_df.to_parquet(parquet_path, index=False)
    res_df.to_csv(os.path.join(OUTDIR, f"results_{mode}.csv"), index=False)
    print(f"\nSaved results -> {parquet_path}")

    return sdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["binary", "three"], default="binary")
    ap.add_argument("--per", type=int, default=50, help="per-class count (three mode)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=None, help="binary: max rows (default 100)")
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(DATA)

    if args.mode == "binary":
        # Step 2: first 100 rows in raw order (lopsided toward high ratings).
        sample = df
        out = os.path.join(OUTDIR, "run_binary_raw.jsonl")
        n = args.n or 100
    else:
        # Step 6: balanced ~50/class from the whole file, fixed seed.
        df["_cls"] = df["rating_int"].map(STAR_CLASS)
        sample = (df.groupby("_cls", group_keys=False)
                  .apply(lambda g: g.sample(min(len(g), args.per), random_state=args.seed))
                  .sample(frac=1, random_state=args.seed)
                  .reset_index(drop=True))
        out = os.path.join(OUTDIR, f"run_three_class_balanced_seed{args.seed}.jsonl")
        n = None

    print(f"Mode={args.mode}  sample_size={len(sample)}")
    print(f"Raw output -> {out}")

    if not args.score_only:
        classify_batch(sample, args.mode, out, max_rows=n)

    score(out, args.mode)


if __name__ == "__main__":
    main()
