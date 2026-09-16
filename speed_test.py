"""
Speed test: download the RoBERTa sentiment transformer and time inference
on a small batch so we can size the full run for CPU.
"""
import os
import time

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DATA = os.path.join(os.path.dirname(__file__), "data", "reviews_clean.parquet")

df = pd.read_parquet(DATA)
print(f"GPU available: {torch.cuda.is_available()} | using device: "
      + ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Device count: {torch.cuda.device_count()}")

print(f"\nDownloading/loading model {MODEL} ...")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.eval()
print(f"Loaded in {time.time()-t0:.1f}s")

# quick speed test on 100 docs, batch 32
sample = df["document"].iloc[:100].tolist()
t0 = time.time()
enc = tok(sample, padding=True, truncation=True, max_length=128, return_tensors="pt")
with torch.no_grad():
    logits = model(**enc).logits
preds = logits.argmax(dim=1).tolist()
dt = time.time() - t0
print(f"Predicted {len(sample)} docs in {dt:.1f}s "
      f"(~{len(sample)/dt:.0f} docs/sec single batch)")
print(f"Prediction distribution: {pd.Series(preds).value_counts().to_dict()}")
print("Labels order: [0=negative, 1=neutral, 2=positive]")
