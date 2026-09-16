"""Proper throughput benchmark: tune batch size + torch threads on CPU."""
import os
import time

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DATA = os.path.join(os.path.dirname(__file__), "data", "reviews_clean.parquet")

torch.set_num_threads(os.cpu_count())  # use all cores for intra-op
print(f"torch threads set to {os.cpu_count()}")

df = pd.read_parquet(DATA)
docs = df["document"].iloc[:2000].tolist()

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.eval()
# Freeze + inline for speed; class label map
id2label = model.config.id2label

for bs in [32, 64, 128, 256]:
    starts = range(0, len(docs), bs)
    batches = [docs[i:i+bs] for i in starts]
    # warmup
    enc = tok(batches[0], padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        model(**enc)
    t0 = time.time()
    n = 0
    for b in batches:
        enc = tok(b, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            model(**enc)
        n += len(b)
    dt = time.time() - t0
    print(f"batch={bs:4d}  {n} docs in {dt:5.1f}s  -> {n/dt:6.1f} docs/sec")
