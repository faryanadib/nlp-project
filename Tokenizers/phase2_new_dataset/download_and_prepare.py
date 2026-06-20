"""
Phase 2 — Step 1: Download and prepare a non-IMDB movie review dataset.

Chosen dataset: **Rotten Tomatoes (Cornell Movie Review / MR)**, available on
HuggingFace as `rotten_tomatoes`.

Why this one:
  * Movie reviews → same domain as our task, but a different source than IMDB
    (critic snippets from rottentomatoes.com, collected by Pang & Lee 2005).
  * Cleanly binary (pos/neg) and perfectly class-balanced.
  * Has official train/validation/test splits (8 530 / 1 066 / 1 066).
  * Small enough to fine-tune quickly, large enough to be meaningful.
  * SST-2 was rejected because its public test split has no labels;
    Yelp/Amazon were rejected because they are not movie-specific.

Outputs CSV splits (columns: text,label) into DATA_DIR.

Usage:
    python download_and_prepare.py
"""

import os

import pandas as pd
from datasets import load_dataset

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET_NAME = "cornell-movie-review-data/rotten_tomatoes"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LABEL_NAMES = {0: "negative", 1: "positive"}


def print_stats(name, df):
    """Print size, class balance and average review length for one split."""
    counts = df["label"].value_counts().sort_index()
    avg_words = df["text"].str.split().str.len().mean()
    avg_chars = df["text"].str.len().mean()
    print(f"  {name:<12} {len(df):>6} rows | "
          + " ".join(f"{LABEL_NAMES[k]}: {v} ({v/len(df):.1%})" for k, v in counts.items())
          + f" | avg length: {avg_words:.1f} words / {avg_chars:.0f} chars")


def main():
    print(f"Loading '{DATASET_NAME}' from HuggingFace…")
    ds = load_dataset(DATASET_NAME)  # has train / validation / test splits

    os.makedirs(DATA_DIR, exist_ok=True)
    splits = {}
    print("\nDataset statistics:")
    for split in ("train", "validation", "test"):
        df = ds[split].to_pandas()[["text", "label"]]
        df = df[df["text"].str.strip().astype(bool)]  # drop empty rows, if any
        splits[split] = df
        print_stats(split, df)

    total = sum(len(d) for d in splits.values())
    print(f"  {'total':<12} {total:>6} rows")

    for split, df in splits.items():
        path = os.path.join(DATA_DIR, f"{split}.csv")
        df.to_csv(path, index=False)
        print(f"Saved → {path}")

    # ── Phase summary ─────────────────────────────────────────────────────────
    ok = all(len(d) > 0 for d in splits.values())
    print("\n" + ("✅ PASSED — dataset downloaded, 3 non-empty splits saved"
                  if ok else "❌ FAILED — one or more splits are empty"))


if __name__ == "__main__":
    main()
