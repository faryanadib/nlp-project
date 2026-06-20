"""
Phase 3 — Step 2b: Evaluate the sarcasm classifier on its held-out test split
(saved by train_sarcasm.py to data/splits/test.csv).

Reports accuracy, F1 and the confusion matrix.

Usage:
    python evaluate_sarcasm.py
    python evaluate_sarcasm.py --model <path>
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Configuration ──────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(HERE, "checkpoints", "best_model")
TEST_CSV = os.path.join(HERE, "data", "splits", "test.csv")
MAX_LENGTH = 64
BATCH_SIZE = 32
F1_THRESHOLD = 0.75       # warn below this

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")


def predict_all(texts, tokenizer, model):
    """Batched inference; returns predicted labels."""
    preds = []
    model.eval()
    for i in range(0, len(texts), BATCH_SIZE):
        enc = tokenizer(texts[i:i + BATCH_SIZE], truncation=True, padding=True,
                        max_length=MAX_LENGTH, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            preds.extend(model(**enc).logits.argmax(-1).cpu().tolist())
    return np.array(preds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=CHECKPOINT_DIR)
    args = parser.parse_args()

    if not os.path.isfile(TEST_CSV):
        sys.exit(f"❌ FAILED — {TEST_CSV} not found. Run train_sarcasm.py first.")

    print(f"Device: {DEVICE}\nModel:  {args.model}\n")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(DEVICE)

    df = pd.read_csv(TEST_CSV)
    texts, y_true = df["text"].astype(str).tolist(), df["label"].to_numpy()
    print(f"Test set: {len(texts):,} examples (sarcastic share {y_true.mean():.1%})")

    y_pred = predict_all(texts, tokenizer, model)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\nAccuracy : {acc:.4f}")
    print(f"F1       : {f1:.4f}")
    print("\nConfusion matrix (rows = true, cols = predicted; 0=not sarcastic, 1=sarcastic):")
    print(f"          pred 0   pred 1")
    print(f"  true 0  {cm[0,0]:>6}   {cm[0,1]:>6}")
    print(f"  true 1  {cm[1,0]:>6}   {cm[1,1]:>6}")

    # ── Phase summary ─────────────────────────────────────────────────────────
    collapsed = len(np.unique(y_pred)) < 2
    if collapsed:
        print(f"\n❌ FAILED — model collapsed to a single class ({np.unique(y_pred)[0]})")
        sys.exit(1)
    if f1 < F1_THRESHOLD:
        print(f"\n⚠️  WARNING: F1 {f1:.4f} below {F1_THRESHOLD}")
    print(f"\n✅ PASSED — accuracy {acc:.4f}, F1 {f1:.4f}, no class collapse")


if __name__ == "__main__":
    main()
