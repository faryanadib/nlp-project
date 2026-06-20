"""
Phase 2 — Step 3: Evaluate the fine-tuned model on the Rotten Tomatoes
test split and run all Phase 2 sanity checks.

Reports: accuracy, precision, recall, F1, confusion matrix, comparison to
random and majority-class baselines. Then:
  * re-runs ALL Phase 1 sanity checks against the new checkpoint
  * warns if F1 < F1_THRESHOLD
  * checks the model did not collapse to a single class

Usage:
    python evaluate.py                  # evaluates CHECKPOINT_DIR
    python evaluate.py --model <path>   # evaluate any checkpoint
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Configuration ──────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(HERE, "checkpoints", "best_model")
TEST_CSV = os.path.join(HERE, "data", "test.csv")
MAX_LENGTH = 256
BATCH_SIZE = 32
F1_THRESHOLD = 0.75       # flag a warning below this
SEED = 42

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")

# make Phase 1 checks importable
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "phase1_sanity_check"))


def predict_all(texts, tokenizer, model):
    """Batched inference; returns predicted labels for all texts."""
    preds = []
    model.eval()
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        enc = tokenizer(batch, truncation=True, padding=True,
                        max_length=MAX_LENGTH, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**enc).logits
        preds.extend(logits.argmax(-1).cpu().tolist())
    return np.array(preds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=CHECKPOINT_DIR)
    args = parser.parse_args()

    np.random.seed(SEED)
    print(f"Device: {DEVICE}\nModel:  {args.model}\n")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(DEVICE)

    df = pd.read_csv(TEST_CSV)
    texts, y_true = df["text"].astype(str).tolist(), df["label"].to_numpy()
    print(f"Test set: {len(texts):,} reviews")

    y_pred = predict_all(texts, tokenizer, model)

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\nAccuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1       : {f1:.4f}")
    print("\nConfusion matrix (rows = true, cols = predicted; 0=neg, 1=pos):")
    print(f"          pred 0   pred 1")
    print(f"  true 0  {cm[0,0]:>6}   {cm[0,1]:>6}")
    print(f"  true 1  {cm[1,0]:>6}   {cm[1,1]:>6}")

    # ── Baselines ─────────────────────────────────────────────────────────────
    rand_pred = np.random.randint(0, 2, size=len(y_true))
    majority = int(np.bincount(y_true).argmax())
    maj_pred = np.full_like(y_true, majority)
    print("\nBaseline comparison (accuracy / F1):")
    print(f"  random         : {accuracy_score(y_true, rand_pred):.4f} / "
          f"{f1_score(y_true, rand_pred, zero_division=0):.4f}")
    print(f"  majority class : {accuracy_score(y_true, maj_pred):.4f} / "
          f"{f1_score(y_true, maj_pred, zero_division=0):.4f}")
    print(f"  our model      : {acc:.4f} / {f1:.4f}")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    failures, warnings = [], []

    # (a) F1 above threshold
    if f1 < F1_THRESHOLD:
        warnings.append(f"F1 {f1:.4f} is below the {F1_THRESHOLD} threshold")

    # (b) no class collapse: both classes must appear in predictions
    unique = np.unique(y_pred)
    if len(unique) < 2:
        failures.append(f"model collapsed: only predicts class {unique[0]}")
    else:
        share = np.bincount(y_pred, minlength=2) / len(y_pred)
        print(f"\nPrediction class share: neg {share[0]:.1%} / pos {share[1]:.1%}")

    # (c) re-run all Phase 1 sanity checks on the new model
    print("\nRe-running Phase 1 sanity checks on the fine-tuned model…")
    import sanity_check as sc
    p1 = sc.run_all_checks(args.model, verbose=True)
    p1_failed = [n for n, ok, _ in p1 if not ok]
    if p1_failed:
        failures.append("Phase 1 checks failed: " + ", ".join(p1_failed))

    # ── Phase summary ─────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    for w in warnings:
        print(f"⚠️  WARNING: {w}")
    if failures:
        print("❌ FAILED — " + "; ".join(failures))
        sys.exit(1)
    print(f"✅ PASSED — F1 {f1:.4f}, no class collapse, "
          f"all {len(p1)} Phase 1 checks passed"
          + (" (with warnings)" if warnings else ""))


if __name__ == "__main__":
    main()
