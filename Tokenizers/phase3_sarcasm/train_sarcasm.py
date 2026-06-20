"""
Phase 3 — Step 2a: Train a BERT-based binary sarcasm classifier.

Loads the sarcasm dataset from phase3_sarcasm/data/ (see
DATASET_INSTRUCTIONS.md), splits it 80/10/10, fine-tunes BERT and saves the
checkpoint with the lowest validation loss.

Supported input formats (auto-detected, first match wins):
  * data/*.json — JSON Lines with fields `headline` + `is_sarcastic`
  * data/sarcasm.csv — CSV with columns `text,label`

Usage:
    python train_sarcasm.py
"""

import glob
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_MODEL = "bert-base-uncased"   # generic BERT; sarcasm ≠ sentiment task
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
CHECKPOINT_DIR = os.path.join(HERE, "checkpoints", "best_model")
SPLITS_DIR = os.path.join(HERE, "data", "splits")  # saved for evaluate_sarcasm.py

MAX_LENGTH = 64           # headlines are short
BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1
MAX_GRAD_NORM = 1.0
TRAIN_FRAC, VAL_FRAC = 0.8, 0.1   # remainder = test
SEED = 42

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sarcasm_data():
    """Load the dataset from DATA_DIR; return a DataFrame(text, label)."""
    json_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
    csv_path = os.path.join(DATA_DIR, "sarcasm.csv")
    if json_files:
        rows = []
        with open(json_files[0], encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rows.append({"text": obj["headline"], "label": int(obj["is_sarcastic"])})
        df = pd.DataFrame(rows)
        print(f"Loaded {len(df):,} rows from {os.path.basename(json_files[0])}")
    elif os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)[["text", "label"]]
        print(f"Loaded {len(df):,} rows from sarcasm.csv")
    else:
        raise FileNotFoundError(
            f"No dataset found in {DATA_DIR}. "
            "See DATASET_INSTRUCTIONS.md for download instructions.")
    df = df.dropna().drop_duplicates(subset="text").reset_index(drop=True)
    print(f"After cleaning: {len(df):,} rows | sarcastic share: {df['label'].mean():.1%}")
    return df


class SarcasmDataset(Dataset):
    """Tokenizes (text, label) rows for the DataLoader."""

    def __init__(self, df, tokenizer):
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True,
                             max_length=MAX_LENGTH, padding="max_length",
                             return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[idx])}


def run_epoch(model, loader, optimizer=None, scheduler=None):
    """Train (if optimizer given) or evaluate one epoch. Returns (loss, acc)."""
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for step, batch in enumerate(loader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.set_grad_enabled(training):
            out = model(**batch)
        if training:
            optimizer.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
        total_loss += out.loss.item() * batch["labels"].size(0)
        correct += (out.logits.argmax(-1) == batch["labels"]).sum().item()
        seen += batch["labels"].size(0)
        if training and step % 50 == 0:
            print(f"    step {step:>4}/{len(loader)}  loss {out.loss.item():.4f}", end="\r")
    return total_loss / seen, correct / seen


def main():
    set_seed()
    print(f"Device: {DEVICE}\nBase model: {BASE_MODEL}\n")

    df = load_sarcasm_data().sample(frac=1.0, random_state=SEED)  # shuffle
    n = len(df)
    n_train, n_val = int(n * TRAIN_FRAC), int(n * VAL_FRAC)
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]
    print(f"Split: train {len(train_df):,} / val {len(val_df):,} / test {len(test_df):,}")

    # persist the test split so evaluate_sarcasm.py scores unseen data
    os.makedirs(SPLITS_DIR, exist_ok=True)
    test_df.to_csv(os.path.join(SPLITS_DIR, "test.csv"), index=False)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2).to(DEVICE)

    train_loader = DataLoader(SarcasmDataset(train_df, tokenizer),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SarcasmDataset(val_df, tokenizer),
                            batch_size=BATCH_SIZE)

    total_steps = len(train_loader) * EPOCHS
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * WARMUP_FRACTION), total_steps)

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, scheduler)
        va_loss, va_acc = run_epoch(model, val_loader)
        print(f"Epoch {epoch}/{EPOCHS}  train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | {(time.time()-t0)/60:.1f} min")
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            model.save_pretrained(CHECKPOINT_DIR)
            tokenizer.save_pretrained(CHECKPOINT_DIR)
            print(f"  ↳ new best (val loss {va_loss:.4f}) saved → {CHECKPOINT_DIR}")

    saved = os.path.isfile(os.path.join(CHECKPOINT_DIR, "config.json"))
    print("\n✅ PASSED — sarcasm model trained and checkpoint saved" if saved
          else "\n❌ FAILED — no checkpoint was saved")


if __name__ == "__main__":
    main()
