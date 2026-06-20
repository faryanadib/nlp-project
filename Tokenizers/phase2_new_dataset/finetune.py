"""
Phase 2 — Step 2: Fine-tune the existing BERT sentiment model on the
Rotten Tomatoes dataset prepared by download_and_prepare.py.

Standard PyTorch training loop with per-epoch validation-loss tracking.
The checkpoint with the lowest validation loss is saved to CHECKPOINT_DIR
(in HuggingFace format, so it can be loaded with from_pretrained()).

Usage:
    python finetune.py
"""

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
BASE_MODEL = "fabriceyhc/bert-base-uncased-imdb"   # our existing fine-tuned BERT
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
CHECKPOINT_DIR = os.path.join(HERE, "checkpoints", "best_model")

MAX_LENGTH = 256          # RT snippets are short; 256 is plenty
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1     # fraction of total steps used for LR warm-up
MAX_GRAD_NORM = 1.0       # gradient clipping
SEED = 42

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ReviewDataset(Dataset):
    """Tokenizes (text, label) pairs lazily for the DataLoader."""

    def __init__(self, csv_path, tokenizer):
        df = pd.read_csv(csv_path)
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
    """One pass over `loader`. Trains if optimizer is given, else evaluates.

    Returns (mean_loss, accuracy)."""
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, correct, seen = 0.0, 0, 0

    for step, batch in enumerate(loader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.set_grad_enabled(training):
            out = model(**batch)
            loss = out.loss
        if training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
        total_loss += loss.item() * batch["labels"].size(0)
        correct += (out.logits.argmax(-1) == batch["labels"]).sum().item()
        seen += batch["labels"].size(0)
        if training and step % 50 == 0:
            print(f"    step {step:>4}/{len(loader)}  loss {loss.item():.4f}", end="\r")
    return total_loss / seen, correct / seen


def main():
    set_seed()
    print(f"Device: {DEVICE}\nBase model: {BASE_MODEL}\n")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2).to(DEVICE)

    train_ds = ReviewDataset(os.path.join(DATA_DIR, "train.csv"), tokenizer)
    val_ds = ReviewDataset(os.path.join(DATA_DIR, "validation.csv"), tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}\n")

    total_steps = len(train_loader) * EPOCHS
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * WARMUP_FRACTION), total_steps)

    best_val_loss = float("inf")
    history = []
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, scheduler)
        val_loss, val_acc = run_epoch(model, val_loader)
        history.append((epoch, train_loss, val_loss, val_acc))
        print(f"Epoch {epoch}/{EPOCHS}  "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
              f"{(time.time()-t0)/60:.1f} min")

        # keep only the best checkpoint (lowest validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            model.save_pretrained(CHECKPOINT_DIR)
            tokenizer.save_pretrained(CHECKPOINT_DIR)
            print(f"  ↳ new best (val loss {val_loss:.4f}) saved → {CHECKPOINT_DIR}")

    # ── Phase summary ─────────────────────────────────────────────────────────
    print("\nValidation-loss history: "
          + ", ".join(f"ep{e}: {v:.4f}" for e, _, v, _ in history))
    saved = os.path.isfile(os.path.join(CHECKPOINT_DIR, "config.json"))
    print("✅ PASSED — fine-tuning finished, best checkpoint saved" if saved
          else "❌ FAILED — no checkpoint was saved")


if __name__ == "__main__":
    main()
