"""
Group Session 1 — BERT Embeddings + Classifier
Use pre-trained DistilBERT as a feature extractor (no fine-tuning).
Each review → [CLS] token vector (768-dim) → sklearn classifier.

Install: pip install transformers torch scikit-learn
"""

import csv, os, random, sys, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Data/IMDB Dataset.csv")
# ── Model selection ───────────────────────────────────────────────────────────
# Uncomment the model you want to use. Results and embeddings are saved
# automatically with the model name, so switching won't overwrite previous runs.
#
# OPTION 1 — Generic DistilBERT (no sentiment knowledge)
#   No domain adaptation. Weakest results (F1 ≈ 0.86).
#   Use as a "zero-knowledge" baseline.
# MODEL = "distilbert-base-uncased"
#
# OPTION 2 — DistilBERT fine-tuned on SST-2 (Rotten Tomatoes sentiment)
#   Knows sentiment but trained on a different movie review dataset.
#   Mid-range results (F1 ≈ 0.89).
# MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
#
# OPTION 3 — DistilBERT fine-tuned on IMDB
#   Lightweight model trained on the same domain as our task.
#   Good results (F1 ≈ 0.93).
# MODEL = "lvwerra/distilbert-imdb"
#
# OPTION 4 — Full BERT fine-tuned on IMDB  ← current selection
#   Largest model, trained on the same domain. Best results (F1 ≈ 0.95).
#   ~400MB download.
MODEL  = "fabriceyhc/bert-base-uncased-imdb"
SEED   = 42
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")


def load_imdb(path, n=10_000):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= n: break
            t, s = row["review"].strip(), row["sentiment"].strip().lower()
            if t and s in ("positive", "negative"):
                texts.append(t)
                labels.append(1 if s == "positive" else 0)
    return texts, labels


def split(texts, labels, test=0.2, seed=SEED):
    data = list(zip(texts, labels))
    random.seed(seed); random.shuffle(data)
    cut = int(len(data) * (1 - test))
    tr, te = data[:cut], data[cut:]
    return [t for t,_ in tr], [t for t,_ in te], [l for _,l in tr], [l for _,l in te]


def get_embeddings(texts, tokenizer, model, batch_size=32):
    """Pass texts through BERT, return [CLS] token vectors — no gradient updates."""
    model.eval()
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch   = texts[i : i + batch_size]
        inputs  = tokenizer(batch, truncation=True, padding=True,
                            max_length=256, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        # [CLS] token is the first token — represents the whole sequence
        cls_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_vectors)
        if (i // batch_size) % 10 == 0:
            print(f"  {i + len(batch)}/{len(texts)} reviews embedded...", end="\r")
    print()
    return np.vstack(all_embeddings)


def main():
    print(f"Device : {DEVICE}")
    print(f"Model  : {MODEL}\n")

    print("Loading data...")
    texts, labels = load_imdb(DATA, n=10_000)
    tr_t, te_t, tr_l, te_l = split(texts, labels)
    print(f"Train: {len(tr_t):,}   Test: {len(te_t):,}\n")

    print("Loading pre-trained BERT...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model     = AutoModel.from_pretrained(MODEL).to(DEVICE)

    print("Extracting embeddings (train)...")
    t0       = time.time()
    X_train  = get_embeddings(tr_t, tokenizer, model)

    print("Extracting embeddings (test)...")
    X_test   = get_embeddings(te_t, tokenizer, model)
    embed_time = time.time() - t0
    print(f"Embedding done in {embed_time/60:.1f} min — shape: {X_train.shape}\n")

    # ── Save embeddings ───────────────────────────────────────────────────
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(OUT, exist_ok=True)
    model_slug = MODEL.replace("/", "_")
    np.savez(os.path.join(OUT, f"embeddings_{model_slug}.npz"),
             X_train=X_train, X_test=X_test,
             y_train=np.array(tr_l), y_test=np.array(te_l))
    print(f"Embeddings saved → embeddings_{model_slug}.npz\n")

    results = []
    for name, clf in [("BERT embeddings + LR",  LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)),
                      ("BERT embeddings + SVM", LinearSVC(max_iter=2000, C=1.0, random_state=SEED))]:
        print(f"Training {name}...")
        t0    = time.time()
        clf.fit(X_train, tr_l)
        preds = clf.predict(X_test)
        results.append((name,
                        accuracy_score(te_l, preds),
                        precision_score(te_l, preds, zero_division=0),
                        recall_score(te_l, preds, zero_division=0),
                        f1_score(te_l, preds, zero_division=0),
                        time.time() - t0))

    # ── Print table ───────────────────────────────────────────────────────
    ref = [
        ("LR scratch  BoW binary", None, None, None, 0.8856, None, "Week05 baseline"),
        ("TF-IDF bigrams + SVM",   None, None, None, 0.9230, None, "Session 1 best"),
    ]
    header = f"\n{'─'*68}\n  {'Setup':<30}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}\n{'─'*68}"
    print(header)
    for name, _, _, _, f1, _, tag in ref:
        print(f"  {name:<30}  {'─':>6}  {'─':>6}  {'─':>6}  {f1:.4f}  ({tag})")
    for name, acc, prec, rec, f1, t in results:
        print(f"  {name:<30}  {acc:.4f}  {prec:.4f}  {rec:.4f}  {f1:.4f}  ← this run")
    print(f"{'─'*68}")

    # ── Save results to CSV ───────────────────────────────────────────────
    csv_path = os.path.join(OUT, f"results_{model_slug}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["setup", "accuracy", "precision", "recall", "f1", "note"])
        writer.writerow(["LR scratch BoW binary", "", "", "", 0.8856, "Week05 baseline"])
        writer.writerow(["TF-IDF bigrams + SVM",  "", "", "", 0.9230, "Session 1 best"])
        for name, acc, prec, rec, f1, t in results:
            writer.writerow([name, round(acc,4), round(prec,4), round(rec,4), round(f1,4), "this run"])
    print(f"Results saved  → bert_results.csv")


if __name__ == "__main__":
    main()
