"""
Group Session 1 — TF-IDF + Bigrams
Baseline: our own Logistic Regression from Week05 (BoW binary, from scratch)
Goal: show how TF-IDF + bigrams improves over it using sklearn.
"""

import os, sys, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "../Week02"))
sys.path.append(os.path.join(HERE, "../Week03"))
sys.path.append(os.path.join(HERE, "../Week05"))

from Eval import accuracy, precision, recall, f1_score as nb_f1
from calssifier import load_imdb, tokenize
from logistic_regression import build_vocab, vectorize, train as lr_train, predict as lr_predict

DATA = os.path.join(HERE, "../Data/IMDB Dataset.csv")
SEED = 42


def split(data, test=0.2, seed=SEED):
    import random
    random.seed(seed)
    random.shuffle(data)
    cut = int(len(data) * (1 - test))
    return data[:cut], data[cut:]


def main():
    print("Loading data...")
    raw = load_imdb(DATA, limit=50000)
    train_raw, test_raw = split(raw)

    train_texts = [t for t, _ in train_raw]
    test_texts  = [t for t, _ in test_raw]
    train_labels_str = [l for _, l in train_raw]
    test_labels_str  = [l for _, l in test_raw]
    train_labels = [1 if l == "positive" else 0 for l in train_labels_str]
    test_labels  = [1 if l == "positive" else 0 for l in test_labels_str]

    print(f"Train: {len(train_texts):,}   Test: {len(test_texts):,}\n")

    results = []

    # ── Baseline: our from-scratch LR (Week05) ────────────────────────────
    print("Training from-scratch LR (Week05 baseline)...")
    t0    = time.time()
    vocab = build_vocab(train_raw)
    w, b  = lr_train(train_texts, train_labels, vocab, epochs=10, lr=0.1)
    preds = [lr_predict(vectorize(t, vocab), w, b) for t in test_texts]
    elapsed = time.time() - t0
    results.append(("LR scratch  BoW binary",
                     accuracy(test_labels, preds), precision(test_labels, preds),
                     recall(test_labels, preds),   nb_f1(test_labels, preds), elapsed))

    # ── sklearn experiments ───────────────────────────────────────────────
    experiments = [
        ("TF-IDF unigrams + LR",  (1,1), LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)),
        ("TF-IDF unigrams + SVM", (1,1), LinearSVC(max_iter=2000, C=1.0, random_state=SEED)),
        ("TF-IDF bigrams  + LR",  (1,2), LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)),
        ("TF-IDF bigrams  + SVM", (1,2), LinearSVC(max_iter=2000, C=1.0, random_state=SEED)),
    ]

    for name, ngram, clf in experiments:
        print(f"Training {name}...")
        t0  = time.time()
        vec = TfidfVectorizer(ngram_range=ngram, min_df=2, sublinear_tf=True)
        clf.fit(vec.fit_transform(train_texts), train_labels)
        preds   = clf.predict(vec.transform(test_texts))
        elapsed = time.time() - t0
        results.append((name,
                         accuracy_score(test_labels, preds),
                         precision_score(test_labels, preds, zero_division=0),
                         recall_score(test_labels, preds, zero_division=0),
                         f1_score(test_labels, preds, zero_division=0),
                         elapsed))

    # ── Results table ─────────────────────────────────────────────────────
    base_f1 = results[0][4]
    print(f"\n{'─'*70}")
    print(f"  {'Setup':<28}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Time':>6}")
    print(f"{'─'*70}")
    for name, acc, prec, rec, f1, t in results:
        tag = "  (baseline)" if name == results[0][0] else f"  (+{(f1-base_f1)*100:.1f}pp)"
        print(f"  {name:<28}  {acc:.4f}  {prec:.4f}  {rec:.4f}  {f1:.4f}  {t:5.1f}s{tag}")
    print(f"{'─'*70}")


if __name__ == "__main__":
    main()
