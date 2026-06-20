"""
Session 1 (Group) — TF-IDF + Bigrams vs BoW Baseline
Compare our Week03 Naive Bayes against sklearn TF-IDF classifiers.
"""

import csv, math, os, random, sys, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append(os.path.join(os.path.dirname(__file__), "../Week02"))
from Eval import accuracy, precision, recall, f1_score as nb_f1, confusion_matrix

DATA = os.path.join(os.path.dirname(__file__), "../Data/IMDB Dataset.csv")
SEED = 42


# ── Data ─────────────────────────────────────────────────────────────────────

def load_imdb(path, n=50_000):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= n: break
            t, s = row["review"].strip(), row["sentiment"].strip().lower()
            if t and s in ("positive", "negative"):
                texts.append(t)
                labels.append(1 if s == "positive" else 0)
    return texts, labels


def split(texts, labels, test=0.2):
    data = list(zip(texts, labels))
    random.seed(SEED)
    random.shuffle(data)
    cut = int(len(data) * (1 - test))
    tr, te = data[:cut], data[cut:]
    return [t for t,_ in tr], [t for t,_ in te], [l for _,l in tr], [l for _,l in te]


# ── Baseline: Naive Bayes from scratch (Week03) ───────────────────────────────

def tokenize(text):
    text, clean, inside = text.lower(), [], False
    for ch in text:
        if ch == "<": inside = True
        elif ch == ">": inside = False
        elif not inside: clean.append(ch)
    return "".join(clean).split()


def nb_train(train_texts, train_labels):
    cc, wc, tw, vocab = {}, {}, {}, set()
    for text, label in zip(train_texts, train_labels):
        cc[label] = cc.get(label, 0) + 1
        wc.setdefault(label, {}); tw.setdefault(label, 0)
        for w in tokenize(text):
            vocab.add(w)
            wc[label][w] = wc[label].get(w, 0) + 1
            tw[label] += 1
    n, vs = len(train_texts), len(vocab)
    priors = {l: cc[l] / n for l in cc}
    lh = {l: {w: (wc[l].get(w, 0) + 1) / (tw[l] + vs) for w in vocab} for l in wc}
    return priors, lh, tw, vs


def nb_predict(text, priors, lh, tw, vs):
    best_l, best_s = None, None
    for l in priors:
        s = math.log(priors[l])
        for w in tokenize(text):
            s += math.log(lh[l][w]) if w in lh[l] else math.log(1 / (tw[l] + vs))
        if best_s is None or s > best_s: best_s, best_l = s, l
    return best_l


# ── Run ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading data..."); texts, labels = load_imdb(DATA)
    tr_t, te_t, tr_l, te_l = split(texts, labels)
    print(f"Train: {len(tr_t):,}  Test: {len(te_t):,}\n")

    results = []

    # Baseline
    print("Training Naive Bayes (baseline)...")
    t0 = time.time()
    priors, lh, tw, vs = nb_train(tr_t, tr_l)
    preds = [nb_predict(t, priors, lh, tw, vs) for t in te_t]
    elapsed = time.time() - t0
    results.append(("NB + BoW (scratch)",
                     accuracy(te_l, preds), precision(te_l, preds),
                     recall(te_l, preds),   nb_f1(te_l, preds), elapsed))

    # sklearn experiments
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
        clf.fit(vec.fit_transform(tr_t), tr_l)
        preds   = clf.predict(vec.transform(te_t))
        elapsed = time.time() - t0
        results.append((name,
                         accuracy_score(te_l, preds),
                         precision_score(te_l, preds, zero_division=0),
                         recall_score(te_l, preds, zero_division=0),
                         f1_score(te_l, preds, zero_division=0),
                         elapsed))

    # Table
    print(f"\n{'─'*72}")
    print(f"  {'Setup':<30}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Time':>6}")
    print(f"{'─'*72}")
    base_f1 = results[0][4]
    for name, acc, prec, rec, f1, t in results:
        gain = f"  (+{(f1-base_f1)*100:.1f}pp)" if f1 > base_f1 else "  (baseline)"
        print(f"  {name:<30}  {acc:.4f}  {prec:.4f}  {rec:.4f}  {f1:.4f}  {t:5.1f}s{gain}")
    print(f"{'─'*72}")


if __name__ == "__main__":
    main()
