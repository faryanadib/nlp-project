import csv
import math
import os
import random
import sys

# Evaluation functions (Week02), tokenizer (Week03), LR model (Week05)
sys.path.append(os.path.join(os.path.dirname(__file__), "../Week02"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../Week03"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../Week05"))

from Eval import accuracy, precision, recall, f1_score, confusion_matrix
from calssifier import tokenize
from logistic_regression import build_vocab, vectorize, predict as lr_predict, train as lr_train


MAX_SAMPLES = 50000   # total number of reviews loaded from the dataset
EPOCHS      = 10      # number of passes over training data for Perceptron and LR
SEED        = 42      # fixed seed ensures the same shuffle every run


# ══════════════════════════════════════════════════════════════
#  Data Loading — called once, shared across all three models
#  All models must train and test on the exact same data.
# ══════════════════════════════════════════════════════════════

def load_imdb(path, max_samples=MAX_SAMPLES):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_samples:
                break
            text      = row["review"].strip()
            sentiment = row["sentiment"].strip().lower()
            # skip empty rows or rows with unexpected label values
            if text and sentiment in ("positive", "negative"):
                texts.append(text)
                labels.append(sentiment)
    return texts, labels


def train_test_split(texts, labels, test_size=0.2, seed=SEED):
    combined = list(zip(texts, labels))
    random.seed(seed)
    random.shuffle(combined)                           # shuffle with fixed seed
    split = int(len(combined) * (1 - test_size))      # 80% train, 20% test
    train = combined[:split]
    test  = combined[split:]
    tr_t, tr_l = zip(*train)
    te_t, te_l = zip(*test)
    return list(tr_t), list(te_t), list(tr_l), list(te_l)


def to_int(labels):
    # Eval.py expects numeric labels — convert strings to integers
    # "positive" → 1  |  "negative" → 0
    return [1 if l == "positive" else 0 for l in labels]


# ══════════════════════════════════════════════════════════════
#  Model 1 — Naive Bayes
#  Logic from Week03. Rewritten here to avoid naming conflicts
#  (train, predict) with the other two models.
# ══════════════════════════════════════════════════════════════

def nb_train(train_texts, train_labels):
    class_counts, word_counts, total_words, vocab = {}, {}, {}, set()

    for text, label in zip(train_texts, train_labels):
        # count how many documents belong to each class
        class_counts[label] = class_counts.get(label, 0) + 1
        if label not in word_counts:
            word_counts[label] = {}
            total_words[label] = 0
        # count word occurrences per class
        for word in tokenize(text):
            vocab.add(word)
            word_counts[label][word] = word_counts[label].get(word, 0) + 1
            total_words[label] += 1

    total_docs = len(train_texts)

    # P(class) = number of documents in class / total documents
    priors     = {l: class_counts[l] / total_docs for l in class_counts}
    vocab_size = len(vocab)

    # P(word | class) with Laplace smoothing to avoid multiplying by zero
    # Formula: (count + 1) / (total_words_in_class + vocab_size)
    likelihoods = {}
    for label in word_counts:
        likelihoods[label] = {
            word: (word_counts[label].get(word, 0) + 1) / (total_words[label] + vocab_size)
            for word in vocab
        }

    return priors, likelihoods, total_words, vocab_size


def nb_predict(text, priors, likelihoods, total_words, vocab_size):
    best_label, best_score = None, None
    for label in priors:
        # start with log P(class) — log space avoids floating point underflow
        score = math.log(priors[label])
        for word in tokenize(text):
            if word in likelihoods[label]:
                score += math.log(likelihoods[label][word])
            else:
                # word not seen in training — apply smoothed probability
                score += math.log(1 / (total_words[label] + vocab_size))
        # class with the highest log-probability wins
        if best_score is None or score > best_score:
            best_score = score
            best_label = label
    return best_label


# ══════════════════════════════════════════════════════════════
#  Model 2 — Perceptron
#  Logic from Week04. Functions prefixed with perc_ to avoid
#  naming conflicts with the LR functions imported above.
# ══════════════════════════════════════════════════════════════

def perc_build_vocab(texts):
    # build vocabulary from text only (labels not needed here)
    vocab = {}
    for text in texts:
        for word in tokenize(text):
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


def perc_vectorize(text, vocab):
    # same sparse-dict vectorization as Week04
    vec = {}
    for word in tokenize(text):
        if word in vocab:
            vec[vocab[word]] = 1
    return vec


def perc_train(train_texts, train_labels, epochs=EPOCHS):
    # Perceptron uses +1 / -1 labels (not 1 / 0)
    label_map = {"positive": 1, "negative": -1}
    vocab = perc_build_vocab(train_texts)
    w, b  = {}, 0   # weights and bias both start at zero

    def _predict(x):
        # positive score → positive class, negative score → negative class
        score = sum(w.get(i, 0) * x[i] for i in x) + b
        return 1 if score > 0 else -1

    for epoch in range(epochs):
        errors = 0
        for text, label in zip(train_texts, train_labels):
            y = label_map[label]
            x = perc_vectorize(text, vocab)
            if _predict(x) != y:
                # update only on mistakes — key difference from LR
                for i in x:
                    w[i] = w.get(i, 0) + y
                b += y
                errors += 1
        print(f"  Epoch {epoch+1:2d} | errors = {errors}")

    return vocab, w, b


def perc_predict(text, vocab, w, b):
    x     = perc_vectorize(text, vocab)
    score = sum(w.get(i, 0) * x[i] for i in x) + b
    return "positive" if score > 0 else "negative"


# ══════════════════════════════════════════════════════════════
#  Word Weight Helpers — for presentation slide content
#
#  Perceptron weights are integers (accumulated vote counts).
#  LR weights are floats (gradient-descent values).
#  Both are stored as sparse dicts {word_index: weight}.
# ══════════════════════════════════════════════════════════════

def print_top_words_perc(perc_vocab, perc_w, n=10):
    idx_to_word = {idx: word for word, idx in perc_vocab.items()}
    valid    = [(idx, w) for idx, w in perc_w.items() if idx in idx_to_word]
    sorted_w = sorted(valid, key=lambda x: x[1], reverse=True)

    print(f"\n{'─'*44}")
    print("  Perceptron — Top Word Weights")
    print(f"{'─'*44}")
    print("  POSITIVE signal words (integer weights):")
    for idx, w in sorted_w[:n]:
        print(f"    {idx_to_word[idx]:25s}  {w:+d}")
    print("  NEGATIVE signal words (integer weights):")
    for idx, w in sorted_w[-n:]:
        print(f"    {idx_to_word[idx]:25s}  {w:+d}")


def print_top_words_lr(lr_vocab, lr_w, n=10):
    idx_to_word = {idx: word for word, idx in lr_vocab.items()}
    valid    = [(idx, w) for idx, w in lr_w.items() if idx in idx_to_word]
    sorted_w = sorted(valid, key=lambda x: x[1], reverse=True)

    print(f"\n{'─'*44}")
    print("  Logistic Regression — Top Word Weights")
    print(f"{'─'*44}")
    print("  POSITIVE signal words (float weights):")
    for idx, w in sorted_w[:n]:
        print(f"    {idx_to_word[idx]:25s}  {w:+.4f}")
    print("  NEGATIVE signal words (float weights):")
    for idx, w in sorted_w[-n:]:
        print(f"    {idx_to_word[idx]:25s}  {w:+.4f}")


# ══════════════════════════════════════════════════════════════
#  Cross-Model Error Analysis — finds real examples where models
#  agree or disagree, for use in presentation slides.
#
#  Three cases:
#    CASE 1 — NB wrong + Perceptron wrong + LR correct
#             Shows LR's advantage over the other two.
#    CASE 2 — All three wrong
#             Shows the shared BoW limitation (negation, sarcasm…).
#    CASE 3 — Only NB correct (Perceptron and LR both wrong)
#             Shows NB's high-precision cautious behavior.
# ══════════════════════════════════════════════════════════════

def cross_model_analysis(test_texts, y_true, nb_preds, perc_preds, lr_preds, n=4):
    label_map = {1: "positive", 0: "negative"}

    case1 = [                         # NB ✗  Perc ✗  LR ✓
        (t, true, nb, perc, lr)
        for t, true, nb, perc, lr
        in zip(test_texts, y_true, nb_preds, perc_preds, lr_preds)
        if true != nb and true != perc and true == lr
    ]
    case2 = [                         # NB ✗  Perc ✗  LR ✗  (all wrong)
        (t, true, nb, perc, lr)
        for t, true, nb, perc, lr
        in zip(test_texts, y_true, nb_preds, perc_preds, lr_preds)
        if true != nb and true != perc and true != lr
    ]
    case3 = [                         # NB ✓  Perc ✗  LR ✗
        (t, true, nb, perc, lr)
        for t, true, nb, perc, lr
        in zip(test_texts, y_true, nb_preds, perc_preds, lr_preds)
        if true == nb and true != perc and true != lr
    ]

    print(f"\n\n{'═'*60}")
    print("   CROSS-MODEL ERROR ANALYSIS  (for presentation slides)")
    print(f"{'═'*60}")

    print(f"\n[CASE 1] NB wrong + Perceptron wrong + LR correct: {len(case1)} reviews")
    print("→ Use these for slide: 'Where LR outperforms the others'")
    for t, true, nb, perc, lr in case1[:n]:
        print(f"\n  True={label_map[true]}  |  NB={label_map[nb]} ✗  "
              f"|  Perc={label_map[perc]} ✗  |  LR={label_map[lr]} ✓")
        print(f"  Review: \"{t[:220]}\"")

    print(f"\n[CASE 2] All three models wrong: {len(case2)} reviews")
    print("→ Use these for slide: 'Shared BoW limitation'")
    for t, true, nb, perc, lr in case2[:n]:
        print(f"\n  True={label_map[true]}  |  NB={label_map[nb]} ✗  "
              f"|  Perc={label_map[perc]} ✗  |  LR={label_map[lr]} ✗")
        print(f"  Review: \"{t[:220]}\"")

    print(f"\n[CASE 3] Only NB correct (Perceptron + LR wrong): {len(case3)} reviews")
    print("→ Use these for slide: 'NB high-precision cautious behavior'")
    for t, true, nb, perc, lr in case3[:n]:
        print(f"\n  True={label_map[true]}  |  NB={label_map[nb]} ✓  "
              f"|  Perc={label_map[perc]} ✗  |  LR={label_map[lr]} ✗")
        print(f"  Review: \"{t[:220]}\"")


# ══════════════════════════════════════════════════════════════
#  Result Display Helpers
# ══════════════════════════════════════════════════════════════

def print_results(model_name, y_true, y_pred):
    # print all four metrics + confusion matrix for one model
    tp, fp, fn, tn = confusion_matrix(y_true, y_pred)
    print(f"\n{'━'*44}")
    print(f"  {model_name}")
    print(f"{'━'*44}")
    print(f"  Accuracy  : {accuracy(y_true, y_pred):.4f}")
    print(f"  Precision : {precision(y_true, y_pred):.4f}")
    print(f"  Recall    : {recall(y_true, y_pred):.4f}")
    print(f"  F1 Score  : {f1_score(y_true, y_pred):.4f}")
    print(f"  Confusion → TP={tp}  FP={fp}  FN={fn}  TN={tn}")


def print_error_samples(model_name, test_texts, y_true, y_pred, n=3):
    # find and display examples where the model predicted incorrectly
    label_map = {1: "positive", 0: "negative"}
    errors = [
        (text, true, pred)
        for text, true, pred in zip(test_texts, y_true, y_pred)
        if true != pred   # only wrong predictions
    ]
    print(f"\n  ── {model_name}: {len(errors)} errors ──")
    for text, true, pred in errors[:n]:
        print(f"  True: {label_map[true]:8s} | Pred: {label_map[pred]:8s}")
        print(f"  \"{text[:100]}...\"")


# ══════════════════════════════════════════════════════════════
#  Main — run all three models on the exact same data
# ══════════════════════════════════════════════════════════════

def main():
    base         = os.path.dirname(os.path.abspath(__file__))
    dataset_file = os.path.join(base, "../Data/IMDB Dataset.csv")

    # ── load and split once — all models share this data ──────
    print(f"Loading {MAX_SAMPLES} samples from IMDB dataset...")
    texts, labels = load_imdb(dataset_file)

    print("Splitting 80/20 (seed=42)...")
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels)
    print(f"Train: {len(train_texts)}  |  Test: {len(test_texts)}")

    # shared ground truth in numeric format — converted once for all models
    test_labels_int = to_int(test_labels)

    # ── Model 1: Naive Bayes ───────────────────────────────────
    print("\n[1/3] Training Naive Bayes...")
    priors, likelihoods, total_words, vocab_size = nb_train(train_texts, train_labels)
    nb_preds     = [nb_predict(t, priors, likelihoods, total_words, vocab_size) for t in test_texts]
    nb_preds_int = to_int(nb_preds)   # convert "positive"/"negative" → 1/0

    # ── Model 2: Perceptron ────────────────────────────────────
    print(f"\n[2/3] Training Perceptron ({EPOCHS} epochs)...")
    perc_vocab, perc_w, perc_b = perc_train(train_texts, train_labels)
    perc_preds     = [perc_predict(t, perc_vocab, perc_w, perc_b) for t in test_texts]
    perc_preds_int = to_int(perc_preds)

    # ── Model 3: Logistic Regression ──────────────────────────
    print("\n[3/3] Training Logistic Regression (10 epochs, lr=0.1)...")
    # vocab built from training data only — labels are ignored inside build_vocab
    lr_vocab   = build_vocab(list(zip(train_texts, train_labels)))
    # LR expects integer labels (1/0), not strings
    lr_w, lr_b = lr_train(train_texts, to_int(train_labels), lr_vocab, epochs=10, lr=0.1)
    lr_preds   = [lr_predict(vectorize(t, lr_vocab), lr_w, lr_b) for t in test_texts]

    # ══════════════════════════════════════════════════════════
    #  Comparison results — side by side
    # ══════════════════════════════════════════════════════════
    print("\n\n" + "═"*44)
    print(f"   MODEL COMPARISON  (n={MAX_SAMPLES}, 80/20)")
    print("═"*44)

    print_results("Naive Bayes",         test_labels_int, nb_preds_int)
    print_results("Perceptron",          test_labels_int, perc_preds_int)
    print_results("Logistic Regression", test_labels_int, lr_preds)

    # ══════════════════════════════════════════════════════════
    #  Error analysis — sample wrong predictions per model
    # ══════════════════════════════════════════════════════════
    print("\n\n" + "═"*44)
    print("       ERROR ANALYSIS SAMPLES")
    print("═"*44)
    print_error_samples("Naive Bayes",         test_texts, test_labels_int, nb_preds_int)
    print_error_samples("Perceptron",          test_texts, test_labels_int, perc_preds_int)
    print_error_samples("Logistic Regression", test_texts, test_labels_int, lr_preds)

    # ══════════════════════════════════════════════════════════
    #  Word weight analysis — real learned weights per model
    # ══════════════════════════════════════════════════════════
    print("\n\n" + "═"*44)
    print("   WORD WEIGHT ANALYSIS")
    print("═"*44)
    print_top_words_perc(perc_vocab, perc_w)
    print_top_words_lr(lr_vocab, lr_w)

    # ══════════════════════════════════════════════════════════
    #  Cross-model error analysis — real examples from dataset
    # ══════════════════════════════════════════════════════════
    cross_model_analysis(test_texts, test_labels_int, nb_preds_int, perc_preds_int, lr_preds)


if __name__ == "__main__":
    main()
