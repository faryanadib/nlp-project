import math
import os
import random
import sys

# Reuse evaluation functions (Week02) and text processing (Week03)
sys.path.append(os.path.join(os.path.dirname(__file__), "../Week02"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../Week03"))

from Eval import accuracy, precision, recall, f1_score, confusion_matrix
from calssifier import load_imdb, tokenize


# ══════════════════════════════════════════════════════════════
#  Build Vocabulary
#  Assigns a unique index to every word seen in training data.
#  Example: {"good": 0, "bad": 1, "movie": 2, ...}
#
#  Same logic as Week04 — copied here because homework04.py
#  has no __main__ guard and runs its full pipeline on import.
# ══════════════════════════════════════════════════════════════

def build_vocab(data):
    vocab = {}
    for text, _ in data:
        for word in tokenize(text):
            if word not in vocab:
                vocab[word] = len(vocab)   # assign the next available index
    return vocab


# ══════════════════════════════════════════════════════════════
#  Vectorize — convert text to a numerical vector (Bag of Words)
#  Uses a sparse dict {word_index: 1} to save memory.
#  Only stores words that actually appear — zeros are skipped.
#  Example: "good movie" → {0: 1, 2: 1}
# ══════════════════════════════════════════════════════════════

def vectorize(text, vocab):
    vec = {}
    for word in tokenize(text):
        if word in vocab:
            vec[vocab[word]] = 1   # mark this word as present
    return vec


# ══════════════════════════════════════════════════════════════
#  Train/Test Split — 80% train, 20% test
#  seed=42 ensures the same shuffle every run (reproducible).
# ══════════════════════════════════════════════════════════════

def train_test_split(texts, labels, test_size=0.2, seed=42):
    combined = list(zip(texts, labels))
    random.seed(seed)
    random.shuffle(combined)                           # randomize order
    split = int(len(combined) * (1 - test_size))      # index where test begins
    train = combined[:split]
    test  = combined[split:]
    tr_t, tr_l = zip(*train)
    te_t, te_l = zip(*test)
    return list(tr_t), list(te_t), list(tr_l), list(te_l)


# ══════════════════════════════════════════════════════════════
#  Sigmoid Function — the core of Logistic Regression
#
#  Squashes any real number z into a probability in [0, 1]:
#
#      sigmoid(z) = 1 / (1 + e^(-z))
#
#  Large z  → probability close to 1  (confident positive)
#  Small z  → probability close to 0  (confident negative)
#  z = 0    → probability = 0.5       (uncertain)
#
#  Clamping z to [-500, 500] prevents overflow in math.exp.
# ══════════════════════════════════════════════════════════════

def sigmoid(z):
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))


# ══════════════════════════════════════════════════════════════
#  Forward Pass — compute predicted probability for one text
#
#  Step 1: z = w·x + b
#          Weighted sum of input features (same idea as Perceptron).
#          w.get(i, 0) returns 0 if this word has no weight yet.
#
#  Step 2: p = sigmoid(z)
#          Convert the raw score into a probability.
# ══════════════════════════════════════════════════════════════

def predict_proba(x, w, b):
    z = sum(w.get(i, 0) * x[i] for i in x) + b   # weighted sum + bias
    return sigmoid(z)                              # convert to probability


def predict(x, w, b):
    # Threshold at 0.5: above → positive (1), below → negative (0)
    return 1 if predict_proba(x, w, b) >= 0.5 else 0


# ══════════════════════════════════════════════════════════════
#  Training — Gradient Descent
#
#  Key difference from Perceptron:
#  → Perceptron only updates weights when it makes a mistake (binary)
#  → LR updates on every example, proportional to how wrong it was
#
#  Update rule per example:
#      error  = p - y              (predicted probability minus true label)
#      w[i]  -= lr * error * x[i] (nudge each weight toward correct answer)
#      b     -= lr * error        (nudge bias)
#
#  Large error → large update.  Small error → small update.
#
#  Cross-entropy loss is printed each epoch to track learning progress.
#  Lower loss = model is learning to fit the training data better.
# ══════════════════════════════════════════════════════════════

def train(train_texts, train_labels, vocab, epochs=10, lr=0.1):
    w = {}    # weights — all start at zero (sparse dict)
    b = 0.0   # bias — starts at zero

    for epoch in range(epochs):
        total_loss = 0.0

        for text, y in zip(train_texts, train_labels):
            x     = vectorize(text, vocab)      # convert text to feature vector
            p     = predict_proba(x, w, b)      # predict probability
            error = p - y                       # how wrong were we?

            # cross-entropy loss — used only for logging progress
            p_safe      = max(1e-10, min(1 - 1e-10, p))   # avoid log(0)
            total_loss += -(y * math.log(p_safe) + (1 - y) * math.log(1 - p_safe))

            # update the weight of every word that appeared in this text
            for i in x:
                w[i] = w.get(i, 0) - lr * error * x[i]

            # update bias
            b = b - lr * error

        avg_loss = total_loss / len(train_texts)
        print(f"  Epoch {epoch+1:2d} | loss = {avg_loss:.4f}")

    return w, b


# ══════════════════════════════════════════════════════════════
#  Top Word Weights — what signal the model learned per word
#
#  Sorted descending by weight:
#    highest positive weights → strongest POSITIVE signal words
#    lowest  negative weights → strongest NEGATIVE signal words
#
#  Used for presentation and understanding model behavior.
# ══════════════════════════════════════════════════════════════

def top_words(vocab, w, n=10):
    idx_to_word = {idx: word for word, idx in vocab.items()}
    valid    = [(idx, weight) for idx, weight in w.items() if idx in idx_to_word]
    sorted_w = sorted(valid, key=lambda x: x[1], reverse=True)

    print("\n===== Top Word Weights (Logistic Regression) =====")
    print("  POSITIVE signal words:")
    for idx, weight in sorted_w[:n]:
        print(f"    {idx_to_word[idx]:25s}  {weight:+.4f}")
    print("  NEGATIVE signal words:")
    for idx, weight in sorted_w[-n:]:
        print(f"    {idx_to_word[idx]:25s}  {weight:+.4f}")


# ══════════════════════════════════════════════════════════════
#  Evaluate and show error samples (uses Week02 Eval.py)
# ══════════════════════════════════════════════════════════════

def evaluate(test_texts, test_labels, vocab, w, b):
    # run prediction on all test samples
    predictions = [predict(vectorize(t, vocab), w, b) for t in test_texts]

    tp, fp, fn, tn = confusion_matrix(test_labels, predictions)

    print("\n===== Results =====")
    print(f"  Accuracy  : {accuracy(test_labels, predictions):.4f}")
    print(f"  Precision : {precision(test_labels, predictions):.4f}")
    print(f"  Recall    : {recall(test_labels, predictions):.4f}")
    print(f"  F1 Score  : {f1_score(test_labels, predictions):.4f}")
    print(f"\n  Confusion Matrix")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    # collect examples where the model was wrong
    label_map = {1: "positive", 0: "negative"}
    errors = [(t, tr, pr) for t, tr, pr in zip(test_texts, test_labels, predictions) if tr != pr]

    print(f"\n===== Error Analysis =====")
    print(f"  Total errors: {len(errors)} / {len(test_labels)}")
    for text, true, pred in errors[:5]:
        print(f"\n  True: {label_map[true]:8s} | Predicted: {label_map[pred]}")
        print(f"  \"{text[:110]}...\"")

    return predictions


# ══════════════════════════════════════════════════════════════
#  Main — full pipeline
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    base         = os.path.dirname(os.path.abspath(__file__))
    dataset_file = os.path.join(base, "../Data/IMDB Dataset.csv")

    print("Loading dataset (50000 samples)...")
    raw    = load_imdb(dataset_file, limit=50000)
    texts  = [t for t, _ in raw]
    labels = [1 if l == "positive" else 0 for _, l in raw]   # positive=1, negative=0

    print("Splitting 80/20 (seed=42)...")
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels)
    print(f"Train: {len(train_texts)}  |  Test: {len(test_texts)}")

    # vocab is built from training data only — test data must stay unseen
    print("\nBuilding vocabulary...")
    vocab = build_vocab(list(zip(train_texts, train_labels)))
    print(f"Vocabulary size: {len(vocab)}")

    print("\nTraining Logistic Regression (10 epochs, lr=0.1)...")
    w, b = train(train_texts, train_labels, vocab, epochs=10, lr=0.1)

    evaluate(test_texts, test_labels, vocab, w, b)

    print("\nTop word weights learned by Logistic Regression...")
    top_words(vocab, w)
