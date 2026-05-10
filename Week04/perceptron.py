import sys
import os

# ── Imports from previous weeks ───────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), "../Week03"))
from calssifier import tokenize, precision, recall, f1_score


# ─────────────────────────────────────────────────────────────────
# 1. Dataset  (from lecture slides)
# ─────────────────────────────────────────────────────────────────

dataset = [
    ("free money now",  1),
    ("win money now",   1),
    ("call me now",    -1),
    ("let's meet now", -1),
]


# ─────────────────────────────────────────────────────────────────
# 2. Build Vocabulary
# ─────────────────────────────────────────────────────────────────

def build_vocab(data):
    vocab = {}
    for text, _ in data:
        for word in tokenize(text):
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


# ─────────────────────────────────────────────────────────────────
# 3. Vectorize
# ─────────────────────────────────────────────────────────────────

def vectorize(text):
    vec = [0] * len(vocab)
    for word in tokenize(text):
        if word in vocab:
            vec[vocab[word]] = 1
    return vec


# ─────────────────────────────────────────────────────────────────
# 4. Initialize Model
# ─────────────────────────────────────────────────────────────────

vocab = build_vocab(dataset)

w = [0] * len(vocab)
b = 0


# ─────────────────────────────────────────────────────────────────
# 5. Predict
# ─────────────────────────────────────────────────────────────────

def predict(x):
    score = sum(w[i] * x[i] for i in range(len(x))) + b
    return 1 if score > 0 else -1


# ─────────────────────────────────────────────────────────────────
# 6. Update
# ─────────────────────────────────────────────────────────────────

def update(x, y):
    global w, b
    if predict(x) != y:
        w = [w[i] + y * x[i] for i in range(len(w))]
        b = b + y


# ─────────────────────────────────────────────────────────────────
# 7. Training Loop
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    epochs = 10
    for _ in range(epochs):
        for text, label in dataset:
            update(vectorize(text), label)

    # ─────────────────────────────────────────────────────────────
    # 8. Test the Model
    # ─────────────────────────────────────────────────────────────

    tests = [
        "free money",
        "call me",
        "meet now",
    ]
    print("--- Test ---")
    for t in tests:
        print(t, "->", predict(vectorize(t)))

    # ─────────────────────────────────────────────────────────────
    # 9. Inspect Weights
    # ─────────────────────────────────────────────────────────────

    print("\n--- Weights ---")
    for word, i in vocab.items():
        print(word, w[i])

    # ─────────────────────────────────────────────────────────────
    # 10. Evaluation  
    # ─────────────────────────────────────────────────────────────

    y_true = [label for _, label in dataset]
    y_pred = [predict(vectorize(text)) for text, _ in dataset]

    print("\n--- Evaluation ---")
    for label in [1, -1]:
        p = precision(y_true, y_pred, label)
        r = recall(y_true, y_pred, label)
        f = f1_score(y_true, y_pred, label)
        name = "spam  " if label == 1 else "normal"
        print(f"{name}  precision={p:.2f}  recall={r:.2f}  f1={f:.2f}")
