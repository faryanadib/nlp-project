import sys
import os

# ── Imports from previous weeks ───────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), "../Week03"))
from calssifier import tokenize, load_imdb, precision, recall, f1_score


# ─────────────────────────────────────────────────────────────────
# 1. Load Data  (limit keeps pure-Python runtime reasonable)
# ─────────────────────────────────────────────────────────────────

base      = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base, "../Data/IMDB Dataset.csv")

raw   = load_imdb(data_path, limit=2000)          # returns (text, "positive"/"negative")
data  = [(text, 1 if label == "positive" else -1) # map to +1 / -1 for perceptron
         for text, label in raw]

split      = int(len(data) * 0.8)
train_data = data[:split]
test_data  = data[split:]

print(f"Train: {len(train_data)}  |  Test: {len(test_data)}")


# ─────────────────────────────────────────────────────────────────
# 2. Build Vocabulary  (from training set only)
# ─────────────────────────────────────────────────────────────────

def build_vocab(data):
    vocab = {}
    for text, _ in data:
        for word in tokenize(text):
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


# ─────────────────────────────────────────────────────────────────
# 3. Vectorize  (sparse: dict of {word_index: 1} — skips zeros)
# ─────────────────────────────────────────────────────────────────

def vectorize(text, vocab):
    vec = {}
    for word in tokenize(text):
        if word in vocab:
            vec[vocab[word]] = 1
    return vec


# ─────────────────────────────────────────────────────────────────
# 4. Initialize Model
# ─────────────────────────────────────────────────────────────────

vocab = build_vocab(train_data)
print(f"Vocabulary size: {len(vocab)}")

w = {}   # sparse weights: only store non-zero
b = 0


# ─────────────────────────────────────────────────────────────────
# 5. Predict
# ─────────────────────────────────────────────────────────────────

def predict(x):
    score = sum(w.get(i, 0) * x[i] for i in x) + b
    return 1 if score > 0 else -1


# ─────────────────────────────────────────────────────────────────
# 6. Update
# ─────────────────────────────────────────────────────────────────

def update(x, y):
    global w, b
    if predict(x) != y:
        for i in x:
            w[i] = w.get(i, 0) + y
        b = b + y


# ─────────────────────────────────────────────────────────────────
# 7. Training Loop
# ─────────────────────────────────────────────────────────────────

epochs = 10
for epoch in range(epochs):
    errors = 0
    for text, label in train_data:
        x = vectorize(text, vocab)
        update(x, label)
        if predict(x) != label:
            errors += 1
    print(f"Epoch {epoch + 1:2d}  |  errors={errors}")


# ─────────────────────────────────────────────────────────────────
# 8. Test the Model
# ─────────────────────────────────────────────────────────────────

y_true = [label for _, label in test_data]
y_pred = [predict(vectorize(text, vocab)) for text, _ in test_data]


# ─────────────────────────────────────────────────────────────────
# 9. Inspect Weights  (top positive / negative)
# ─────────────────────────────────────────────────────────────────

idx_to_word = {idx: word for word, idx in vocab.items()}

print("\n--- Top positive words ---")
for idx in sorted(w, key=lambda i: w[i], reverse=True)[:10]:
    print(f"  {idx_to_word[idx]:15s}  w={w[idx]}")

print("\n--- Top negative words ---")
for idx in sorted(w, key=lambda i: w[i])[:10]:
    print(f"  {idx_to_word[idx]:15s}  w={w[idx]}")


# ─────────────────────────────────────────────────────────────────
# 10. Evaluation  (using eval functions from Week03)
# ─────────────────────────────────────────────────────────────────

print("\n--- Evaluation ---")
for label in [1, -1]:
    p = precision(y_true, y_pred, label)
    r = recall(y_true, y_pred, label)
    f = f1_score(y_true, y_pred, label)
    name = "positive" if label == 1 else "negative"
    print(f"{name}  precision={p:.2f}  recall={r:.2f}  f1={f:.2f}")
