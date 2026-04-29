import csv
import math

# 1. Load Data
def load_imdb(path, limit=2000):
    data = []
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            data.append((row['review'], row['sentiment']))
    return data


# 2. Tokenize (simple whitespace + lowercase + remove html tags)
def tokenize(text):
    text = text.lower()
    # remove html tags
    clean = ""
    inside_tag = False
    for ch in text:
        if ch == '<':
            inside_tag = True
        elif ch == '>':
            inside_tag = False
        elif not inside_tag:
            clean += ch
    return clean.split()


# 3. Train  (Naive Bayes with Laplace smoothing)

def train(data):
    class_counts = {}
    word_counts  = {}
    total_words  = {}
    vocab        = set()

    for text, label in data:
        class_counts[label] = class_counts.get(label, 0) + 1
        if label not in word_counts:
            word_counts[label] = {}
            total_words[label] = 0
        for word in tokenize(text):
            vocab.add(word)
            word_counts[label][word] = word_counts[label].get(word, 0) + 1
            total_words[label] += 1

    total_docs = len(data)
    priors     = {label: class_counts[label] / total_docs for label in class_counts}
    vocab_size = len(vocab)

    # P(word | class)  with Laplace smoothing  →  (count + 1) / (total + vocab_size)
    likelihoods = {}
    for label in word_counts:
        likelihoods[label] = {}
        for word in vocab:
            count = word_counts[label].get(word, 0)
            likelihoods[label][word] = (count + 1) / (total_words[label] + vocab_size)

    return priors, likelihoods, total_words, vocab_size


# ─────────────────────────────────────────────────────────────────
# 4. Predict
# ─────────────────────────────────────────────────────────────────
def predict(text, priors, likelihoods, total_words, vocab_size):
    best_label = None
    best_score = None

    for label in priors:
        score = math.log(priors[label])          # log P(class)
        for word in tokenize(text):
            if word in likelihoods[label]:
                score += math.log(likelihoods[label][word])
            else:
                # unseen word  →  1 / (total + vocab_size)
                score += math.log(1 / (total_words[label] + vocab_size))

        if best_score is None or score > best_score:
            best_score = score
            best_label = label

    return best_label


# 5. Evaluation  (precision / recall / F1 )

def precision(y_true, y_pred, pos_label):
    TP = sum(1 for t, p in zip(y_true, y_pred) if p == pos_label and t == pos_label)
    FP = sum(1 for t, p in zip(y_true, y_pred) if p == pos_label and t != pos_label)
    return TP / (TP + FP) if TP + FP > 0 else 0

def recall(y_true, y_pred, pos_label):
    TP = sum(1 for t, p in zip(y_true, y_pred) if t == pos_label and p == pos_label)
    FN = sum(1 for t, p in zip(y_true, y_pred) if t == pos_label and p != pos_label)
    return TP / (TP + FN) if TP + FN > 0 else 0

def f1_score(y_true, y_pred, pos_label):
    p = precision(y_true, y_pred, pos_label)
    r = recall(y_true, y_pred, pos_label)
    return 2 * p * r / (p + r) if p + r > 0 else 0


# 6. Main

if __name__ == "__main__":

    # Load  (2000 rows so it runs fast)
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    data = load_imdb(os.path.join(base, "../Data/IMDB Dataset.csv"), limit=2000)

    # 80 / 20  train / test split
    split      = int(len(data) * 0.8)
    train_data = data[:split]
    test_data  = data[split:]

    print(f"Train: {len(train_data)}  |  Test: {len(test_data)}")

    # Train
    priors, likelihoods, total_words, vocab_size = train(train_data)

    # Predict
    y_true = [label for _, label in test_data]
    y_pred = [predict(text, priors, likelihoods, total_words, vocab_size)
              for text, _ in test_data]

    # Evaluate
    print("\n--- Results ---")
    for label in ["positive", "negative"]:
        p = precision(y_true, y_pred, label)
        r = recall(y_true, y_pred, label)
        f = f1_score(y_true, y_pred, label)
        print(f"{label:10s}  precision={p:.2f}  recall={r:.2f}  f1={f:.2f}")