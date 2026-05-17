# Week 05 — From-Scratch NLP Classifiers

This folder contains two scripts for **Sentiment Analysis on IMDB Movie Reviews**,
implemented from scratch using only standard Python (no ML libraries).

---

## Files

| File | Description |
|------|-------------|
| `logistic_regression.py` | Logistic Regression classifier — standalone |
| `compare_models.py` | Runs Naive Bayes, Perceptron, and Logistic Regression on the same data and compares results |

---

## Project Structure

Make sure your folder looks like this before running:

```
Project/
├── Data/
│   └── IMDB Dataset.csv        ← dataset goes here
├── Week02/
│   └── Eval.py
├── Week03/
│   └── calssifier.py
├── Week04/
│   └── homework04.py
└── Week05/
    ├── logistic_regression.py
    ├── compare_models.py
    └── README.md
```

---

## Dataset

Download the IMDB dataset from Kaggle:
**https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews**

Place the file at:
```
Project/Data/IMDB Dataset.csv
```

> The CSV must have two columns: `review` and `sentiment` (`positive` / `negative`).

---

## Requirements

No external libraries needed. Only the Python standard library is used:
- `math`
- `csv`
- `os`
- `random`
- `sys`

Python version: **3.8 or higher**

---

## How to Run

### Option 1 — Logistic Regression only

```bash
cd Project/Week05
python logistic_regression.py
```

Expected output:
```
Loading dataset (10000 samples)...
Splitting 80/20 (seed=42)...
Train: 8000  |  Test: 2000
Building vocabulary...
Vocabulary size: ~127000
Training Logistic Regression (10 epochs, lr=0.1)...
  Epoch  1 | loss = 0.7068
  ...
  Epoch 10 | loss = 0.0069
===== Results =====
  Accuracy  : 0.8440
  F1 Score  : 0.8591
```

---

### Option 2 — Compare all three models

```bash
cd Project/Week05
python compare_models.py
```

Expected output:
```
[1/3] Training Naive Bayes...
[2/3] Training Perceptron (10 epochs)...
[3/3] Training Logistic Regression (10 epochs, lr=0.1)...

════════════════════════════════════════════
   MODEL COMPARISON  (n=10000, 80/20)
════════════════════════════════════════════
  Naive Bayes        → Accuracy: 0.8250  F1: 0.8241
  Perceptron         → Accuracy: 0.8420  F1: 0.8580
  Logistic Regression→ Accuracy: 0.8440  F1: 0.8591
```

> Runtime: ~15–25 minutes on a standard laptop (pure Python, no numpy, full 50k dataset).

---

## Configuration

To change the number of samples, open either file and edit the constant at the top:

**`compare_models.py`** — line 17:
```python
MAX_SAMPLES = 50000   # full dataset — change to a smaller number for faster runs
```

**`logistic_regression.py`** — line 156:
```python
raw = load_imdb(dataset_file, limit=50000)   # change to smaller number for faster runs
```

To change the number of training epochs or learning rate:

**`logistic_regression.py`** — line 169:
```python
w, b = train(train_texts, train_labels, vocab, epochs=10, lr=0.1)
#                                                       ^       ^
#                                                    epochs    lr
```

**`compare_models.py`** — line 18:
```python
EPOCHS = 10   # change this to run more or fewer training passes
```

---

## If the Dataset Path Does Not Work

Both scripts build the path automatically relative to their own location.
If you get a `FileNotFoundError`, check that the CSV file is placed at exactly:

```
Project/Data/IMDB Dataset.csv
```

Note the **capital D** in `Data` and the **space** in `IMDB Dataset.csv`.
