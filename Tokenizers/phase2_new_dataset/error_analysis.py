"""
Phase 2 -- Error analysis of the (non-fine-tuned) IMDB model on the Rotten
Tomatoes test set.

Builds on evaluate.py's headline numbers (Acc 0.8096 / F1 0.8260) by digging
into *which* reviews are misclassified and *why*:
  * false positives vs. false negatives, with confidence (softmax prob)
  * does the model's confidence differ on errors vs. correct predictions?
    (ties back to the Phase 1 calibration FAIL -- the model is overconfident)
  * does review length correlate with errors?
  * lexical cues in errors: negation, contrast conjunctions ("but", "however")
    -- proxies for the subtly-negative / sarcastic reviews Phase 3 was built
    to catch
  * the most confidently-wrong examples of each type

Output: a single PDF report (charts + example tables) plus a raw CSV of every
misclassified row, both saved under results/phase2/.

Usage:
    python error_analysis.py
    python error_analysis.py --model fabriceyhc/bert-base-uncased-imdb
    python error_analysis.py --top 15      # more examples in the PDF
"""

import argparse
import os
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# -- Configuration ------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = "fabriceyhc/bert-base-uncased-imdb"  # same model evaluate.py scored
TEST_CSV = os.path.join(HERE, "data", "test.csv")
OUT_DIR = os.path.join(os.path.dirname(HERE), "results", "phase2")
ERRORS_CSV = os.path.join(OUT_DIR, "error_analysis_errors.csv")
REPORT_PDF = os.path.join(OUT_DIR, "ERROR_ANALYSIS.pdf")
MAX_LENGTH = 256
BATCH_SIZE = 32
TOP_N = 10

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")

NEGATION_RE = re.compile(r"\b(not|no|never|n't|none|nobody|nothing|neither|nor)\b", re.I)
CONTRAST_RE = re.compile(r"\b(but|however|yet|although|though|despite|while)\b", re.I)


def predict_all(texts, tokenizer, model):
    """Batched inference; returns (pred_labels, prob_of_predicted_class, prob_pos)."""
    preds, conf, prob_pos = [], [], []
    model.eval()
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        enc = tokenizer(batch, truncation=True, padding=True,
                        max_length=MAX_LENGTH, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1)
        labels = probs.argmax(-1)
        preds.extend(labels.cpu().tolist())
        conf.extend(probs.gather(1, labels.unsqueeze(1)).squeeze(1).cpu().tolist())
        prob_pos.extend(probs[:, 1].cpu().tolist())
    return np.array(preds), np.array(conf), np.array(prob_pos)


def lexical_rate(texts, pattern):
    return np.mean([bool(pattern.search(t)) for t in texts]) if len(texts) else 0.0


# -- PDF page builders ---------------------------------------------------------

def page_title(pdf, model_name, df_full, fp, fn, errors, correct):
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle("Phase 2 -- Error Analysis\nRotten Tomatoes test set", fontsize=16, y=0.97)
    acc = df_full["correct"].mean()
    lines = [
        f"Model: {model_name}",
        "(no fine-tuning -- see EVALUATION_SUMMARY.md)",
        "",
        f"Test set size: {len(df_full):,} reviews",
        f"Accuracy: {acc:.4f}",
        "",
        "Error breakdown",
        f"  Total errors: {len(errors)} ({len(errors)/len(df_full):.1%})",
        f"  False positives (true negative -> predicted positive): {len(fp)}",
        f"  False negatives (true positive -> predicted negative): {len(fn)}",
        "",
        "Confidence (overconfidence check)",
        f"  Correct predictions  -- mean confidence {correct['confidence'].mean():.3f}",
        f"  Errors (all)          -- mean confidence {errors['confidence'].mean():.3f}",
        f"    false positives      -- mean confidence {fp['confidence'].mean():.3f}",
        f"    false negatives      -- mean confidence {fn['confidence'].mean():.3f}",
        f"  Errors made with confidence >= 0.90: "
        f"{(errors['confidence'] >= 0.9).sum()}/{len(errors)} "
        f"({(errors['confidence'] >= 0.9).mean():.1%})",
        "",
        "The model is almost as confident when wrong as when right -- this",
        "matches the Phase 1 calibration FAIL (mean conf 0.993, near-certain",
        "on everything). Confidence is not a useful signal for filtering bad",
        "predictions here.",
        "",
        "Review length (words)",
        f"  Correct -- mean {correct['n_words'].mean():.1f}  median {correct['n_words'].median():.0f}",
        f"  Errors  -- mean {errors['n_words'].mean():.1f}  median {errors['n_words'].median():.0f}",
        "",
        "Lexical cues (share of reviews containing the cue)",
        f"  Negation   -- correct {lexical_rate(correct['text'], NEGATION_RE):.1%}"
        f" | errors {lexical_rate(errors['text'], NEGATION_RE):.1%}"
        f" | false positives {lexical_rate(fp['text'], NEGATION_RE):.1%}",
        f"  Contrast   -- correct {lexical_rate(correct['text'], CONTRAST_RE):.1%}"
        f" | errors {lexical_rate(errors['text'], CONTRAST_RE):.1%}"
        f" | false positives {lexical_rate(fp['text'], CONTRAST_RE):.1%}",
        "",
        "Both cues are over-represented among false positives: reviews that",
        "mix a negative clause with a softer/backhanded second clause",
        '("X, but Y") are where the model most often reads the wrong polarity',
        "-- the same subtly-negative / sarcastic pattern that motivated",
        "Phase 3.",
    ]
    fig.text(0.08, 0.90, "\n".join(lines), fontsize=9.5, family="monospace",
              va="top")
    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def page_charts(pdf, df_full, fp, fn, correct, errors):
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))

    # Confusion matrix heatmap
    cm = np.array([
        [(df_full[(df_full.true == 0)].pred == 0).sum(), len(fp)],
        [len(fn), (df_full[(df_full.true == 1)].pred == 1).sum()],
    ])
    ax = axes[0, 0]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred 0", "pred 1"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true 0", "true 1"])
    ax.set_title("Confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12)

    # Mean confidence: correct vs FP vs FN
    ax = axes[0, 1]
    cats = ["Correct", "False pos.", "False neg."]
    vals = [correct["confidence"].mean(), fp["confidence"].mean(), fn["confidence"].mean()]
    ax.bar(cats, vals, color=["#4caf50", "#e57373", "#ff9800"])
    ax.set_ylim(0, 1)
    ax.set_title("Mean confidence")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")

    # Confidence histogram: correct vs errors
    ax = axes[1, 0]
    ax.hist(correct["confidence"], bins=20, alpha=0.6, label="correct", color="#4caf50")
    ax.hist(errors["confidence"], bins=20, alpha=0.6, label="errors", color="#e57373")
    ax.set_title("Confidence distribution")
    ax.set_xlabel("confidence (prob of predicted class)")
    ax.legend(fontsize=8)

    # Lexical cue rates
    ax = axes[1, 1]
    cue_cats = ["Negation", "Contrast"]
    correct_rates = [lexical_rate(correct["text"], NEGATION_RE),
                     lexical_rate(correct["text"], CONTRAST_RE)]
    fp_rates = [lexical_rate(fp["text"], NEGATION_RE), lexical_rate(fp["text"], CONTRAST_RE)]
    x = np.arange(len(cue_cats))
    width = 0.35
    ax.bar(x - width/2, correct_rates, width, label="correct", color="#4caf50")
    ax.bar(x + width/2, fp_rates, width, label="false positives", color="#e57373")
    ax.set_xticks(x); ax.set_xticklabels(cue_cats)
    ax.set_title("Lexical cue rate")
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle("Error analysis -- charts", fontsize=13)
    pdf.savefig(fig)
    plt.close(fig)


def page_examples(pdf, sub, title, top_n):
    """One or more pages listing the top-N most confident examples in `sub`."""
    sub = sub.sort_values("confidence", ascending=False).head(top_n)
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle(title, fontsize=13, y=0.97)
    lines = []
    for _, row in sub.iterrows():
        wrapped = textwrap.fill(row["text"], width=95)
        lines.append(f"[conf {row['confidence']:.3f}]")
        lines.append(wrapped)
        lines.append("")
    fig.text(0.06, 0.92, "\n".join(lines), fontsize=8.5, family="monospace", va="top")
    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top", type=int, default=TOP_N,
                        help="how many top examples to include per error type")
    args = parser.parse_args()

    print(f"Device: {DEVICE}\nModel:  {args.model}\n")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(DEVICE)

    df = pd.read_csv(TEST_CSV)
    texts, y_true = df["text"].astype(str).tolist(), df["label"].to_numpy()
    print(f"Test set: {len(texts):,} reviews")

    y_pred, conf, prob_pos = predict_all(texts, tokenizer, model)

    df_full = pd.DataFrame({
        "text": texts, "true": y_true, "pred": y_pred,
        "confidence": conf, "prob_pos": prob_pos,
        "n_words": [len(t.split()) for t in texts],
        "has_negation": [bool(NEGATION_RE.search(t)) for t in texts],
        "has_contrast": [bool(CONTRAST_RE.search(t)) for t in texts],
    })
    df_full["correct"] = df_full["true"] == df_full["pred"]

    fp = df_full[(df_full["true"] == 0) & (df_full["pred"] == 1)]   # true neg -> predicted pos
    fn = df_full[(df_full["true"] == 1) & (df_full["pred"] == 0)]   # true pos -> predicted neg
    errors = pd.concat([fp, fn])
    correct = df_full[df_full["correct"]]

    acc = df_full["correct"].mean()
    print(f"\nAccuracy: {acc:.4f}  |  Errors: {len(errors)}/{len(df_full)} "
          f"({len(errors)/len(df_full):.1%})")
    print(f"  False positives (neg->pos): {len(fp)}")
    print(f"  False negatives (pos->neg): {len(fn)}")
    print(f"  Mean confidence -- correct {correct['confidence'].mean():.3f} | "
          f"errors {errors['confidence'].mean():.3f}")

    # -- Save artifacts -------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    errors.sort_values("confidence", ascending=False).to_csv(ERRORS_CSV, index=False)
    print(f"\nSaved {len(errors)} misclassified rows -> {ERRORS_CSV}")

    with PdfPages(REPORT_PDF) as pdf:
        page_title(pdf, args.model, df_full, fp, fn, errors, correct)
        page_charts(pdf, df_full, fp, fn, correct, errors)
        page_examples(pdf, fp, "Most confident FALSE POSITIVES\n"
                              "(actually negative, predicted positive)", args.top)
        page_examples(pdf, fn, "Most confident FALSE NEGATIVES\n"
                              "(actually positive, predicted negative)", args.top)
    print(f"Saved PDF report -> {REPORT_PDF}")


if __name__ == "__main__":
    main()
