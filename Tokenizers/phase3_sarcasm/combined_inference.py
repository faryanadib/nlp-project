"""
Phase 3 — Step 2c: Combined sentiment + sarcasm inference.

Pipeline per review:
  1. Sentiment model (Phase 2 fine-tuned checkpoint) → label + confidence
  2. Sarcasm model (Phase 3 checkpoint)              → label + confidence
  3. Verdict logic:
       Positive + Not Sarcastic → Genuinely Positive
       Negative + Not Sarcastic → Genuinely Negative
       Positive + Sarcastic     → Likely Negative (sarcastic positive)
       Negative + Sarcastic     → Likely Negative (sarcastic negative)

Usage:
    python combined_inference.py "Oh sure, this was the BEST movie ever 🙄"
    python combined_inference.py --test     # run built-in sanity checks

The CombinedAnalyzer class is also imported by the Phase 5 web app.
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Configuration ──────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Phase 2 fine-tuned sentiment model; falls back to the original checkpoint
# if Phase 2 has not been run yet.
SENTIMENT_MODEL = os.path.join(ROOT, "phase2_new_dataset", "checkpoints", "best_model")
SENTIMENT_FALLBACK = "fabriceyhc/bert-base-uncased-imdb"
SARCASM_MODEL = os.path.join(HERE, "checkpoints", "movie_reddit_model")

MAX_LENGTH = 256
SENTIMENT_LABELS = {0: "Negative", 1: "Positive"}
SARCASM_LABELS = {0: "Not Sarcastic", 1: "Sarcastic"}

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")

# Handcrafted test examples: (text, expected_sentiment, expected_sarcasm)
# expected_sarcasm None = don't assert (borderline). Covers all 4 verdict cases.
TEST_EXAMPLES = [
    ("This film was a beautiful, moving masterpiece. I cried at the end.",
     "Positive", "Not Sarcastic"),                       # case 1: genuinely positive
    ("Awful. Two hours of my life I will never get back.",
     "Negative", "Not Sarcastic"),                       # case 2: genuinely negative
    ("Oh sure, this was the BEST movie ever made 🙄",
     "Positive", "Sarcastic"),                           # case 3: sarcastic positive
    ("Yeah, terrible movie, just terrible — said no one who enjoys fun.",
     "Negative", "Sarcastic"),                           # case 4: sarcastic negative
    ("Wow, what a thrill, watching paint dry would have been more exciting.",
     None, "Sarcastic"),                                 # extra sarcastic example
    ("Absolutely loved the soundtrack and the acting was superb!",
     "Positive", "Not Sarcastic"),                       # extra genuine positive
]


class CombinedAnalyzer:
    """Loads both models once; analyzes texts with sentiment+sarcasm logic."""

    def __init__(self, sentiment_path=None, sarcasm_path=SARCASM_MODEL):
        if sentiment_path is None:
            # prefer the Phase 2 fine-tuned checkpoint when available
            sentiment_path = (SENTIMENT_MODEL
                              if os.path.isfile(os.path.join(SENTIMENT_MODEL, "config.json"))
                              else SENTIMENT_FALLBACK)
        print(f"Sentiment model: {sentiment_path}")
        print(f"Sarcasm model:   {sarcasm_path}")
        self.sent_tok = AutoTokenizer.from_pretrained(sentiment_path)
        self.sent_model = AutoModelForSequenceClassification.from_pretrained(
            sentiment_path).to(DEVICE).eval()
        self.sarc_tok = AutoTokenizer.from_pretrained(sarcasm_path)
        self.sarc_model = AutoModelForSequenceClassification.from_pretrained(
            sarcasm_path).to(DEVICE).eval()

    def _predict(self, text, tokenizer, model):
        """Return (label_id, confidence) for one text with one model."""
        enc = tokenizer(text, truncation=True, max_length=MAX_LENGTH,
                        return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            probs = F.softmax(model(**enc).logits, dim=-1)[0]
        label = int(probs.argmax())
        return label, float(probs[label])

    @staticmethod
    def verdict(sentiment, sarcasm):
        """Apply the 4-case combination logic."""
        if sentiment == "Positive" and sarcasm == "Not Sarcastic":
            return "Genuinely Positive"
        if sentiment == "Negative" and sarcasm == "Not Sarcastic":
            return "Genuinely Negative"
        if sentiment == "Positive" and sarcasm == "Sarcastic":
            return "Likely Negative (sarcastic positive)"
        return "Likely Negative (sarcastic negative)"

    def analyze(self, text):
        """Full pipeline for one review; returns a result dict."""
        s_label, s_conf = self._predict(text, self.sent_tok, self.sent_model)
        c_label, c_conf = self._predict(text, self.sarc_tok, self.sarc_model)
        sentiment = SENTIMENT_LABELS[s_label]
        sarcasm = SARCASM_LABELS[c_label]
        return {
            "text": text,
            "sentiment": sentiment, "sentiment_confidence": s_conf,
            "sarcasm": sarcasm, "sarcasm_confidence": c_conf,
            "verdict": self.verdict(sentiment, sarcasm),
            # effective polarity after sarcasm correction (used by Phase 5)
            "effective_positive": sentiment == "Positive" and sarcasm == "Not Sarcastic",
        }


def print_result(r):
    print(f"\nReview   : {r['text']}")
    print(f"Sentiment: {r['sentiment']} ({r['sentiment_confidence']:.1%})")
    print(f"Sarcasm  : {r['sarcasm']} ({r['sarcasm_confidence']:.1%})")
    print(f"Verdict  : {r['verdict']}")


def run_tests(analyzer):
    """Phase 3 sanity checks: all 4 verdict cases + sarcasm direction asserts."""
    n_pass = 0
    for text, exp_sent, exp_sarc in TEST_EXAMPLES:
        r = analyzer.analyze(text)
        ok = ((exp_sent is None or r["sentiment"] == exp_sent) and
              (exp_sarc is None or r["sarcasm"] == exp_sarc))
        n_pass += ok
        print(f"[{'PASS' if ok else 'FAIL'}] '{text[:55]}…'" if len(text) > 55
              else f"[{'PASS' if ok else 'FAIL'}] '{text}'")
        print(f"       got {r['sentiment']}/{r['sarcasm']} "
              f"(expected {exp_sent or 'any'}/{exp_sarc or 'any'}) → {r['verdict']}")
    print("\n" + "─" * 60)
    print(f"SUMMARY: {n_pass}/{len(TEST_EXAMPLES)} examples passed")
    print("✅ PASSED" if n_pass == len(TEST_EXAMPLES) else "❌ FAILED")
    return n_pass == len(TEST_EXAMPLES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", help="review text to analyze")
    parser.add_argument("--test", action="store_true",
                        help="run built-in sanity-check examples")
    args = parser.parse_args()

    analyzer = CombinedAnalyzer()
    if args.test:
        sys.exit(0 if run_tests(analyzer) else 1)
    if not args.text:
        parser.error("provide a review text or use --test")
    print_result(analyzer.analyze(args.text))


if __name__ == "__main__":
    main()
