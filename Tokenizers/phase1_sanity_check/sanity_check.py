"""
Phase 1 — Sanity Check for the fine-tuned BERT sentiment model.

Runs 8 sanity-check categories against the model and prints a PASS/FAIL
report for each, plus an overall summary.

Usage:
    python sanity_check.py                     # uses MODEL_PATH below
    python sanity_check.py --model <path>      # any local dir or HF checkpoint

The same check functions are reused by test_sanity.py (pytest) and by
Phase 2 to re-validate the newly fine-tuned model.
"""

import argparse
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Configuration (edit here, no magic numbers below) ─────────────────────────
MODEL_PATH = "fabriceyhc/bert-base-uncased-imdb"  # ← swap with local path if needed
MAX_LENGTH = 512          # tokenizer truncation length
NUM_LABELS = 2            # binary sentiment: 0 = negative, 1 = positive
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0
PROB_SUM_TOL = 1e-4       # tolerance for softmax-sums-to-1 check
DETERMINISM_RUNS = 3      # how many repeated runs for reproducibility check
BATCH_TOL = 1e-4          # max allowed |single - batched| logit difference
CALIBRATION_MIN_SPREAD = 0.15  # min std-dev of confidences across mixed inputs
SEED = 42

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")

# Test sentences --------------------------------------------------------------
POSITIVE_SENTENCES = [
    "This movie was absolutely fantastic!",
    "A masterpiece — brilliant acting and a beautiful story.",
    "I loved every single minute of this wonderful film.",
]
NEGATIVE_SENTENCES = [
    "This was the worst film I've ever seen.",
    "Terrible. Boring plot, awful acting, a complete waste of time.",
    "I hated this movie, it was painfully bad.",
]
NEUTRAL_SENTENCES = [
    "The movie was released in cinemas last Friday.",
    "It had some good parts and some bad parts.",
    "The film is about two hours long.",
]
EDGE_CASES = {
    "empty string": "",
    "single word": "good",
    "all punctuation": "!?!...,;:-()!!!",
    "very long text (512+ tokens)": "this movie was great and I enjoyed it a lot " * 100,
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_model(model_path=MODEL_PATH):
    """Load tokenizer + classification model and put model in eval mode."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(DEVICE).eval()
    return tokenizer, model


def predict(texts, tokenizer, model):
    """Return (logits, probs, predicted_labels) for a list of texts."""
    if isinstance(texts, str):
        texts = [texts]
    inputs = tokenizer(texts, truncation=True, padding=True,
                       max_length=MAX_LENGTH, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)
    preds = probs.argmax(dim=-1)
    return logits.cpu(), probs.cpu(), preds.cpu()


# ── Checks (each returns (passed: bool, detail: str)) ─────────────────────────
def check_output_shape(tokenizer, model):
    """1. Logits shape is (batch, NUM_LABELS); softmax rows sum to 1."""
    texts = POSITIVE_SENTENCES + NEGATIVE_SENTENCES
    logits, probs, _ = predict(texts, tokenizer, model)
    if logits.shape != (len(texts), NUM_LABELS):
        return False, f"expected shape {(len(texts), NUM_LABELS)}, got {tuple(logits.shape)}"
    if logits.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False, f"unexpected logits dtype {logits.dtype}"
    sums = probs.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=PROB_SUM_TOL):
        return False, f"softmax rows do not sum to 1 (max dev {(sums - 1).abs().max():.2e})"
    return True, f"shape {tuple(logits.shape)} OK, prob sums within {PROB_SUM_TOL}"


def check_prediction_direction(tokenizer, model):
    """2. Obviously positive → POSITIVE_LABEL, obviously negative → NEGATIVE_LABEL."""
    _, _, pos_preds = predict(POSITIVE_SENTENCES, tokenizer, model)
    _, _, neg_preds = predict(NEGATIVE_SENTENCES, tokenizer, model)
    wrong = []
    for s, p in zip(POSITIVE_SENTENCES, pos_preds.tolist()):
        if p != POSITIVE_LABEL:
            wrong.append(f"'{s[:40]}…' → {p}")
    for s, p in zip(NEGATIVE_SENTENCES, neg_preds.tolist()):
        if p != NEGATIVE_LABEL:
            wrong.append(f"'{s[:40]}…' → {p}")
    if wrong:
        return False, "misclassified: " + "; ".join(wrong)
    return True, f"all {len(POSITIVE_SENTENCES) + len(NEGATIVE_SENTENCES)} clear-cut sentences correct"


def check_edge_cases(tokenizer, model):
    """3. Model must not crash / produce NaN on degenerate inputs."""
    failures = []
    for name, text in EDGE_CASES.items():
        try:
            logits, probs, _ = predict(text, tokenizer, model)
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                failures.append(f"{name}: NaN/Inf in logits")
            elif logits.shape != (1, NUM_LABELS):
                failures.append(f"{name}: bad shape {tuple(logits.shape)}")
        except Exception as e:  # noqa: BLE001 — any crash is a failure here
            failures.append(f"{name}: raised {type(e).__name__}: {e}")
    if failures:
        return False, "; ".join(failures)
    return True, f"all {len(EDGE_CASES)} edge cases handled without crash/NaN"


def check_confidence(tokenizer, model):
    """4. Mean confidence on neutral sentences < mean confidence on clear ones."""
    _, clear_probs, _ = predict(POSITIVE_SENTENCES + NEGATIVE_SENTENCES, tokenizer, model)
    _, neutral_probs, _ = predict(NEUTRAL_SENTENCES, tokenizer, model)
    clear_conf = clear_probs.max(dim=-1).values.mean().item()
    neutral_conf = neutral_probs.max(dim=-1).values.mean().item()
    detail = f"clear conf {clear_conf:.3f} vs neutral conf {neutral_conf:.3f}"
    return neutral_conf < clear_conf, detail


def check_reproducibility(tokenizer, model):
    """5. Identical input → identical logits across repeated runs."""
    text = POSITIVE_SENTENCES[0]
    runs = []
    for _ in range(DETERMINISM_RUNS):
        logits, _, _ = predict(text, tokenizer, model)
        runs.append(logits)
    for i in range(1, len(runs)):
        if not torch.equal(runs[0], runs[i]):
            dev = (runs[0] - runs[i]).abs().max().item()
            # allow tiny non-determinism on some GPU backends
            if dev > 1e-5:
                return False, f"run 0 vs run {i} differ by {dev:.2e}"
    return True, f"identical logits across {DETERMINISM_RUNS} runs"


def check_calibration(tokenizer, model):
    """6. Confidences over a mixed set are meaningfully spread (not all ~0.5 or ~1.0)."""
    texts = POSITIVE_SENTENCES + NEGATIVE_SENTENCES + NEUTRAL_SENTENCES
    _, probs, _ = predict(texts, tokenizer, model)
    conf = probs.max(dim=-1).values
    spread = conf.max().item() - conf.min().item()
    mean = conf.mean().item()
    detail = f"conf mean {mean:.3f}, range {spread:.3f}"
    if spread < CALIBRATION_MIN_SPREAD:
        # all confidences nearly identical — degenerate behaviour
        if mean > 0.95:
            return False, detail + " — all predictions near-certain (overconfident)"
        if abs(mean - 1.0 / NUM_LABELS) < 0.05:
            return False, detail + " — all predictions near-uniform (model not discriminating)"
        return False, detail + f" — spread below {CALIBRATION_MIN_SPREAD}"
    return True, detail


def check_batch_consistency(tokenizer, model):
    """7. Single-sample predictions == batched predictions (within tolerance)."""
    texts = POSITIVE_SENTENCES + NEGATIVE_SENTENCES
    batch_logits, _, batch_preds = predict(texts, tokenizer, model)
    max_dev = 0.0
    for i, t in enumerate(texts):
        single_logits, _, single_preds = predict(t, tokenizer, model)
        max_dev = max(max_dev, (single_logits[0] - batch_logits[i]).abs().max().item())
        if single_preds[0] != batch_preds[i]:
            return False, f"label flip for sample {i} (single {single_preds[0]} vs batch {batch_preds[i]})"
    if max_dev > BATCH_TOL:
        return False, f"max logit deviation {max_dev:.2e} > {BATCH_TOL}"
    return True, f"max logit deviation {max_dev:.2e} (≤ {BATCH_TOL}), labels identical"


def check_tokenization(tokenizer, model):
    """8. Tokenizer truncates long input to MAX_LENGTH and pads batches correctly."""
    long_text = "word " * (MAX_LENGTH * 3)
    enc = tokenizer(long_text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    if enc["input_ids"].shape[1] > MAX_LENGTH:
        return False, f"truncation failed: {enc['input_ids'].shape[1]} > {MAX_LENGTH}"
    # padding: short + long together must yield equal-length rows with a mask
    batch = tokenizer(["hi", long_text], truncation=True, padding=True,
                      max_length=MAX_LENGTH, return_tensors="pt")
    ids, mask = batch["input_ids"], batch["attention_mask"]
    if ids.shape[0] != 2 or ids.shape[1] > MAX_LENGTH:
        return False, f"bad padded batch shape {tuple(ids.shape)}"
    short_real = int(mask[0].sum())   # non-pad tokens in the short row
    if short_real >= ids.shape[1]:
        return False, "short sequence was not padded"
    pad_id = tokenizer.pad_token_id
    if not (ids[0][short_real:] == pad_id).all():
        return False, "padding region does not contain pad_token_id"
    return True, (f"truncated to {enc['input_ids'].shape[1]} tokens, "
                  f"padded batch {tuple(ids.shape)} with correct mask")


ALL_CHECKS = [
    ("Output shape & type",        check_output_shape),
    ("Prediction direction",       check_prediction_direction),
    ("Edge cases",                 check_edge_cases),
    ("Confidence (neutral<clear)", check_confidence),
    ("Reproducibility",            check_reproducibility),
    ("Probability calibration",    check_calibration),
    ("Batch consistency",          check_batch_consistency),
    ("Tokenization (trunc/pad)",   check_tokenization),
]


def run_all_checks(model_path=MODEL_PATH, verbose=True):
    """Run every check; return list of (name, passed, detail)."""
    torch.manual_seed(SEED)
    if verbose:
        print(f"Device : {DEVICE}\nModel  : {model_path}\nLoading model…\n")
    tokenizer, model = load_model(model_path)
    results = []
    for name, fn in ALL_CHECKS:
        try:
            passed, detail = fn(tokenizer, model)
        except Exception as e:  # a crashing check is a failing check
            passed, detail = False, f"check raised {type(e).__name__}: {e}"
        results.append((name, passed, detail))
        if verbose:
            print(f"[{'PASS' if passed else 'FAIL'}] {name:<28} {detail}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_PATH,
                        help="local path or HuggingFace checkpoint name")
    args = parser.parse_args()

    results = run_all_checks(args.model)
    n_pass = sum(p for _, p, _ in results)
    n_fail = len(results) - n_pass

    print("\n" + "─" * 60)
    print(f"SUMMARY: {n_pass}/{len(results)} checks passed, {n_fail} failed")
    if n_fail == 0:
        print("✅ PASSED — model behaves sanely on all checks")
    else:
        failed = [n for n, p, _ in results if not p]
        print(f"❌ FAILED — failing checks: {', '.join(failed)}")
    print("─" * 60)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
