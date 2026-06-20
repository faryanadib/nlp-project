# Phase 3 — Sarcasm Dataset Instructions

## Which dataset to use

**News Headlines Dataset for Sarcasm Detection (v2)** by Rishabh Misra
(headlines from *The Onion* = sarcastic, *HuffPost* = not sarcastic).

### Why this one

- **High label quality.** Labels are derived from the publication source, not
  from noisy `/s` self-tags (the Reddit SARC corpus) or third-party annotation
  (iSarcasm, which is also tiny and Twitter-specific). Misclabeled examples
  are rare.
- **Clean, formal-ish English** with no @mentions/hashtags — transfers better
  to review text than Twitter data.
- **Good size & balance:** ~28,600 headlines, ≈ 47% sarcastic.
- **Caveat (be honest in your report):** it is news-headline domain, not
  reviews. No public *review-domain* sarcasm corpus of comparable quality
  exists; this is the standard compromise. The Reddit SARC corpus is the
  fallback if you want informal commentary text (much larger but far noisier).

## How to download

**Option A — Kaggle (recommended):**
1. Go to: https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection
2. Click **Download** (free Kaggle account required), or use the CLI:
   ```bash
   kaggle datasets download rmisra/news-headlines-dataset-for-sarcasm-detection
   unzip news-headlines-dataset-for-sarcasm-detection.zip
   ```

**Option B — author's page:** https://rishabhmisra.github.io/publications/
(look for "News Headlines Dataset For Sarcasm Detection").

## Where to put the file

Place this file in `phase3_sarcasm/data/`:

```
phase3_sarcasm/data/Sarcasm_Headlines_Dataset_v2.json
```

(v1, `Sarcasm_Headlines_Dataset.json`, also works — the training script
auto-detects any `*.json` file in `data/`.)

## Expected format

JSON Lines — one JSON object per line with these fields:

```json
{"is_sarcastic": 1, "headline": "thirtysomething scientists unveil doomsday clock of hair loss", "article_link": "https://..."}
```

The training script uses `headline` as the text and `is_sarcastic` (0/1) as
the label; `article_link` is ignored.

**Alternative:** if you prefer another dataset, save it as a CSV named
`phase3_sarcasm/data/sarcasm.csv` with columns `text,label` (label: 1 =
sarcastic, 0 = not). The training script supports both formats.
