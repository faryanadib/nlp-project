# Phase 2 — Step 1: Dataset (Rotten Tomatoes)

**Date:** 2026-06-12
**Dataset:** `cornell-movie-review-data/rotten_tomatoes` (HuggingFace)
**Saved to:** `phase2_new_dataset/data/{train,validation,test}.csv`

## Splits

| Split | Rows | negative | positive | avg length |
|-------|------|----------|----------|------------|
| train | 8530 | 4265 (50%) | 4265 (50%) | 21.0 words / 114 chars |
| validation | 1066 | 533 (50%) | 533 (50%) | 21.0 words / 114 chars |
| test | 1066 | 533 (50%) | 533 (50%) | 21.2 words / 116 chars |
| **total** | **10662** | | | |

Perfectly class-balanced, short critic snippets (~21 words). Columns: `text,label`
(0 = negative, 1 = positive).

## Sample rows

**Positive (label=1):**
- "the rock is destined to be the 21st century's new conan ..."
- "the gorgeously elaborate continuation of the lord of the rings trilogy ..."
- "effective but too-tepid biopic"

**Negative (label=0):**
- "simplistic , silly and tedious ."
- "it's so laddish and juvenile , only teenage boys could possibly find it funny ."
- "exploitative and largely devoid of the depth or sophistication ..."

## Note — code change required

`huggingface_hub 1.19` no longer accepts the bare dataset id `rotten_tomatoes`.
Changed `DATASET_NAME` in `download_and_prepare.py` to the canonical namespaced id
`cornell-movie-review-data/rotten_tomatoes` (same dataset, no logic change).
Approved by user.
