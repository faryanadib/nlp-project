# Phase 3 — Combined sentiment + sarcasm inference (with ready models)

**Date:** 2026-06-12
**How:** Imported the project's `CombinedAnalyzer` unchanged and passed ready models:
- Sentiment: `fabriceyhc/bert-base-uncased-imdb` (Phase 2 fallback, no fine-tuning)
- Sarcasm: `helinivan/english-sarcasm-detector` (pre-trained, no training from scratch)

No project code was edited — `CombinedAnalyzer` already accepts a `sarcasm_path` argument.

## Built-in sanity test: 4 / 6 passed

| Example | Got | Expected | OK |
|---------|-----|----------|----|
| "This film was a beautiful, moving masterpiece…" | Positive / Not Sarcastic | Positive / Not Sarc | ✅ |
| "Awful. Two hours of my life I will never get back." | Negative / Not Sarcastic | Negative / Not Sarc | ✅ |
| "Oh sure, this was the BEST movie ever made 🙄" | Positive / **Not Sarcastic** | Positive / Sarcastic | ❌ |
| "Yeah, terrible movie… said no one who enjoys fun." | Negative / Sarcastic | Negative / Sarcastic | ✅ |
| "Wow, what a thrill, watching paint dry…" | Positive / **Not Sarcastic** | any / Sarcastic | ❌ |
| "Absolutely loved the soundtrack and acting…" | Positive / Not Sarcastic | Positive / Not Sarc | ✅ |

Extra movie-review examples (all sarcasm missed):
- "Oh sure, BEST movie ever 🙄" → Sentiment Positive(100%), Sarcasm **Not Sarcastic**(69%) → verdict *Genuinely Positive* ❌
- "Wow, two hours I'll never get back. Just brilliant. /s" → Positive(100%), **Not Sarcastic**(97%) → *Genuinely Positive* ❌

## Key finding — domain gap (this is the important result)

The 2–3 failures are **all conversational, review-style sarcasm** (emojis 🙄, `/s`,
hyperbole). The sarcasm model scores **~94% on news headlines (in-domain)** but
**fails on movie-review sarcasm**, because it was trained on Onion/HuffPost
*headlines*, a different style from spoken-review sarcasm.

This compounds the Phase 2 weakness: the sentiment model calls sarcastic-negative
reviews "Positive", and the sarcasm model can't catch them either → the combined
verdict is wrong exactly on the hard, important cases.

**Honest conclusion:** a headline-trained sarcasm model (whether the ready
`helinivan` one or a from-scratch `train_sarcasm.py` model on the same data) does
**not** reliably transfer to movie-review sarcasm. Closing this gap would need a
review-domain sarcasm corpus, which (per DATASET_INSTRUCTIONS.md) does not exist
at comparable quality.

## Files
- `combined_inference_output.log` — full terminal output
- `PRETRAINED_MODEL_SUMMARY.md` — in-domain evaluation of the sarcasm model
