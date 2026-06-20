"""
Phase 5 — End-to-end pipeline sanity check (no user input required).

Checks:
  1. app.py imports without errors (i.e. the app can start)
  2. run_pipeline() works end-to-end for a hardcoded movie
  3. the verdict is one of the expected values

Usage:
    python test_pipeline_e2e.py
"""

import sys

# ── Configuration ──────────────────────────────────────────────────────────────
TEST_MOVIE = "Inception"            # hardcoded, well-known movie
EXPECTED_VERDICTS = {"GOOD", "BAD", "MIXED"}

checks = []  # (name, passed, detail)


def record(name, passed, detail=""):
    checks.append((name, passed, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def main():
    # 1. App imports/starts without errors -------------------------------------
    try:
        import app  # noqa: F401 — import is the test
        record("App imports without errors", True)
    except Exception as e:
        record("App imports without errors", False, f"{type(e).__name__}: {e}")
        finish()

    # 2. End-to-end pipeline ----------------------------------------------------
    try:
        res = app.run_pipeline(TEST_MOVIE)
        ok = res["n_reviews"] > 0
        record("Pipeline returns reviews", ok,
               f"{res['n_reviews']} reviews for '{TEST_MOVIE}'")
    except Exception as e:
        record("Pipeline returns reviews", False, f"{type(e).__name__}: {e}")
        finish()

    # 3. Verdict is one of the expected values ----------------------------------
    if res["n_reviews"] > 0:
        record("Verdict is GOOD/BAD/MIXED",
               res["verdict"] in EXPECTED_VERDICTS,
               f"verdict={res['verdict']}, confidence={res['confidence']}%")
        record("Sarcasm stats present",
               isinstance(res.get("n_sarcastic"), int),
               f"{res.get('n_sarcastic')} sarcastic reviews")
    else:
        record("Verdict is GOOD/BAD/MIXED", False,
               "no reviews collected — cannot compute a verdict")

    finish()


def finish():
    n_pass = sum(p for _, p, _ in checks)
    print("\n" + "─" * 60)
    print(f"SUMMARY: {n_pass}/{len(checks)} checks passed")
    ok = n_pass == len(checks)
    print("✅ PASSED — end-to-end pipeline works" if ok else "❌ FAILED — see above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
