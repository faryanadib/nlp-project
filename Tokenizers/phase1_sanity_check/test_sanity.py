"""
Pytest wrapper around the Phase 1 sanity checks.

Run with:
    pytest phase1_sanity_check/test_sanity.py -v

Override the model under test:
    pytest phase1_sanity_check/test_sanity.py --model <path-or-checkpoint>
"""

import pytest

import sanity_check as sc


@pytest.fixture(scope="module")
def model_and_tokenizer(request):
    """Load the model once for the whole module (loading is slow)."""
    model_path = request.config.getoption("--model", default=sc.MODEL_PATH)
    tokenizer, model = sc.load_model(model_path)
    return tokenizer, model


# Parametrize over the same registry used by sanity_check.py so the two
# entry points can never drift apart.
@pytest.mark.parametrize("name,check_fn", sc.ALL_CHECKS, ids=[n for n, _ in sc.ALL_CHECKS])
def test_sanity(name, check_fn, model_and_tokenizer):
    tokenizer, model = model_and_tokenizer
    passed, detail = check_fn(tokenizer, model)
    assert passed, f"{name}: {detail}"
