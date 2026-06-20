"""Pytest configuration for the sanity-check tests (adds --model option)."""

import os
import sys

# Make `import sanity_check` work no matter where pytest is invoked from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sanity_check as sc  # noqa: E402


def pytest_addoption(parser):
    """Allow `pytest --model <path>` to test any checkpoint."""
    parser.addoption("--model", action="store", default=sc.MODEL_PATH,
                     help="local path or HuggingFace checkpoint name")
