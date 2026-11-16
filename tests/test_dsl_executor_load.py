"""
LocalExecutor refuses to fall back to random weights when a named model fails to load.
"""

import pytest

from turing.dsl.executor import LocalExecutor


def test_unloadable_model_raises_instead_of_random_weights():
    with pytest.raises(RuntimeError, match="(?s)Could not load weights for '/nonexistent/turing-model'.*test-tiny"):
        LocalExecutor("/nonexistent/turing-model", device="cpu")


def test_test_tiny_still_gives_synthetic_weights():
    ex = LocalExecutor("test-tiny", device="cpu")
    assert ex.tokenizer is None and ex.config.name
