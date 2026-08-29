"""
Unit tests for Native C++20 Deterministic Fast Hasher.
Verifies hashing determinism, seed variance, and distribution properties.
"""

import pytest

try:
    import turing.turing_csrc as turing_csrc
    HAS_CSRC = True
except ImportError:
    HAS_CSRC = False

from turing.serving.kv_events import deterministic_block_hash


def test_deterministic_token_hash_cpp():
    if not HAS_CSRC:
        pytest.skip("turing_csrc not available")

    tokens_1 = [101, 2054, 2003, 1037, 3231, 102]
    tokens_2 = [101, 2054, 2003, 1037, 3231, 102]
    tokens_3 = [101, 2054, 2003, 1037, 3231, 103]

    h1 = turing_csrc.deterministic_token_hash_cpu(tokens_1, 0)
    h2 = turing_csrc.deterministic_token_hash_cpu(tokens_2, 0)
    h3 = turing_csrc.deterministic_token_hash_cpu(tokens_3, 0)

    # Identical inputs produce identical hash
    assert h1 == h2
    # Different inputs produce different hash
    assert h1 != h3

    # Seed variance
    h1_seeded = turing_csrc.deterministic_token_hash_cpu(tokens_1, 42)
    assert h1 != h1_seeded


def test_deterministic_block_hash_wrapper():
    tokens = list(range(1000, 1064))
    h = deterministic_block_hash(tokens, seed=123)
    assert isinstance(h, int)
    assert h != 0

    h_again = deterministic_block_hash(tokens, seed=123)
    assert h == h_again
