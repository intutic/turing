"""
Unit tests for ZeroMQ KV block event publisher and deterministic hashing (llm-d integration).
"""

import pytest
import time
from turing.serving.kv_events import (
    deterministic_block_hash,
    tokenids_to_block_hashes,
    KVBlockEventPublisher,
)


def test_deterministic_block_hash_consistency():
    """Test that deterministic_block_hash produces consistent 64-bit unsigned hashes."""
    toks1 = [101, 202, 303, 404]
    toks2 = [101, 202, 303, 404]
    toks3 = [101, 202, 303, 405]

    h1 = deterministic_block_hash(toks1, seed=0)
    h2 = deterministic_block_hash(toks2, seed=0)
    h3 = deterministic_block_hash(toks3, seed=0)

    assert isinstance(h1, int)
    assert h1 > 0
    assert h1 == h2
    assert h1 != h3


def test_tokenids_to_block_hashes_chaining():
    """Test that tokenids_to_block_hashes chunks sequence and chains parent hashes."""
    toks = list(range(150))
    blocks = tokenids_to_block_hashes(toks, block_size=64)

    assert len(blocks) == 3
    assert len(blocks[0]["token_ids"]) == 64
    assert len(blocks[1]["token_ids"]) == 64
    assert len(blocks[2]["token_ids"]) == 22

    assert blocks[0]["parent_hash"] == 0
    assert blocks[1]["parent_hash"] == blocks[0]["hash"]
    assert blocks[2]["parent_hash"] == blocks[1]["hash"]


def test_kv_block_event_publisher_lifecycle():
    """Test that KVBlockEventPublisher starts, publishes events, and cleans up."""
    pub = KVBlockEventPublisher(
        model_name="test-model",
        pod_ip="127.0.0.1",
        pod_port=8000,
        pub_endpoint="tcp://127.0.0.1:55990",
        replay_endpoint="tcp://127.0.0.1:55991",
    )

    assert pub.topic == "kv@127.0.0.1:8000@test-model"

    # Start publisher
    pub.start()

    # Emit events
    pub.on_block_stored([12345, 67890], parent_hash=0, token_ids=[1, 2, 3])
    pub.on_block_removed([12345])
    pub.on_all_blocks_cleared()

    # Replay buffer check
    assert len(pub._replay_buffer) == 3

    # Stop cleanly
    pub.stop()
    assert not pub._started
