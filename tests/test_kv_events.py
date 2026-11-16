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


def _free_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_engine_prefix_cache_events_reach_a_zmq_subscriber():
    """Two requests sharing a 32-token prefix through the real engine: a ZMQ SUB socket receives vLLM-format
    msgpack BlockStored events whose hashes, parents and tokens are the prefix cache's pages, then BlockRemoved on
    eviction and AllBlocksCleared on clear; the ROUTER socket replays them."""
    import asyncio
    import msgpack
    import zmq
    from turing.config import TuringConfig
    from turing.models.registry import get_model_config
    from turing.serving.engine import ContinuousBatchEngine
    from turing.serving.kv_events import attach_kv_events
    from turing.serving.prefix_cache import page_hashes

    engine = ContinuousBatchEngine(get_model_config("test-tiny"), TuringConfig(device="cpu"))
    pub_port, replay_port = _free_port(), _free_port()
    pub = KVBlockEventPublisher("test-tiny", pod_ip="127.0.0.1", pod_port=8000,
                                pub_endpoint=f"tcp://127.0.0.1:{pub_port}",
                                replay_endpoint=f"tcp://127.0.0.1:{replay_port}")
    attach_kv_events(engine, pub)
    pub.start()
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://127.0.0.1:{pub_port}")
    sub.setsockopt(zmq.SUBSCRIBE, b"kv@")
    time.sleep(0.3)  # let the subscription reach the publisher

    shared = list(range(100, 132))
    p1, p2 = shared + list(range(200, 220)), shared + list(range(300, 320))

    async def run():
        await engine.start()
        for p in (p1, p2):
            async for _ in engine.stream_generate(p, max_new_tokens=2, temperature=0.0):
                pass
        await engine.stop()

    asyncio.run(run())
    engine.prefix_cache.evict(1)
    engine.prefix_cache.clear()

    events, seqs = [], []
    while sub.poll(1000):
        topic, seq, payload = sub.recv_multipart()
        assert topic == b"kv@127.0.0.1:8000@test-tiny"
        ts, batch = msgpack.unpackb(payload)
        assert isinstance(ts, float)
        events += batch
        seqs.append(int.from_bytes(seq, "big"))
    assert seqs == list(range(len(seqs)))

    stored = [e for e in events if e[0] == "BlockStored"]
    h1, h2 = page_hashes(p1), page_hashes(p2)
    # request 1's prefill stores its 3 full pages from the root (parent None), 16 tokens each
    assert stored[0][1:] == [h1, None, p1[:48], 16, None, "GPU", None]
    # request 2 reuses the 2 shared pages: only its third page is stored, chained to the shared prefix's last page
    second = next(e for e in stored if h2[2] in e[1])
    assert second[1][0] == h2[2] and second[2] == h1[1] == h2[1] and second[3][:16] == p2[32:48]
    assert all(len(e[3]) == 16 * len(e[1]) and e[4] == 16 for e in stored)
    removed = [e for e in events if e[0] == "BlockRemoved"]
    assert len(removed) == 1 and len(removed[0][1]) == 1 and removed[0][2] == "GPU"
    assert events[-1] == ["AllBlocksCleared"]

    # replay from sequence 1: everything after the first message, then the end marker
    dealer = ctx.socket(zmq.DEALER)
    dealer.connect(f"tcp://127.0.0.1:{replay_port}")
    dealer.send_multipart([b"", (1).to_bytes(8, "big")])
    replayed = []
    while dealer.poll(2000):
        frames = dealer.recv_multipart()
        if frames[2] == KVBlockEventPublisher.END_SEQ:
            break
        replayed.append(int.from_bytes(frames[2], "big"))
    assert replayed == seqs[1:]

    for s in (sub, dealer):
        s.close(linger=0)
    ctx.term()
    pub.stop()


def test_attach_kv_events_needs_the_prefix_cache():
    from turing.config import TuringConfig
    from turing.models.registry import get_model_config
    from turing.serving.engine import ContinuousBatchEngine
    from turing.serving.kv_events import attach_kv_events

    engine = ContinuousBatchEngine(get_model_config("test-tiny"), TuringConfig(device="cpu"), prefix_caching=False)
    with pytest.raises(ValueError):
        attach_kv_events(engine, KVBlockEventPublisher("m", pub_endpoint="tcp://127.0.0.1:0",
                                                       replay_endpoint="tcp://127.0.0.1:0"))
