"""
ZeroMQ KV Block Event Publisher for llm-d Router Integration.
Publishes real-time KV cache block mutations (store/remove/clear) over ZMQ PUB socket,
enabling llm-d's EPP router to build a live prefix-cache affinity index for Turing Engine pods.
"""

import struct
import time
import hashlib
import threading
from collections import deque
from typing import List, Optional, Dict, Any

try:
    import msgpack
    import zmq
    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False


def deterministic_block_hash(token_ids: List[int], seed: int = 0) -> int:
    """
    Deterministic 64-bit hash of a token ID sequence.
    Uses native C++20 xxHash64 / SHA-NI when available with zero allocations,
    falling back to SHA-256 truncated to 64 bits.
    """
    try:
        import turing.turing_csrc as turing_csrc
        return int(turing_csrc.deterministic_token_hash_cpu(token_ids, seed))
    except Exception:
        data = struct.pack(f"<Q{'I' * len(token_ids)}", seed, *token_ids)
        digest = hashlib.sha256(data).digest()
        return struct.unpack("<Q", digest[:8])[0]



class KVBlockEventPublisher:
    """
    Publishes KV cache block events over ZeroMQ in vLLM's wire format, which llm-d's KV-cache indexer reads
    (vllm/distributed/kv_events.py, ZmqEventPublisher):

    - PUB socket (default tcp://*:5556): multipart (topic, sequence number as 8 big-endian bytes, payload), the
      payload a msgpack array [timestamp, [event, ...]] with each event a tagged array:
      ["BlockStored", block_hashes, parent_block_hash, token_ids, block_size, lora_id, medium, lora_name],
      ["BlockRemoved", block_hashes, medium] or ["AllBlocksCleared"].
    - ROUTER socket (default tcp://*:5559): a request (client id, b"", start sequence as 8 big-endian bytes) is
      answered with every buffered message from that sequence on, (client id, b"", topic, seq, payload), then
      (client id, b"", b"", -1 as 8 signed big-endian bytes, b"").

    Topic: kv@<POD_IP>:<POD_PORT>@<MODEL_NAME> (llm-d reads the pod and the model from it).
    """

    END_SEQ = (-1).to_bytes(8, "big", signed=True)

    def __init__(
        self,
        model_name: str,
        pod_ip: str = "0.0.0.0",
        pod_port: int = 8000,
        pub_endpoint: str = "tcp://*:5556",
        replay_endpoint: str = "tcp://*:5559",
        replay_buffer_size: int = 10000,
        block_size: int = 16,  # the KV pool's page size
        medium: str = "GPU",
    ):
        if not HAS_ZMQ:
            raise RuntimeError("KV events need pyzmq and msgpack: pip install 'turing-engine[llmd]'")
        self.model_name = model_name
        self.pod_ip = pod_ip
        self.pod_port = pod_port
        self.pub_endpoint = pub_endpoint
        self.replay_endpoint = replay_endpoint
        self.block_size = block_size
        self.medium = medium
        self.topic = f"kv@{pod_ip}:{pod_port}@{model_name}"

        self._replay_buffer: deque = deque(maxlen=replay_buffer_size)  # (seq, payload)
        self._seq = 0
        self._started = False
        self._lock = threading.Lock()

        self._ctx: Optional[Any] = None
        self._pub_socket: Optional[Any] = None
        self._replay_socket: Optional[Any] = None
        self._replay_thread: Optional[threading.Thread] = None

    def start(self):
        """Bind ZMQ sockets and start replay listener thread."""
        if self._started:
            return
        self._started = True

        self._ctx = zmq.Context()

        # PUB socket for live events
        self._pub_socket = self._ctx.socket(zmq.PUB)
        self._pub_socket.setsockopt(zmq.SNDHWM, 50000)
        self._pub_socket.bind(self.pub_endpoint)

        # ROUTER socket for replay requests
        self._replay_socket = self._ctx.socket(zmq.ROUTER)
        self._replay_socket.bind(self.replay_endpoint)

        # Start replay listener in background thread
        self._replay_thread = threading.Thread(target=self._replay_listener, daemon=True)
        self._replay_thread.start()

    def stop(self):
        """Close ZMQ sockets and context."""
        self._started = False
        if self._replay_thread is not None:
            self._replay_thread.join(timeout=2)
            self._replay_thread = None
        if self._pub_socket is not None:
            self._pub_socket.close(linger=100)
            self._pub_socket = None
        if self._replay_socket is not None:
            self._replay_socket.close(linger=100)
            self._replay_socket = None
        if self._ctx is not None:
            self._ctx.term()
            self._ctx = None

    def on_block_stored(
        self,
        block_hashes: List[int],
        parent_hash: Optional[int],
        token_ids: List[int],
        lora_name: Optional[str] = None,
    ):
        """Publish a BlockStored event: full blocks of block_size tokens newly cached, token_ids their tokens in
        order, parent_hash the block before the first (None at the start of a sequence)."""
        self._publish(["BlockStored", list(block_hashes), parent_hash, list(token_ids), self.block_size, None,
                       self.medium, lora_name])

    def on_block_removed(self, block_hashes: List[int]):
        """Publish a BlockRemoved event when KV blocks are evicted."""
        self._publish(["BlockRemoved", list(block_hashes), self.medium])

    def on_all_blocks_cleared(self):
        """Publish an AllBlocksCleared event (e.g., on model reload)."""
        self._publish(["AllBlocksCleared"])

    def _publish(self, event: list):
        """Serialize one event as a batch, publish it on the PUB socket and buffer it for replay."""
        payload = msgpack.packb([time.time(), [event]])
        with self._lock:
            seq = self._seq
            self._seq += 1
            self._replay_buffer.append((seq, payload))

        if not self._started or self._pub_socket is None:
            return
        try:
            self._pub_socket.send_multipart([self.topic.encode(), seq.to_bytes(8, "big"), payload],
                                            flags=zmq.NOBLOCK)
        except zmq.ZMQError:
            pass  # Drop event if HWM exceeded rather than blocking

    def _replay_listener(self):
        """Background thread serving replay requests on the ROUTER socket."""
        while self._started and self._replay_socket is not None:
            try:
                if not self._replay_socket.poll(timeout=100):
                    continue
                frames = self._replay_socket.recv_multipart()
                if len(frames) != 3:
                    continue
                identity, start = frames[0], int.from_bytes(frames[2], "big")
                with self._lock:
                    events = [(s, p) for s, p in self._replay_buffer if s >= start]
                for seq, payload in events:
                    self._replay_socket.send_multipart(
                        [identity, b"", self.topic.encode(), seq.to_bytes(8, "big"), payload])
                self._replay_socket.send_multipart([identity, b"", b"", self.END_SEQ, b""])
            except zmq.ZMQError:
                if not self._started:
                    break

    @property
    def is_active(self) -> bool:
        return self._started and HAS_ZMQ


def attach_kv_events(engine, publisher: KVBlockEventPublisher) -> None:
    """Publish the engine's prefix-cache changes: pages indexed (BlockStored, with their chained hashes, the hash of
    the page before, their token ids), pages evicted (BlockRemoved) and the index cleared (AllBlocksCleared)."""
    if engine.prefix_cache is None:
        raise ValueError("KV events need the engine's prefix cache, which is off (distributed serving or "
                         "prefix_caching=False)")
    publisher.block_size = engine.kv_pool.page_size

    def on_event(kind, hashes, parent, token_ids):
        if kind == "stored":
            # parent 0 is the root of a sequence; with a LoRA adapter the root is that adapter's hash
            publisher.on_block_stored(hashes, parent or None, token_ids)
        elif kind == "removed":
            publisher.on_block_removed(hashes)
        else:  # "cleared"
            publisher.on_all_blocks_cleared()

    engine.prefix_cache.on_event = on_event


def tokenids_to_block_hashes(
    token_ids: List[int], block_size: int = 16
) -> List[Dict[str, Any]]:
    """
    Chunk a token ID sequence into block-sized segments and compute
    deterministic hashes for each block, with parent hash chaining.

    Returns list of dicts: [{"hash": int, "parent_hash": int, "token_ids": [int]}]
    """
    blocks = []
    parent_hash = 0
    for i in range(0, len(token_ids), block_size):
        chunk = token_ids[i : i + block_size]
        block_hash = deterministic_block_hash(chunk, seed=parent_hash)
        blocks.append({
            "hash": block_hash,
            "parent_hash": parent_hash,
            "token_ids": chunk,
        })
        parent_hash = block_hash
    return blocks
