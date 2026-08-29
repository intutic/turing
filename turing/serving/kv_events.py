"""
ZeroMQ KV Block Event Publisher for llm-d Router Integration.
Publishes real-time KV cache block mutations (store/remove/clear) over ZMQ PUB socket,
enabling llm-d's EPP router to build a live prefix-cache affinity index for Turing Engine pods.
"""

import struct
import json
import time
import hashlib
import threading
from collections import deque
from typing import List, Optional, Dict, Any

try:
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
    Publishes KV cache block events over ZeroMQ matching the llm-d wire protocol.

    Transport topology:
    - PUB socket (default tcp://*:5556): Live event stream
    - ROUTER socket (default tcp://*:5559): Replay buffer for EPP recovery

    Topic format: kv@<POD_IP>:<POD_PORT>@<MODEL_NAME>
    """

    def __init__(
        self,
        model_name: str,
        pod_ip: str = "0.0.0.0",
        pod_port: int = 8000,
        pub_endpoint: str = "tcp://*:5556",
        replay_endpoint: str = "tcp://*:5559",
        replay_buffer_size: int = 10000,
        block_size: int = 64,
    ):
        self.model_name = model_name
        self.pod_ip = pod_ip
        self.pod_port = pod_port
        self.pub_endpoint = pub_endpoint
        self.replay_endpoint = replay_endpoint
        self.block_size = block_size
        self.topic = f"kv@{pod_ip}:{pod_port}@{model_name}"

        self._replay_buffer: deque = deque(maxlen=replay_buffer_size)
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

        if not HAS_ZMQ:
            return

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
        parent_hash: int,
        token_ids: List[int],
        tier: str = "gpu",
        lora_name: Optional[str] = None,
    ):
        """Publish a BlockStored event when new KV blocks are cached."""
        event = {
            "type": "BlockStored",
            "block_hashes": block_hashes,
            "parent_hash": parent_hash,
            "token_ids": token_ids,
            "tier": tier,
            "timestamp": time.time(),
        }
        if lora_name:
            event["lora_name"] = lora_name
        self._publish(event)

    def on_block_removed(
        self,
        block_hashes: List[int],
        tier: str = "gpu",
    ):
        """Publish a BlockRemoved event when KV blocks are evicted."""
        event = {
            "type": "BlockRemoved",
            "block_hashes": block_hashes,
            "tier": tier,
            "timestamp": time.time(),
        }
        self._publish(event)

    def on_all_blocks_cleared(self):
        """Publish an AllBlocksCleared event (e.g., on model reload)."""
        event = {
            "type": "AllBlocksCleared",
            "timestamp": time.time(),
        }
        self._publish(event)

    def _publish(self, event: Dict[str, Any]):
        """Serialize and publish event on PUB socket, buffer for replay."""
        payload = json.dumps(event).encode("utf-8")
        with self._lock:
            self._replay_buffer.append(payload)

        if not self._started or self._pub_socket is None or not HAS_ZMQ:
            return

        topic_bytes = self.topic.encode("utf-8")
        try:
            self._pub_socket.send_multipart([topic_bytes, payload], flags=zmq.NOBLOCK)
        except Exception:
            pass  # Drop event if HWM exceeded rather than blocking


    def _replay_listener(self):
        """Background thread serving replay requests from EPP on ROUTER socket."""
        while self._started and self._replay_socket is not None:
            try:
                if self._replay_socket.poll(timeout=500):
                    frames = self._replay_socket.recv_multipart()
                    if len(frames) >= 2:
                        identity = frames[0]
                        # Replay all buffered events
                        with self._lock:
                            events = list(self._replay_buffer)
                        for payload in events:
                            try:
                                self._replay_socket.send_multipart(
                                    [identity, b"", payload], flags=zmq.NOBLOCK
                                )
                            except zmq.ZMQError:
                                break
                        # Send end-of-replay marker
                        try:
                            self._replay_socket.send_multipart(
                                [identity, b"", b"END_REPLAY"], flags=zmq.NOBLOCK
                            )
                        except zmq.ZMQError:
                            pass
            except zmq.ZMQError:
                if not self._started:
                    break

    @property
    def is_active(self) -> bool:
        return self._started and HAS_ZMQ


def tokenids_to_block_hashes(
    token_ids: List[int], block_size: int = 64
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
