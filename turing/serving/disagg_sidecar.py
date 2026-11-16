"""
Prefill/decode disaggregation: one process prefills prompts, another decodes, and the request's KV pages travel
between them over TCP (turing/serving/kv_transfer.py).

- Prefill side: `PrefillWorker(engine).serve_forever()` listens on a port. Per request it runs the prompt through the
  model into a fresh cache (in the engine's prefill chunks), samples the first token with the request's settings,
  sends back the token and the cache's pages, and frees them.
- Decode side: `add_remote_prefilled_request(engine, (host, port), prompt, ...)` asks a prefill worker for the
  prompt, imports the pages into this engine's pool, and admits the request to continue after the first token:
  the engine's output is every token after it. Greedy output is identical to one engine doing both.

Both engines must run the same model with the same kv_cache_dtype (import_kv checks the pools agree).
"""

import asyncio
import socket
from typing import List, Optional, Set, Tuple

import torch

from ..core.kv_cache import KVCache
from .engine import AsyncSequenceRequest, ContinuousBatchEngine
from .kv_transfer import export_kv, import_kv, recv_msg, send_msg

__all__ = ["PrefillWorker", "remote_prefill", "add_remote_prefilled_request"]


class PrefillWorker:
    """Answers prefill requests on a TCP port, one connection at a time; a connection may send several requests.
    Request header: {"prompt": [ids], "temperature", "top_k", "top_p"}; reply: {"token": first token} plus the KV
    payload header under "kv" and its blob, or {"error": message}. The worker owns its engine's model and pool: do
    not run the engine's scheduler loop in the same process."""

    def __init__(self, engine: ContinuousBatchEngine, host: str = "127.0.0.1", port: int = 0):
        self.engine = engine
        self.sock = socket.create_server((host, port))
        self.port = self.sock.getsockname()[1]

    def prefill(self, prompt: List[int], temperature: float = 0.0, top_k: int = 50,
                top_p: float = 1.0) -> Tuple[int, dict, bytes]:
        eng = self.engine
        if not prompt or not all(isinstance(t, int) and 0 <= t < eng.model_config.vocab_size for t in prompt):
            raise ValueError("prompt must be a non-empty list of token ids in the vocabulary")
        cache = KVCache(eng.kv_pool)
        try:
            with torch.inference_mode():
                for s in range(0, len(prompt), eng.prefill_chunk_size):
                    ids = torch.tensor([prompt[s:s + eng.prefill_chunk_size]], dtype=torch.long, device=eng.device)
                    logits, cache = eng.model(ids, past_key_values=cache, logits_to_keep=1)
                req = AsyncSequenceRequest(prompt, temperature=temperature, top_k=top_k, top_p=top_p)
                token = eng._sample([req], logits[:, -1])[0]
            header, blob = export_kv(cache)
        finally:
            cache.free()
        return token, header, blob

    def handle(self, conn: socket.socket) -> None:
        while True:
            try:
                msg, _ = recv_msg(conn, max_header=1 << 26, max_blob=0)  # a request is a prompt, no blob
            except (ConnectionError, ValueError):  # the client is done, or sent something that is not a request
                return
            try:
                token, header, blob = self.prefill(msg["prompt"], float(msg.get("temperature", 0.0)),
                                                   int(msg.get("top_k", 50)), float(msg.get("top_p", 1.0)))
            except Exception as e:  # a bad request or a full pool: report it, keep serving
                send_msg(conn, {"error": f"{type(e).__name__}: {e}"})
                continue
            send_msg(conn, {"token": token, "kv": header}, blob)

    def serve_forever(self) -> None:
        while True:
            conn, _ = self.sock.accept()
            with conn:
                self.handle(conn)


def remote_prefill(addr: Tuple[str, int], prompt: List[int], temperature: float = 0.0, top_k: int = 50,
                   top_p: float = 1.0, timeout: float = 300.0) -> Tuple[int, dict, bytes]:
    """(first token, KV header, KV blob) of a prompt from the PrefillWorker at addr."""
    with socket.create_connection(addr, timeout=timeout) as sock:
        send_msg(sock, {"prompt": list(prompt), "temperature": temperature, "top_k": top_k, "top_p": top_p})
        reply, blob = recv_msg(sock)
    if "error" in reply:
        raise RuntimeError(f"prefill worker {addr[0]}:{addr[1]}: {reply['error']}")
    return reply["token"], reply["kv"], blob


async def add_remote_prefilled_request(engine: ContinuousBatchEngine, addr: Tuple[str, int], prompt: List[int],
                                       max_new_tokens: int = 32, temperature: float = 0.0, top_k: int = 50,
                                       top_p: float = 1.0, stop_token_ids: Optional[Set[int]] = None,
                                       ) -> Tuple[int, Optional[AsyncSequenceRequest]]:
    """Prefill `prompt` on the worker at addr and continue it on `engine`. Returns the first token and the engine
    request producing the tokens after it, or None when the first token already ends the request (it is a stop token,
    which is not part of the output, or max_new_tokens is 1)."""
    if engine.prefix_cache is not None:
        # the engine would index the imported pages under the continuation's tokens alone
        raise ValueError("a decode engine for remote prefill needs prefix_caching=False")
    token, header, blob = await asyncio.to_thread(remote_prefill, addr, prompt, temperature, top_k, top_p)
    stop = engine.eos_token_ids if stop_token_ids is None else set(stop_token_ids)
    if token in stop or max_new_tokens <= 1:
        return token, None
    cache = import_kv(engine.kv_pool, header, blob)
    try:  # the request's cache shares the imported pages copy-on-write; ours is freed below
        req = await engine.add_request([token], max_new_tokens=max_new_tokens - 1, temperature=temperature,
                                       top_k=top_k, top_p=top_p, stop_token_ids=stop, prefix_kv=cache)
    finally:
        cache.free()
    return token, req
