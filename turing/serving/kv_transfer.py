"""
KV page transfer between processes: export one request's pages from a KVPool, import them into another pool.

The payload carries the request's length and, for every layer, its pages exactly as the pool stores them: K and V in
the pool's storage dtype (native, or int8 / int4 codes) plus the low-bit scales and zeros, and the Quest page bounds
when both pools track them. Nothing is recomputed or re-quantized, so the importing pool reads the same values the
exporting pool would have read.

Wire format of a message (`send_msg` / `recv_msg`, over any connected socket):
    [u64 header length][header: UTF-8 JSON][u64 blob length][blob]
A KV payload's header lists each tensor's field name, dtype and shape; the blob is their raw bytes in that order.
"""

import json
import socket
import struct
from typing import Dict, Tuple

import torch

from ..core.kv_cache import PAGE, KVCache, KVPool

__all__ = ["export_kv", "import_kv", "send_msg", "recv_msg"]

_FIELDS = ("k", "v", "k_scale", "k_zero", "v_scale", "v_zero", "k_hi", "v_hi")


def export_kv(cache: KVCache) -> Tuple[dict, bytes]:
    """(header, blob) of the pages holding cache.length tokens, all layers."""
    pool = cache.pool
    pages = torch.tensor(cache.pages[:-(-cache.length // PAGE)], dtype=torch.long, device=pool.device)
    names = [n for n in _FIELDS if getattr(pool, n) is not None] + (["kmin", "kmax"] if pool.track_pages else [])
    tensors = [getattr(pool, n)[:, pages].contiguous().cpu() for n in names]
    header = {"length": cache.length, "kv_cache_dtype": pool.kv_cache_dtype, "num_layers": pool.num_layers,
              "native_layers": list(pool.native_layers),
              "num_kv_heads": pool.num_kv_heads, "head_dim": pool.head_dim,
              "tensors": [[n, str(t.dtype).replace("torch.", ""), list(t.shape)] for n, t in zip(names, tensors)]}
    return header, b"".join(t.view(torch.uint8).numpy().tobytes() for t in tensors)


def import_kv(pool: KVPool, header: dict, blob: bytes) -> KVCache:
    """A new request cache in `pool` holding the exported pages (the pools must agree on layers, KV heads, head size
    and storage dtype)."""
    if list(header.get("native_layers", [])) != list(pool.native_layers):
        raise ValueError(f"KV payload keeps layers {header.get('native_layers', [])} in full precision, this pool "
                         f"{list(pool.native_layers)}")
    for key in ("kv_cache_dtype", "num_layers", "num_kv_heads", "head_dim"):
        if header[key] != getattr(pool, key):
            raise ValueError(f"KV payload has {key}={header[key]!r}, this pool {getattr(pool, key)!r}")
    if not isinstance(header["length"], int) or header["length"] < 0:
        raise ValueError(f"KV payload length {header['length']!r}")
    cache = KVCache(pool)
    try:
        cache.reserve(header["length"])
        pages = torch.tensor(cache.pages, dtype=torch.long, device=pool.device)
        buf = blob if isinstance(blob, bytearray) else bytearray(blob)  # torch.frombuffer wants a writable buffer
        got, offset = set(), 0
        for name, dtype, shape in header["tensors"]:
            dst = getattr(pool, name) if name in _FIELDS + ("kmin", "kmax") else None
            if dst is None and name in ("kmin", "kmax") and dtype == "float16":  # Quest bounds this pool skips
                offset += torch.Size(shape).numel() * 2
                continue
            n = torch.Size(shape).numel() * (dst.element_size() if dst is not None else 0)
            if dst is None or str(dst.dtype) != "torch." + dtype or offset + n > len(buf) \
                    or shape != [dst.shape[0], len(cache.pages), *dst.shape[2:]]:
                raise ValueError(f"malformed KV payload tensor {name} {dtype} {shape}")
            if n:
                t = torch.frombuffer(buf, dtype=torch.uint8, count=n, offset=offset)
                dst[:, pages] = t.view(dst.dtype).view(shape).to(pool.device)
            got.add(name)
            offset += n
        need = {n for n in _FIELDS if getattr(pool, n) is not None} | ({"kmin", "kmax"} if pool.track_pages else set())
        if not need <= got:
            raise ValueError(f"KV payload lacks {sorted(need - got)}")
    except Exception:
        cache.free()
        raise
    cache.length = header["length"]
    return cache


def _recv_exact(sock: socket.socket, n: int) -> bytearray:
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("peer closed the connection mid-message")
        got += k
    return buf


def send_msg(sock: socket.socket, header: Dict, blob: bytes = b"") -> None:
    h = json.dumps(header).encode()
    sock.sendall(struct.pack("<Q", len(h)) + h + struct.pack("<Q", len(blob)))
    sock.sendall(blob)


def recv_msg(sock: socket.socket, max_header: int = 1 << 24, max_blob: int = 1 << 40) -> Tuple[dict, bytearray]:
    """One message; a header or blob over its limit is refused before anything is allocated for it."""
    (n,) = struct.unpack("<Q", _recv_exact(sock, 8))
    if n > max_header:
        raise ValueError(f"message header of {n} bytes")
    header = json.loads(_recv_exact(sock, n))
    (m,) = struct.unpack("<Q", _recv_exact(sock, 8))
    if m > max_blob:
        raise ValueError(f"message blob of {m} bytes")
    return header, _recv_exact(sock, m)
