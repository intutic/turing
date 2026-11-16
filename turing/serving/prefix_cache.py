"""
Automatic prefix caching over the paged KV pool. Every full 16-token page of a finished prefill (and of a finished
request's prompt + output) is indexed by a chained hash of its tokens: page i's hash covers the hash of page i - 1,
the page's 16 token ids, and the LoRA adapter, so equal hashes mean equal KV. A new request reuses the longest run of
indexed pages from the start of its prompt copy-on-write and prefills only the rest. The index holds one reference
on each page; pages no request uses are evicted least recently used when the pool needs room. Stored and removed
pages are reported to on_event(kind, hashes, parent, token_ids) for llm-d's KV-event subscribers.
"""

import hashlib
import struct
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple

from ..core.kv_cache import PAGE, KVCache, KVPool

ROOT = 0


def page_hashes(tokens: List[int], adapter: Optional[str] = None) -> List[int]:
    """Chained 64-bit hashes of tokens' full pages."""
    out, parent = [], _root(adapter)
    for i in range(len(tokens) // PAGE):
        h = hashlib.blake2b(struct.pack("<Q", parent) + struct.pack(f"<{PAGE}q", *tokens[i * PAGE:(i + 1) * PAGE]),
                            digest_size=8)
        parent = int.from_bytes(h.digest(), "little") & 0x7FFF_FFFF_FFFF_FFFF
        out.append(parent)
    return out


def _root(adapter: Optional[str]) -> int:
    if not adapter:
        return ROOT
    return int.from_bytes(hashlib.blake2b(adapter.encode(), digest_size=8).digest(), "little") & 0x7FFF_FFFF_FFFF_FFFF


class PrefixCache:
    def __init__(self, pool: KVPool, on_event: Optional[Callable] = None):
        self.pool = pool
        self.entries: "OrderedDict[int, int]" = OrderedDict()  # hash -> page id, least recently used first
        self.on_event = on_event
        self.hits = self.queries = 0  # tokens served from the cache / prompt tokens looked up

    def lookup(self, tokens: List[int], adapter: Optional[str] = None, limit: Optional[int] = None) -> List[int]:
        """Pages holding the longest cached prefix of tokens (at most `limit` tokens), most recently used now."""
        pages = []
        for h in page_hashes(tokens[:limit] if limit is not None else tokens, adapter):
            page = self.entries.get(h)
            if page is None:
                break
            self.entries.move_to_end(h)
            pages.append(page)
        self.queries += len(tokens)
        self.hits += len(pages) * PAGE
        return pages

    def attach(self, tokens: List[int], adapter: Optional[str] = None, limit: Optional[int] = None) -> KVCache:
        """A new cache that starts with the cached prefix of tokens (shared pages, one reference each)."""
        pages = self.lookup(tokens, adapter, limit)
        cache = KVCache(self.pool)
        if pages:
            import torch
            self.pool.incref(pages)
            cache.pages = list(pages)
            self.pool.page_table[cache.slot, :len(pages)] = torch.tensor(pages, dtype=torch.int32)
            cache.length = len(pages) * PAGE
        return cache

    def insert(self, tokens: List[int], cache: KVCache, adapter: Optional[str] = None) -> None:
        """Index the full pages of cache, whose first len(tokens) positions hold these tokens."""
        n = min(len(tokens), cache.length) // PAGE
        hashes = page_hashes(tokens[:n * PAGE], adapter)
        stored, parent = [], _root(adapter)
        for i, h in enumerate(hashes):
            if h in self.entries:
                self.entries.move_to_end(h)
            else:
                page = cache.pages[i]
                self.pool.incref([page])
                self.entries[h] = page
                stored.append((h, parent, tokens[i * PAGE:(i + 1) * PAGE]))
            parent = h
        if stored and self.on_event:
            self.on_event("stored", [h for h, _, _ in stored], stored[0][1], [t for _, _, ts in stored for t in ts])

    def evict(self, pages_needed: int) -> int:
        """Drop least recently used entries whose page no request holds until pages_needed pages are free (or no
        such entry is left). Returns the pages freed."""
        freed, removed = 0, []
        for h in list(self.entries):
            if freed >= pages_needed:
                break
            page = self.entries[h]
            if self.pool._refs[page] == 1:  # only the index holds it
                del self.entries[h]
                self.pool.decref([page])
                removed.append(h)
                freed += 1
        if removed and self.on_event:
            self.on_event("removed", removed, None, None)
        return freed

    def clear(self) -> None:
        if self.entries and self.on_event:
            self.on_event("cleared", list(self.entries), None, None)
        self.pool.decref(list(self.entries.values()))
        self.entries.clear()

    def stats(self) -> dict:
        return {"cached_pages": len(self.entries), "hit_tokens": self.hits, "queried_tokens": self.queries,
                "hit_rate": round(self.hits / self.queries, 4) if self.queries else None}
