#!/usr/bin/env python3
"""
Storage ingestion ladder, measured cold.

Every cold row reads a file none of which is in the OS file cache, checked with mincore before and after:
- macOS: the file is written fresh through an F_NOCACHE descriptor before each run (a new inode has no cached pages).
- Linux: the file is written, fsynced, then dropped with posix_fadvise(POSIX_FADV_DONTNEED).
A run is reported only if the resident fraction before it is below 1%.

Rows, each reading the same tensor-shaped segments of the file into host memory:
- Tier 0: plain mmap, copied page by page (demand paging, default kernel readahead)
- Tier 1: mmap + madvise(MADV_WILLNEED), then copied
- Tier 2: NativeAsyncRingReader.read_segments_parallel (returns bytes; available in every build)
- Tier 2: NativeAsyncRingReader.read_segments_into (reads straight into a caller buffer; new builds only),
  also at queue depths 1 and 8
- Tier 5: Tier 2 into a buffer again, with the file fully in the page cache (warm)
Tier 3 (.tgate4 compressed container) and Tier 4 (GPUDirect Storage) are reported as not measured, with the reason.

python scripts/benchmark_ingest_ladder_all_tiers.py [--size-mib 1024] [--repeats 3] [--queue-depth 64] [--out FILE]
"""
import argparse
import ctypes
import json
import mmap
import os
import platform
import statistics
import sys
import time

import numpy as np

PAGE = mmap.PAGESIZE
# One transformer layer of a 7B model in bf16 (q, k, v, o, gate, up, down, two norms), in bytes.
LAYER = [28 << 20, 4 << 20, 4 << 20, 28 << 20, 130 << 20, 130 << 20, 130 << 20, 7 << 10, 7 << 10]

_libc = ctypes.CDLL(None, use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]


def resident_fraction(path):
    """Fraction of the file's pages in the OS file cache (mincore on a fresh shared mapping)."""
    size = os.path.getsize(path)
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = _libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if addr in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_errno(), "mmap failed")
        try:
            vec = (ctypes.c_ubyte * ((size + PAGE - 1) // PAGE))()
            if _libc.mincore(ctypes.c_void_p(addr), size, vec) != 0:
                raise OSError(ctypes.get_errno(), "mincore failed")
            return float(np.count_nonzero(np.frombuffer(vec, dtype=np.uint8) & 1)) / len(vec)
        finally:
            _libc.munmap(ctypes.c_void_p(addr), size)
    finally:
        os.close(fd)


def write_cold(path, size, seed):
    """Write `size` random bytes so that none of them is left in the file cache. Returns the method used."""
    if os.path.exists(path):
        os.unlink(path)  # a new inode starts with no cached pages
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    method = "plain write"
    try:
        if sys.platform == "darwin":
            import fcntl
            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
            method = "F_NOCACHE write"
        rng = np.random.default_rng(seed)
        left = size
        while left:
            view = memoryview(rng.bytes(min(left, 16 << 20)))
            left -= len(view)
            while view:
                view = view[os.write(fd, view):]
        os.fsync(fd)
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            method += " + fsync + POSIX_FADV_DONTNEED"
    finally:
        os.close(fd)
    return method


def segments(size):
    """Tensor-shaped (offset, length) spans covering the file, repeating LAYER."""
    offs, lens, off, i = [], [], 0, 0
    while off < size:
        n = min(LAYER[i % len(LAYER)], size - off)
        offs.append(off)
        lens.append(n)
        off, i = off + n, i + 1
    return offs, lens


def read_mmap(path, size, out, willneed):
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            if willneed:
                mm.madvise(mmap.MADV_WILLNEED)
            out[:size] = np.frombuffer(mm, dtype=np.uint8)
        finally:
            mm.close()


def time_read(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def summarize(name, walls, size, extra):
    gbps = [size / w / 1e9 for w in walls]
    return dict(tier=name, measured=True, n=len(walls), bytes=size,
                gb_per_s=statistics.mean(gbps),
                gb_per_s_se=statistics.stdev(gbps) / len(gbps) ** 0.5 if len(gbps) > 1 else None,
                latency_ms=statistics.mean(walls) * 1e3, **extra)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size-mib", type=int, default=1024)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--queue-depth", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dir", default=None, help="folder for the scratch file (default: a *.noindex folder in the repo)")
    ap.add_argument("--label", default=None, help="label stored in the results, e.g. the reader build")
    ap.add_argument("--out", default="results_ingest_ladder.json")
    a = ap.parse_args(argv)

    from turing.turing_csrc import NativeAsyncRingReader

    size = a.size_mib << 20
    folder = a.dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "ingest_ladder.noindex")
    os.makedirs(folder, exist_ok=True)  # *.noindex keeps Spotlight from reading the file behind our back
    path = os.path.join(folder, "ladder.bin")
    offs, lens = segments(size)
    reader = NativeAsyncRingReader(a.workers, a.queue_depth)
    out = np.empty(size, dtype=np.uint8)
    out.fill(0)  # fault the destination in before timing

    cold = {
        "Tier 0: mmap demand paging": lambda: read_mmap(path, size, out, False),
        "Tier 1: mmap + MADV_WILLNEED": lambda: read_mmap(path, size, out, True),
        "Tier 2: ring reader, read_segments_parallel (bytes)": lambda: reader.read_segments_parallel(path, offs, lens),
    }
    if hasattr(reader, "read_segments_into"):
        cold["Tier 2: ring reader, read_segments_into (buffer)"] = lambda: reader.read_segments_into(path, offs, lens, out)
        for qd in (1, 8):  # how much of the gain comes from requests in flight
            r_qd = NativeAsyncRingReader(a.workers, qd)
            cold[f"Tier 2: ring reader, read_segments_into, queue_depth={qd}"] = \
                lambda r_qd=r_qd: r_qd.read_segments_into(path, offs, lens, out)

    rows, method, skipped = [], None, []
    print(f"file {size >> 20} MiB in {len(offs)} segments, queue_depth {a.queue_depth}, repeats {a.repeats}")
    for name, fn in cold.items():
        walls, res_before, res_after = [], [], []
        for r in range(a.repeats):
            method = write_cold(path, size, seed=r)
            before = resident_fraction(path)
            if before > 0.01:
                skipped.append(dict(tier=name, repeat=r, resident_before=before))
                continue
            walls.append(time_read(fn))
            res_before.append(before)
            res_after.append(resident_fraction(path))
        if walls:
            row = summarize(name, walls, size, dict(cache="cold", resident_before_max=max(res_before),
                                                    resident_after_min=min(res_after)))
            rows.append(row)
            print(f"{name:<55} {row['gb_per_s']:7.2f} GB/s  {row['latency_ms']:9.1f} ms  (n={row['n']})")

    # Warm: make the whole file resident, then read it again through the reader.
    write_cold(path, size, seed=0)
    with open(path, "rb") as f:
        while f.read(64 << 20):
            pass
    warm_res = resident_fraction(path)
    warm_fn = (lambda: reader.read_segments_into(path, offs, lens, out)) if hasattr(reader, "read_segments_into") \
        else (lambda: reader.read_segments_parallel(path, offs, lens))
    if warm_res > 0.99:
        row = summarize("Tier 5: ring reader, file in page cache (warm)",
                        [time_read(warm_fn) for _ in range(a.repeats)], size,
                        dict(cache="warm", resident_before_min=warm_res))
        rows.append(row)
        print(f"{row['tier']:<55} {row['gb_per_s']:7.2f} GB/s  {row['latency_ms']:9.1f} ms  (n={row['n']})")
    else:
        skipped.append(dict(tier="Tier 5 warm", resident_before=warm_res))
    os.remove(path)

    rows += [
        dict(tier="Tier 3: .tgate4 compressed container", measured=False,
             reason="no loader reads a .tgate4 file end to end; the old figure was Tier 2 time x 0.25, not a measurement"),
        dict(tier="Tier 4: GPUDirect Storage + layer pipelining", measured=False,
             reason="needs CUDA with cuFile/GDS; the old figure was a hardcoded 45.2 ms"),
    ]
    import torch
    result = dict(
        script="scripts/benchmark_ingest_ladder_all_tiers.py", label=a.label,
        machine=dict(system=platform.system(), machine=platform.machine(), python=platform.python_version(),
                     torch=torch.__version__, cuda=torch.cuda.is_available()),
        args=dict(file_bytes=size, segments=len(offs), queue_depth=a.queue_depth, workers=a.workers,
                  repeats=a.repeats, cold_method=method, layer_segment_bytes=LAYER),
        rows=rows, skipped=skipped,
        note="GB/s is bytes / wall time with 1 GB = 1e9 bytes; se is the standard error over n runs.",
    )
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
