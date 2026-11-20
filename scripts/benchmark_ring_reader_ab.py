#!/usr/bin/env python3
"""
Cold-read A/B of NativeAsyncRingReader.read_segments_parallel: a baseline build against this checkout's build.

Two builds of turing_csrc cannot load in one process, so each timed read runs in a child process whose PYTHONPATH
selects the build, and baseline and current runs alternate (ABAB...) so machine load hits both alike. Before
every read the file is rewritten cold and checked with mincore, as in benchmark_ingest_ladder_all_tiers.py.

Build the baseline first, e.g.:
  git archive f936008 | tar -x -C /tmp/base && (cd /tmp/base && python setup.py build_ext --inplace)
python scripts/benchmark_ring_reader_ab.py --baseline /tmp/base [--pairs 8] [--size-mib 2048]
"""
import _bootstrap  # noqa: F401,E402  (this repo's root first on sys.path)
import argparse
import json
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_ingest_ladder_all_tiers import resident_fraction, segments, write_cold  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHILD = """
import sys, time
from turing.turing_csrc import NativeAsyncRingReader
path, offs, lens, qd = sys.argv[1], eval(sys.argv[2]), eval(sys.argv[3]), int(sys.argv[4])
r = NativeAsyncRingReader(8, qd)
t = time.perf_counter(); r.read_segments_parallel(path, offs, lens); print(time.perf_counter() - t)
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, help="checkout whose turing/ holds the baseline build")
    ap.add_argument("--baseline-label", default="baseline")
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--size-mib", type=int, default=2048)
    ap.add_argument("--queue-depth", type=int, default=64)
    ap.add_argument("--out", default="benchmarks/results/results_ring_reader_ab.json")
    a = ap.parse_args(argv)

    size = a.size_mib << 20
    folder = os.path.join(REPO, "build", "ingest_ladder.noindex")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "ab.bin")
    offs, lens = segments(size)
    arms = {a.baseline_label: os.path.abspath(a.baseline), "current": REPO}
    walls = {k: [] for k in arms}
    load = [os.getloadavg()[0]]
    for i in range(a.pairs):
        order = list(arms) if i % 2 == 0 else list(arms)[::-1]
        for arm in order:
            method = write_cold(path, size, seed=i)
            res = resident_fraction(path)
            if res > 0.01:
                raise SystemExit(f"file not cold before the read ({res:.3f} resident)")
            env = dict(os.environ, PYTHONPATH=arms[arm])
            out = subprocess.run([sys.executable, "-W", "error", "-c", CHILD, path, repr(offs), repr(lens),
                                  str(a.queue_depth)], env=env, cwd="/", capture_output=True, text=True, check=True)
            walls[arm].append(float(out.stdout.split()[-1]))
            print(f"pair {i} {arm:<10} {size / walls[arm][-1] / 1e9:6.2f} GB/s", flush=True)
    os.remove(path)
    load.append(os.getloadavg()[0])

    rows = []
    for arm, w in walls.items():
        gbps = [size / x / 1e9 for x in w]
        rows.append(dict(arm=arm, pythonpath=arms[arm], n=len(w), gb_per_s=statistics.mean(gbps),
                         gb_per_s_se=statistics.stdev(gbps) / len(gbps) ** 0.5, gb_per_s_median=statistics.median(gbps),
                         latency_ms=statistics.mean(w) * 1e3))
        print(f"{arm:<10} {rows[-1]['gb_per_s']:6.2f} +- {rows[-1]['gb_per_s_se']:.2f} GB/s (n={len(w)})")
    result = dict(script="scripts/benchmark_ring_reader_ab.py", call="read_segments_parallel (returns bytes)",
                  args=dict(file_bytes=size, segments=len(offs), queue_depth=a.queue_depth, workers=8, pairs=a.pairs,
                            cold_method=method, load_average_start=load[0], load_average_end=load[1]),
                  rows=rows, note="GB/s = bytes / wall time of the read call, 1 GB = 1e9 bytes; se over n runs.")
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
