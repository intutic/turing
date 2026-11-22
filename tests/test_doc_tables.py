"""Every numeric table in the docs equals what scripts/make_doc_tables.py generates from the committed results."""
import subprocess
import sys
from pathlib import Path


def test_doc_tables_match_the_results_files():
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "scripts/make_doc_tables.py", "--check"], cwd=root, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_device_time_counts_side_by_side_kernels_once_and_the_gaps_between():
    """scripts/profile_device_time.py: busy is the union of the kernels' intervals, idle the rest of the span."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from profile_device_time import device_time
    # times in us: a and b side by side on two streams for 5 ms, then nothing for 25 ms, then c
    t = device_time([(0, 10e3, 1, "a"), (5e3, 10e3, 2, "b"), (40e3, 5e3, 1, "c")])
    assert (t["busy_ms"], t["idle_ms"], t["span_ms"], t["summed_ms"]) == (20.0, 25.0, 45.0, 25.0)
    assert t["streams"] == 2 and t["launches"] == 3
    assert t["top_gaps"] == [[25.0, 1, "b", "c"]] and t["top_kernels"][0][:2] == [10.0, 1]
