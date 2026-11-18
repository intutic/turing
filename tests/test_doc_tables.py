"""Every numeric table in the docs equals what scripts/make_doc_tables.py generates from the committed results."""
import subprocess
import sys
from pathlib import Path


def test_doc_tables_match_the_results_files():
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "scripts/make_doc_tables.py", "--check"], cwd=root, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
