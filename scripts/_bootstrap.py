"""Put the repo this script lives in first on sys.path.

`import _bootstrap` at the top of a script makes `import turing` load this checkout's package,
not the one an editable install (or another git worktree) put on the path.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.path[:1] != [ROOT]:
    sys.path.insert(0, ROOT)
