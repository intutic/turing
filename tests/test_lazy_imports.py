"""`import turing`, `turing.core` and `turing.experimental` load no submodule (and so not torch); names still resolve on
use."""

import subprocess
import sys

CHECK = """
import sys, turing, turing.core, turing.experimental
loaded = sorted(m for m in sys.modules if m.startswith("turing.") or m == "torch")
assert loaded == ["turing.core", "turing.experimental"], loaded
from turing import SubspaceCausalLM, ModelConfig, chain
from turing.core import GPULRUExpertCache, HostExpertBank
from turing.experimental import TensorSerializer
assert "turing.experimental.paging" not in sys.modules and "turing.experimental.speculation" not in sys.modules
try:
    turing.core.TensorSerializer  # moved to turing.experimental
except AttributeError:
    pass
else:
    raise AssertionError("experimental name still exported from turing.core")
"""


def test_package_imports_are_lazy():
    # no -W error here: sentencepiece segfaults at interpreter exit under it (any interpreter, not this package)
    subprocess.run([sys.executable, "-c", CHECK], check=True)
