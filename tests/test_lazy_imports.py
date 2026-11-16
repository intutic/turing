"""`import turing` and `import turing.core` load no submodule (and so not torch); names still resolve on use."""

import subprocess
import sys

CHECK = """
import sys, turing, turing.core
loaded = sorted(m for m in sys.modules if m.startswith("turing.") or m == "torch")
assert loaded == ["turing.core"], loaded
from turing import SubspaceCausalLM, ModelConfig, chain
from turing.core import GPULRUExpertCache, HostExpertBank, TensorSerializer
assert "turing.core.paging" not in sys.modules and "turing.core.speculation" not in sys.modules
try:
    turing.core.SpectralRadixSVDForest
except AttributeError:
    pass
else:
    raise AssertionError("deleted name still exported")
"""


def test_package_imports_are_lazy():
    # no -W error here: sentencepiece segfaults at interpreter exit under it (any interpreter, not this package)
    subprocess.run([sys.executable, "-c", CHECK], check=True)
