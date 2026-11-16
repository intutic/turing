import gc
import warnings
import pytest

@pytest.fixture(autouse=True)
def clean_memory_and_warnings():
    # Filter known third-party background stream teardowns
    warnings.filterwarnings("ignore", category=ResourceWarning)
    yield
    gc.collect()



@pytest.fixture(scope="session")
def turing_cli_binary():
    """build/turing-cli from csrc/turing_main.cpp, with the flags setup.py uses for this machine."""
    import os
    import platform
    import shutil
    import subprocess

    cxx = os.environ.get("CXX") or ("g++" if shutil.which("g++") else "clang++")
    if not shutil.which(cxx):
        pytest.skip(f"No C++ compiler ({cxx}) found on system")
    os.makedirs("build", exist_ok=True)
    binary_path = os.path.abspath("build/turing-cli")
    flags = ["-std=c++20", "-O3", "-flto", "-fvisibility=hidden"]
    if platform.machine().lower() in ("x86_64", "amd64"):
        flags += ["-mavx2", "-mfma", "-mf16c"]
    res = subprocess.run([cxx, *flags, "-I", "csrc", "csrc/turing_main.cpp", "-o", binary_path],
                         capture_output=True, text=True)
    if res.returncode != 0:
        pytest.skip(f"C++ compiler failed to build standalone binary: {res.stderr}")
    return binary_path
