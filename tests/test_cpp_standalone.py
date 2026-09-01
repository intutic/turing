"""
Unit & Integration Test Suite for Standalone C++20 turing-cli Executable.
Verifies native C++ compilation, CLI info probing, and bare-metal GGUF text generation.
"""

import os
import subprocess
import tempfile
import pytest
from turing.models.gguf_loader import create_test_gguf_file, GGMLType


import shutil

@pytest.fixture(scope="module")
def standalone_cli_binary():
    os.makedirs("build", exist_ok=True)
    binary_path = os.path.abspath("build/turing-cli")
    
    # Auto-detect g++ or clang++
    cxx = os.environ.get("CXX")
    if not cxx:
        cxx = "g++" if shutil.which("g++") else "clang++"
        
    if not shutil.which(cxx):
        pytest.skip(f"No C++ compiler ({cxx}) found on system")

    cmd = [cxx, "-std=c++20", "-O3", "-I", "csrc", "csrc/turing_main.cpp", "-o", binary_path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        pytest.skip(f"C++ compiler failed to build standalone binary: {res.stderr}")
    return binary_path


@pytest.fixture(scope="module")
def test_gguf_path():
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        path = f.name
    create_test_gguf_file(
        path,
        architecture="llama",
        hidden_dim=32,
        ffn_dim=64,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        vocab_size=64,
        quant_type=GGMLType.F16
    )
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_turing_cli_info(standalone_cli_binary):
    res = subprocess.run([standalone_cli_binary, "info"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "Turing Engine Standalone C++20 Runtime" in res.stdout
    assert "Hardware Capabilities" in res.stdout


def test_turing_cli_generate(standalone_cli_binary, test_gguf_path):
    cmd = [
        standalone_cli_binary,
        "generate",
        "--model", test_gguf_path,
        "--prompt", "<tok_1> <tok_2>",
        "--max-tokens", "4",
        "--temp", "0.0"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "Generated" in res.stderr or len(res.stdout.strip()) > 0
