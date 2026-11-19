import gc
import os
import warnings
import pytest

# tests calibrate from scratch (no KV calibration kept in ~/.cache between runs); the cache's own test turns it on
os.environ.setdefault("TURING_CACHE_DIR", "off")

@pytest.fixture(autouse=True)
def clean_memory_and_warnings():
    # Filter known third-party background stream teardowns
    warnings.filterwarnings("ignore", category=ResourceWarning)
    yield
    gc.collect()


@pytest.fixture(scope="module")
def turing_server():
    """`turing serve` in-process on a free port: the real app and engine (test-tiny, synthetic weights, CPU).
    Yields the OpenAI base URL."""
    import socket
    import threading
    import time
    import uvicorn
    from turing.config import TuringConfig
    from turing.models.registry import get_model_config
    from turing.serving.engine import ContinuousBatchEngine
    from turing.serving.server import create_app

    engine = ContinuousBatchEngine(get_model_config("test-tiny"), TuringConfig(device="cpu"))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(engine), log_level="warning", ws="none", lifespan="on"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while not server.started:
        assert thread.is_alive() and time.time() < deadline, "server did not start"
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/v1"
    server.should_exit = True
    thread.join(timeout=30)
    sock.close()


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
