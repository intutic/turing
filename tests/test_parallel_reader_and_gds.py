"""
ParallelReader (io_uring on Linux builds with liburing, pread threads elsewhere) and NativeGDSLoader.read_to_device
(cuFile, else pinned host memory + cudaMemcpy, else an honest "no_cuda" that reads nothing).
"""

import sys

import numpy as np
import pytest
import torch

csrc = pytest.importorskip("turing.turing_csrc")


@pytest.fixture(scope="module")
def payload_file(tmp_path_factory):
    data = np.random.default_rng(1).integers(0, 256, (3 << 20) + 777, dtype=np.uint8).tobytes()
    path = tmp_path_factory.mktemp("reader") / "payload.bin"
    path.write_bytes(data)
    return str(path), data


def test_old_name_is_the_same_class():
    assert csrc.NativeAsyncRingReader is csrc.ParallelReader


def test_backend_is_reported_honestly():
    backend = csrc.ParallelReader().backend
    if sys.platform == "win32":
        assert backend == "win32"
    elif sys.platform.startswith("linux"):
        assert backend in ("io_uring", "pread")
    else:
        assert backend == "pread"  # no io_uring outside Linux


@pytest.mark.parametrize("queue_depth", [1, 2, 7, 64])
def test_many_segments_byte_exact(payload_file, queue_depth):
    path, data = payload_file
    rng = np.random.default_rng(queue_depth)
    offsets = sorted(int(x) for x in rng.integers(0, len(data) - 300000, 40))
    lengths = [int(x) for x in rng.integers(0, 300000, 40)]
    lengths[3] = (1 << 20) + 5  # spans two requests
    lengths[4] = 0
    reader = csrc.ParallelReader(num_workers=4, queue_depth=queue_depth)
    got = reader.read_segments_parallel(path, offsets, lengths)
    assert got == b"".join(data[o:o + n] for o, n in zip(offsets, lengths))
    assert reader.read_exact(path, 12345, 1 << 20) == data[12345:12345 + (1 << 20)]


def test_missing_file_raises(tmp_path):
    with pytest.raises(RuntimeError):
        csrc.ParallelReader().read_segments_parallel(str(tmp_path / "absent.bin"), [0], [10])


def test_read_to_device_without_cuda_reads_nothing(payload_file):
    loader = csrc.NativeGDSLoader()
    if loader.has_cuda():
        pytest.skip("a CUDA runtime is present")
    assert not loader.is_available()
    path, _ = payload_file
    host = np.zeros(64, dtype=np.uint8)  # a pointer that must not be written
    assert loader.read_to_device(path, 0, 64, host.ctypes.data) == (0, "no_cuda")
    assert not host.any()
    assert "No CUDA runtime" in loader.get_status_info()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("allow_cufile", [False, True])
def test_read_to_device_lands_the_bytes(payload_file, allow_cufile):
    loader = csrc.NativeGDSLoader()
    assert loader.has_cuda()
    path, data = payload_file
    offset, n = 4099, (9 << 20) // 4  # not a multiple of the 8 MiB staging chunk
    n = min(n, len(data) - offset)
    dev = torch.zeros(n, dtype=torch.uint8, device="cuda")
    got, route = loader.read_to_device(path, offset, n, dev.data_ptr(), allow_cufile)
    assert got == n
    assert route == "pinned_host_copy" if not allow_cufile else route in ("cufile", "pinned_host_copy")
    assert bytes(dev.cpu().numpy()) == data[offset:offset + n]
