"""
Unit & Parity Tests for Turing Engine 6-Tier Storage & Cold Ingestion Engine.
Verifies NativeAsyncRingReader, NativeGDSLoader, TuringIngestEngine, and PipelinedLayerLoader.
"""

import os
import tempfile
import pytest
import torch
import numpy as np

from turing.models.ingest import TuringIngestEngine, StorageTier, IngestionResult
from turing.experimental.kernels.gds_loader import PipelinedLayerLoader
from turing.turing_csrc import NativeAsyncRingReader, NativeGDSLoader


def test_native_async_ring_reader_exact_read(tmp_path):
    """Verifies C++ NativeAsyncRingReader byte-exact file reading."""
    test_file = tmp_path / "test_exact.bin"
    payload = b"TURING_ENGINE_STORAGE_TIER_2_IOURING_RING_TEST_PAYLOAD" * 100
    test_file.write_bytes(payload)

    reader = NativeAsyncRingReader(num_workers=4, queue_depth=32)
    assert reader.num_workers == 4
    assert reader.queue_depth == 32

    # Read slice
    offset = 14
    length = 64
    read_bytes = reader.read_exact(str(test_file), offset, length)
    assert read_bytes == payload[offset : offset + length]


def test_native_async_ring_reader_parallel_segments(tmp_path):
    """Verifies reading multiple discontinuous segments in parallel worker threads."""
    test_file = tmp_path / "test_segments.bin"
    chunk_a = b"AAAA" * 64
    chunk_b = b"BBBB" * 64
    chunk_c = b"CCCC" * 64
    payload = chunk_a + chunk_b + chunk_c
    test_file.write_bytes(payload)

    reader = NativeAsyncRingReader(num_workers=4, queue_depth=32)
    file_offsets = [0, len(chunk_a), len(chunk_a) + len(chunk_b)]
    byte_lengths = [len(chunk_a), len(chunk_b), len(chunk_c)]

    extracted_bytes = reader.read_segments_parallel(str(test_file), file_offsets, byte_lengths)
    assert extracted_bytes == payload


def test_native_async_ring_reader_splits_large_segments(tmp_path):
    """Segments larger than one 1 MiB request, at odd offsets and lengths, come back byte-exact."""
    payload = np.random.default_rng(0).integers(0, 256, (5 << 20) + 12345, dtype=np.uint8).tobytes()
    test_file = tmp_path / "large.bin"
    test_file.write_bytes(payload)
    offsets = [7, (3 << 20) + 1, 100]
    lengths = [(3 << 20) - 6, (2 << 20) + 12000, 1]

    for qd in (1, 3, 64):
        reader = NativeAsyncRingReader(num_workers=8, queue_depth=qd)
        got = reader.read_segments_parallel(str(test_file), offsets, lengths)
        assert got == b"".join(payload[o:o + n] for o, n in zip(offsets, lengths))

        out = np.zeros(sum(lengths) + 5, dtype=np.uint8)
        assert reader.read_segments_into(str(test_file), offsets, lengths, out) == sum(lengths)
        assert out[:sum(lengths)].tobytes() == got and not out[sum(lengths):].any()


def test_native_async_ring_reader_errors(tmp_path):
    test_file = tmp_path / "small.bin"
    test_file.write_bytes(b"x" * 100)
    reader = NativeAsyncRingReader()
    with pytest.raises(RuntimeError):  # past EOF
        reader.read_segments_parallel(str(test_file), [50], [100])
    with pytest.raises(ValueError):  # destination too small
        reader.read_segments_into(str(test_file), [0], [100], np.zeros(10, dtype=np.uint8))


def test_native_gds_loader_initialization():
    """Verifies NativeGDSLoader initializes cleanly without crashes."""
    loader = NativeGDSLoader()
    assert isinstance(loader.is_available(), bool)
    status = loader.get_status_info()
    assert isinstance(status, str)
    assert len(status) > 0


def test_turing_ingest_engine_auto_detection():
    """Verifies TuringIngestEngine selects appropriate storage tiers across targets."""
    engine = TuringIngestEngine()

    tier_mps = engine.detect_optimal_tier("model.safetensors", device="mps")
    assert tier_mps in [StorageTier.TIER5_WARM_UNIFIED, StorageTier.TIER2_IOURING_RING, StorageTier.TIER1_MADVISE_WILLNEED]

    tier_tgate = engine.detect_optimal_tier("model.tgate4", device="cpu")
    assert tier_tgate == StorageTier.TIER3_SUBSPACE_COMPRESSED

    tier_cpu = engine.detect_optimal_tier("model.safetensors", device="cpu")
    assert tier_cpu in [StorageTier.TIER2_IOURING_RING, StorageTier.TIER1_MADVISE_WILLNEED]


def test_turing_ingest_engine_raw_binary_load(tmp_path):
    """Verifies loading raw weight buffers across tiers."""
    test_file = tmp_path / "raw_weights.bin"
    raw_data = bytes([i % 256 for i in range(1024 * 1024)]) # 1 MB
    test_file.write_bytes(raw_data)

    engine = TuringIngestEngine()
    tensors, res = engine.load_tensors(str(test_file), device="cpu", tier=StorageTier.TIER1_MADVISE_WILLNEED)

    assert "raw_weights" in tensors
    assert tensors["raw_weights"].shape == (1024 * 1024,)
    assert tensors["raw_weights"].dtype == torch.uint8
    assert res.bytes_loaded == 1024 * 1024
    assert res.throughput_gb_s > 0.0


def test_pipelined_layer_loader_cpu_fallback():
    """Verifies PipelinedLayerLoader handles CPU gracefully without errors."""
    loader = PipelinedLayerLoader(device="cpu")
    assert not loader.is_cuda
    loader.stage_layer_async(0, lambda: {"w": torch.randn(10, 10)})
    loader.wait_for_layer(0)
    loader.synchronize_all()
