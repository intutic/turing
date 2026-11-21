import pytest
import torch
import numpy as np


def test_hierarchical_hca_chunk_pool_csrc():
    try:
        from turing import turing_csrc
    except ImportError:
        pytest.skip("turing_csrc not compiled")

    inp = np.ones((256, 4, 32), dtype=np.float32) * 3.5
    out = turing_csrc.hca_chunk_pool(inp, chunk_size=128)

    assert out.shape == (2, 4, 32)
    assert np.allclose(out, 3.5)


def test_fused_rope_transform_csrc():
    try:
        from turing import turing_csrc
    except ImportError:
        pytest.skip("turing_csrc not compiled")

    # [SeqLen, NumHeads, HeadDim]
    x = np.random.randn(8, 2, 32).astype(np.float32)
    # Forward rotation
    x_rot = turing_csrc.fused_rope_transform(x, 500000.0, 0, False)
    assert x_rot.shape == (8, 2, 32)
    assert not np.allclose(x, x_rot)

    # Inverse rotation -> should reconstruct original
    x_rec = turing_csrc.fused_rope_transform(x_rot, 500000.0, 0, True)
    assert np.allclose(x, x_rec, atol=1e-5)


def test_hierarchical_bitmap_allocator_csrc():
    try:
        from turing import turing_csrc
    except ImportError:
        pytest.skip("turing_csrc not compiled")

    alloc = turing_csrc.HierarchicalBitmapAllocator(4, 8, 16)
    assert alloc.get_num_free(512) == 4
    assert alloc.get_num_free(64) == 8
    assert alloc.get_num_free(16) == 16

    # 1040 tokens: 2 Huge (1024) + 1 Small (16)
    entries = alloc.allocate_prompt(1040)
    assert len(entries) == 3
    assert entries[0][0] == 512 and entries[0][2] == 512
    assert entries[1][0] == 512 and entries[1][2] == 512
    assert entries[2][0] == 16 and entries[2][2] == 16

    assert alloc.get_num_free(512) == 2
    assert alloc.get_num_free(16) == 15

    # Free the small block
    alloc.free_block(16, entries[2][1])
    assert alloc.get_num_free(16) == 16

def test_shannon_entropy_csrc():
    try:
        from turing import turing_csrc
    except ImportError:
        pytest.skip("turing_csrc not compiled")

    # Sharp distribution -> low entropy
    sharp = np.array([[100.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    ent_sharp = turing_csrc.compute_shannon_entropy(sharp)
    assert np.allclose(ent_sharp, 0.0, atol=1e-3)

    # Uniform distribution -> entropy = ln(4) ~= 1.38629
    uniform = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    ent_uniform = turing_csrc.compute_shannon_entropy(uniform)
    assert np.allclose(ent_uniform, np.log(4.0), atol=1e-3)
