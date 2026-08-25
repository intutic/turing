import pytest
import torch
import numpy as np
from turing.core.subspace import SubspaceManager

def test_subspace_quantization_math():
    x = torch.randn(4, 64)
    mgr = SubspaceManager(hidden_dim=64, rank=16)
    q_int8, scale = mgr.quantize_subspace_int8(x)

    assert q_int8.dtype == torch.int8
    assert scale.shape == (4, 1)

    recon = mgr.dequantize_subspace_int8(q_int8, scale)
    # Cosine similarity should be > 0.99
    cos_sim = torch.cosine_similarity(x.view(-1), recon.view(-1), dim=0)
    assert cos_sim.item() > 0.98

def test_bitmask_encoding_decoding():
    active_tiles = [0, 3, 5, 7, 11]
    total_tiles = 16
    mask_int, mask_bytes = SubspaceManager.encode_bitmask(active_tiles, total_tiles)

    assert mask_int == (1 << 0) | (1 << 3) | (1 << 5) | (1 << 7) | (1 << 11)
    assert len(mask_bytes) == 16

    decoded = SubspaceManager.decode_bitmask(mask_bytes, total_tiles)
    assert decoded == active_tiles

def test_native_csrc_extension():
    try:
        from turing import turing_csrc
    except ImportError:
        pytest.skip("turing_csrc not compiled")

    in_arr = np.random.randn(64).astype(np.float32)
    content_out = np.zeros(16, dtype=np.int8)
    fluency_out = np.zeros(24, dtype=np.uint8)

    turing_csrc.subspace_quantize(
        in_arr, content_out, fluency_out,
        scale_content=1.0, scale_fluency=1.0,
        content_dim=16, fluency_dim=48
    )

    assert len(content_out) == 16
    assert len(fluency_out) == 24

    # Test Paged Attention Engine C++ binding
    engine = turing_csrc.TuringPagedAttentionEngine(num_heads=2, head_dim=64, block_size=16, num_blocks=4)
    query = np.random.randn(2, 64).astype(np.float32)
    block_table = np.array([0, 1], dtype=np.int32)
    out = engine.forward_selective_attention(query, block_table, active_page_mask=0x3)
    assert out.shape == (2, 64)

def test_birkhoff_manifold_csrc():
    try:
        from turing import turing_csrc
    except ImportError:
        pytest.skip("turing_csrc not compiled")

    mat = np.random.randn(4, 4).astype(np.float32)
    proj = turing_csrc.birkhoff_project(mat, num_iterations=25, eps=1e-6)

    assert proj.shape == (4, 4)
    assert np.all(proj >= 0)
    assert np.allclose(proj.sum(axis=-1), 1.0, atol=1e-3)
    assert np.allclose(proj.sum(axis=-2), 1.0, atol=1e-3)

def test_dag_tree_mask_csrc():
    try:
        from turing import turing_csrc
    except ImportError:
        pytest.skip("turing_csrc not compiled")

    # Tree: 0 (root) -> 1 -> 2, 0 -> 3
    parents = np.array([-1, 0, 1, 0], dtype=np.int32)
    mask = turing_csrc.build_dag_tree_mask(parents)

    assert mask.shape == (4, 4)
    assert mask[0, 0] == 0.0
    assert mask[0, 1] == -np.inf
    assert mask[2, 0] == 0.0
    assert mask[2, 1] == 0.0
    assert mask[2, 2] == 0.0
    assert mask[2, 3] == -np.inf
    assert mask[3, 0] == 0.0
    assert mask[3, 3] == 0.0

def test_sinkhorn_ot_eviction_csrc():
    try:
        from turing import turing_csrc
    except ImportError:
        pytest.skip("turing_csrc not compiled")

    q = np.random.randn(8, 64).astype(np.float32)
    k = np.random.randn(32, 64).astype(np.float32)
    budget = 10

    retained, mass = turing_csrc.sinkhorn_ot_eviction(q, k, budget, epsilon=0.05, num_iters=15)

    assert len(retained) == budget
    assert len(mass) == 32
    assert np.all(retained >= 0) and np.all(retained < 32)
    assert np.all(np.diff(retained) > 0) # Sorted indices

def test_hierarchical_hca_chunk_pool_csrc():
    try:
        from turing import turing_csrc
    except ImportError:
        pytest.skip("turing_csrc not compiled")

    inp = np.ones((256, 4, 32), dtype=np.float32) * 3.5
    out = turing_csrc.hca_chunk_pool(inp, chunk_size=128)

    assert out.shape == (2, 4, 32)
    assert np.allclose(out, 3.5)

def test_tensor_serializer_csrc():
    try:
        from turing import turing_csrc
    except ImportError:
        pytest.skip("turing_csrc not compiled")

    tensor = np.random.randn(2, 8, 16).astype(np.float32)
    ser_bytes = turing_csrc.serialize_tensor_int8(tensor)
    assert isinstance(ser_bytes, bytes)
    assert len(ser_bytes) > 2 * 8 * 16

    deser = turing_csrc.deserialize_tensor_int8(ser_bytes[4:])
    assert deser.shape == (2, 8, 16)
    # Relative error should be low for INT8 quantization
    assert np.corrcoef(tensor.flatten(), deser.flatten())[0, 1] > 0.98

def test_radix_trie_csrc():
    try:
        from turing import turing_csrc
    except ImportError:
        pytest.skip("turing_csrc not compiled")

    trie = turing_csrc.RadixTrieIndex()
    id1 = trie.insert([101, 202, 303, 404])
    id2 = trie.insert([101, 202, 505, 606])

    assert id1 > 0 and id2 > 0

    node_id, match_len = trie.match_longest_prefix([101, 202, 303, 999])
    assert match_len == 3
    assert node_id > 0

def test_apc_hash_mask_csrc():
    try:
        from turing import turing_csrc
    except ImportError:
        pytest.skip("turing_csrc not compiled")

    mask1 = np.array([1, 0, 1, 1, 0, 1, 0, 0], dtype=np.uint8)
    mask2 = np.array([1, 0, 1, 1, 0, 1, 0, 0], dtype=np.uint8)
    mask3 = np.array([0, 0, 1, 1, 0, 1, 0, 0], dtype=np.uint8)

    h1 = turing_csrc.apc_hash_mask(mask1)
    h2 = turing_csrc.apc_hash_mask(mask2)
    h3 = turing_csrc.apc_hash_mask(mask3)

    assert h1 == h2
    assert h1 != h3

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

def test_lru_expert_cache_fast_csrc():
    try:
        from turing import turing_csrc
    except ImportError:
        pytest.skip("turing_csrc not compiled")

    cache = turing_csrc.LRUExpertCacheFast(4)
    assert not cache.contains(0, 1)

    # Allocate 4 slots
    s1, e_l1, e_e1 = cache.allocate_or_evict_slot(0, 1)
    s2, e_l2, e_e2 = cache.allocate_or_evict_slot(0, 2)
    s3, e_l3, e_e3 = cache.allocate_or_evict_slot(1, 1)
    s4, e_l4, e_e4 = cache.allocate_or_evict_slot(1, 2)

    assert cache.contains(0, 1) and cache.contains(1, 2)
    assert e_l1 == -1 and e_l4 == -1

    # Access (0, 1) -> moves to MRU
    slot = cache.get_slot(0, 1)
    assert slot == s1
    assert cache.hits == 1

    # Miss access
    miss = cache.get_slot(2, 5)
    assert miss == -1
    assert cache.misses == 1

    # Now allocate a 5th expert -> should evict LRU (which is (0, 2))
    s5, ev_l, ev_e = cache.allocate_or_evict_slot(2, 3)
    assert ev_l == 0 and ev_e == 2
    assert s5 == s2

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


