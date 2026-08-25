import pytest
import os
import torch
from turing.core.paging import StaticPagedKVPool
from turing.models.streaming_loader import StreamingHFSubspaceLoader

def test_static_paged_kv_pool_allocation_and_free():
    pool = StaticPagedKVPool(
        num_layers=2,
        num_heads=4,
        head_dim=32,
        page_size=16,
        max_total_pages=16,
        device=torch.device("cpu"),
        dtype=torch.float32
    )

    stats0 = pool.get_stats()
    assert stats0["total_pages"] == 16
    assert stats0["free_pages"] == 16
    assert stats0["used_pages"] == 0

    # Allocate 4 pages for req-1
    pages1 = pool.allocate_pages("req-1", 4)
    assert len(pages1) == 4
    stats1 = pool.get_stats()
    assert stats1["used_pages"] == 4
    assert stats1["free_pages"] == 12

    # Allocate 8 pages for req-2
    pages2 = pool.allocate_pages("req-2", 8)
    assert len(pages2) == 8
    stats2 = pool.get_stats()
    assert stats2["used_pages"] == 12
    assert stats2["free_pages"] == 4

    # Free req-1 -> pages recycled
    pool.free_pages("req-1")
    stats3 = pool.get_stats()
    assert stats3["used_pages"] == 8
    assert stats3["free_pages"] == 8

    # Free req-2 -> all pages free
    pool.free_pages("req-2")
    stats4 = pool.get_stats()
    assert stats4["used_pages"] == 0
    assert stats4["free_pages"] == 16

def test_streaming_hf_subspace_loader():
    loader = StreamingHFSubspaceLoader(
        repo_id="test-org/test-model",
        sparsity_ratio=0.50
    )
    
    out_dir = "tests/scratch_weights"
    res = loader.convert_model_streaming(total_layers=2, output_dir=out_dir)
    assert res["layers_converted"] == 2
    assert len(res["files"]) == 2
    for f in res["files"]:
        assert os.path.exists(f)
        os.remove(f)
    if os.path.exists(out_dir):
        os.rmdir(out_dir)

