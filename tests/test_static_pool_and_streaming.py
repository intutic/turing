import pytest
import os
import torch
from turing.experimental.paging import StaticPagedKVPool
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

@pytest.mark.parametrize("arch", ["llama", "gemma2", "gpt2", "mixtral", "qwen2_moe", "phi3"])
def test_streaming_conversion_matches_in_memory_quantization(tmp_path, arch):
    """The layer-by-layer converter reads safetensors shards (as HF saves them) and gives the same packed model as
    quantizing the converted in-memory model."""
    import copy
    from tests.test_hf_parity import _tiny
    from turing.models.hf_loader import RealHuggingFaceLoader
    from turing.models.quant import load_quantized, quantize_model
    hf = _tiny(arch)
    ckpt = tmp_path / "ckpt"
    hf.save_pretrained(ckpt, max_shard_size="200KB")  # several shards
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained("gpt2").save_pretrained(ckpt)
    res = StreamingHFSubspaceLoader(str(ckpt), method="rtn").convert(str(tmp_path / "out"))
    assert res["layers_converted"] == hf.config.num_hidden_layers
    streamed, tok = load_quantized(str(tmp_path / "out"))
    ref = quantize_model(RealHuggingFaceLoader.convert_hf_model(hf), "rtn")
    ids = torch.randint(0, 128, (1, 10))
    torch.testing.assert_close(streamed(ids)[0], ref(ids)[0])
    assert tok is not None
