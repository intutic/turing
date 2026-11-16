"""Tensor and pipeline parallelism across real processes (gloo on CPU): every rank builds the same HF-converted tiny
model, the driver shards it, rank 0 drives prefill, chunked prefill, decode, batched decode and pipelined
micro-batches, and the logits must equal a single-process model's at atol 1e-4."""
import os
import socket

import pytest
import torch
import torch.multiprocessing as tmp

from turing.serving.distributed import DistributedConfig, DistributedInferenceDriver, PlacementPolicy
from turing.core.kv_cache import KVCache
from turing.models.tensor_parallel import stage_layer_range

ARCHS = ["llama", "qwen2_qkv_bias", "gemma2", "mixtral", "qwen2_moe"]  # GQA 4:2; biases; softcap + window; MoE;
# MoE with a shared expert and dense layers
ATOL = 1e-4


def _model(arch):
    from tests.test_hf_parity import _tiny
    from turing.models.hf_loader import RealHuggingFaceLoader
    model = RealHuggingFaceLoader.convert_hf_model(_tiny(arch), sparsity_ratio=0.0, device="cpu",
                                                   enable_recirculation=False)
    model.config.kv_pool_gib = 0.01
    return model


def _close(got, ref, what):
    torch.testing.assert_close(got, ref, atol=ATOL, rtol=0, msg=lambda m: f"{what}: {m}")


def _drive(arch, driver, ref):
    """Rank 0: the same calls through the driver and through the unsharded reference."""
    ids = torch.randint(0, 128, (2, 24), generator=torch.Generator().manual_seed(1)).to(ref.lm_head.weight.device)
    fwd = driver.forward_distributed
    # one request: prefill 8, chunked prefill 4, then single tokens (crossing Gemma 2's window of 6)
    got, c = fwd(ids[:1, :8])
    want, rc = ref(ids[:1, :8])
    _close(got, want, f"{arch} prefill")
    got, c = fwd(ids[:1, 8:12], past_key_values=c)
    _close(got, ref(ids[:1, 8:12], past_key_values=rc)[0], f"{arch} chunked prefill")
    for t in range(12, 18):
        got, c = fwd(ids[:1, t:t + 1], past_key_values=c)
        _close(got, ref(ids[:1, t:t + 1], past_key_values=rc)[0], f"{arch} decode {t}")
    # a second request, then batched decode steps over both (list of caches)
    got, c2 = fwd(ids[1:, :5], logits_to_keep=1)
    want, rc2 = ref(ids[1:, :5], logits_to_keep=1)
    _close(got, want, f"{arch} second prefill, last logits")
    for t in range(3):
        step = torch.tensor([[ids[0, 18 + t]], [ids[1, 5 + t]]])
        got, _ = fwd(step, past_key_values=[c, c2])
        _close(got, ref(step, past_key_values=[rc, rc2])[0], f"{arch} batched decode {t}")
    c.free()  # the other ranks free their shards of it on the next step
    got, c2 = fwd(ids[1:, 8:9], past_key_values=c2)
    _close(got, ref(ids[1:, 8:9], past_key_values=rc2)[0], f"{arch} decode after a free")
    # three prompts of different lengths as pipelined micro-batches
    prompts = [ids[:1, :7], ids[1:, :11], ids[:1, 3:6]]
    caches = [KVCache(driver.model.kv_pool()) for _ in prompts]
    outs = driver.forward_many(list(zip(prompts, caches)))
    for i, (p, out) in enumerate(zip(prompts, outs)):
        _close(out, ref(p)[0], f"{arch} micro-batch {i}")


def _engine_generate(model, dist_driver=None):
    """Greedy tokens for three concurrent requests from a ContinuousBatchEngine (rank 0's engine when distributed)."""
    import asyncio
    from turing.config import TuringConfig
    from turing.serving.engine import ContinuousBatchEngine
    engine = ContinuousBatchEngine(model.config, TuringConfig(device="cpu"), model=model, dist_driver=dist_driver,
                                   prefix_caching=False)
    prompts = [[int(t) for t in torch.randint(0, 128, (n,), generator=torch.Generator().manual_seed(n))]
               for n in (9, 20, 4)]

    async def go():
        await engine.start()
        try:
            reqs = [await engine.add_request(p, max_new_tokens=10, temperature=0.0) for p in prompts]
            return [[t async for t in engine.stream_request(r)] for r in reqs]
        finally:
            await engine.stop()
    return asyncio.run(go())


def _worker(rank, world, tp, pp, port, archs, backend, mode="logits"):
    torch.set_num_threads(1)
    dc = DistributedConfig(tp_size=tp, pp_size=pp, master_addr="127.0.0.1", master_port=port, backend=backend,
                           rank=rank, world_size=world, local_rank=rank)
    for arch in archs:
        driver = DistributedInferenceDriver(_model(arch), dist_config=dc)
        pool = driver.model.kv_pool()  # each rank's pool holds only its KV heads and its stage's layers
        assert (pool.num_kv_heads, pool.num_layers) == (max(1, 2 // tp), 4 // pp), (pool.num_kv_heads, pool.num_layers)
        if rank:
            driver.worker_loop()
            if mode == "logits":  # 5 requests reached the workers (ids 0-4); request 0's free reached them too
                assert sorted(driver._caches) == [1, 2, 3, 4], sorted(driver._caches)
            continue
        if mode == "engine":
            got = _engine_generate(driver.model, driver)
            assert got == _engine_generate(_model(arch)), arch
        else:
            with torch.inference_mode():
                _drive(arch, driver, _model(arch).to(dc.device))
        driver.shutdown()
    torch.distributed.destroy_process_group()
    os._exit(0)  # skip interpreter teardown (importing transformers under -W error can crash it at exit)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _launch(tp, pp, archs=ARCHS, backend="gloo", mode="logits"):
    # spawn (gloo's init_process_group crashes in a process forked after torch was imported, on macOS)
    tmp.start_processes(_worker, args=(tp * pp, tp, pp, _free_port(), archs, backend, mode), nprocs=tp * pp, join=True,
                        start_method="spawn")


@pytest.mark.parametrize("tp,pp", [(2, 1), (4, 1), (1, 2), (2, 2)], ids=["tp2", "tp4", "pp2", "tp2xpp2"])
def test_parallel_logits_match_single_process(tp, pp):
    # TP=4 on 4 query / 2 KV heads: each rank keeps a copy of its KV head; TP=2xPP=2 runs one architecture
    _launch(tp, pp, ARCHS if tp * pp < 4 or pp == 1 else ["qwen2_qkv_bias"])


def test_engine_on_rank0_drives_tp2_x_pp2_workers():
    """The serving layout: rank 0 runs ContinuousBatchEngine with the driver, ranks 1-3 run worker_loop(); greedy
    output equals a single-process engine's."""
    _launch(2, 2, ["llama"], mode="engine")


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="needs 2 CUDA GPUs (NCCL)")
def test_tensor_parallel_nccl_two_gpus():
    _launch(2, 1, ["llama", "mixtral"], backend="nccl")


def test_stage_layer_ranges_cover_every_layer_once():
    assert [stage_layer_range(10, s, 4) for s in range(4)] == [(0, 3), (3, 6), (6, 8), (8, 10)]
    with pytest.raises(ValueError):
        stage_layer_range(2, 0, 3)


def test_placement_policy():
    assert PlacementPolicy.auto_decompose(1) == (1, 1)
    assert PlacementPolicy.auto_decompose(4, model_params_billion=70.0) == (2, 2)
    assert PlacementPolicy.auto_decompose(8, model_params_billion=70.0) == (4, 2)
