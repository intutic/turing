"""KV page export/import between pools, and prefill/decode disaggregation across two processes over TCP: the decode
process continues from the prefill process's pages and first token with output identical to one engine."""
import asyncio
import multiprocessing as mp

import pytest
import torch

from turing.config import TuringConfig
from turing.serving.disagg_sidecar import PrefillWorker, add_remote_prefilled_request
from turing.serving.engine import ContinuousBatchEngine
from turing.serving.kv_transfer import export_kv, import_kv


def _model(kv_dtype):
    from tests.test_hf_parity import _tiny
    from turing.models.hf_loader import RealHuggingFaceLoader
    model = RealHuggingFaceLoader.convert_hf_model(_tiny("llama"), sparsity_ratio=0.0, device="cpu",
                                                   enable_recirculation=False)
    model.config.kv_cache_dtype, model.config.kv_pool_gib = kv_dtype, 0.01
    return model


def _engine(kv_dtype):
    model = _model(kv_dtype)
    return ContinuousBatchEngine(model.config, TuringConfig(device="cpu"), model=model, prefix_caching=False)


PROMPT = [int(t) for t in torch.randint(0, 128, (37,), generator=torch.Generator().manual_seed(3))]


@pytest.mark.parametrize("kv_dtype", ["native", "int8", "int4"])
@torch.no_grad()
def test_exported_pages_import_into_another_pool_bit_exact(kv_dtype):
    src, dst = _model(kv_dtype), _model(kv_dtype)
    _, cache = src(torch.tensor([PROMPT]))
    cache.reserve(cache.length + 40)  # extra reserved pages are not exported
    header, blob = export_kv(cache)
    dst.kv_pool().alloc_pages(3)  # the importing pool hands out different page ids
    got = import_kv(dst.kv_pool(), header, blob)
    assert got.length == cache.length == len(PROMPT) and len(got.pages) == 3 and got.pages != cache.pages[:3]
    for layer in range(src.config.num_layers):
        for a, b in zip(src.kv_pool().read(layer, cache, cache.length), dst.kv_pool().read(layer, got, got.length)):
            assert torch.equal(a, b)
    # continuing from the imported cache gives the source's next-token logits
    nxt = torch.tensor([[5]])
    torch.testing.assert_close(dst(nxt, past_key_values=got)[0], src(nxt, past_key_values=cache)[0], atol=0, rtol=0)


def test_import_rejects_mismatched_or_malformed_payloads():
    src, dst = _model("int8"), _model("native")
    _, cache = src(torch.tensor([PROMPT]))
    header, blob = export_kv(cache)
    with pytest.raises(ValueError, match="kv_cache_dtype"):
        import_kv(dst.kv_pool(), header, blob)
    free = src.kv_pool().num_free_pages
    for bad in (dict(header, tensors=header["tensors"][1:]),                        # a field missing
                dict(header, tensors=[[n, d, [s[0], s[1] + 1, *s[2:]]] for n, d, s in header["tensors"]]),
                dict(header, length=-1)):
        with pytest.raises(ValueError):
            import_kv(src.kv_pool(), bad, blob)
    with pytest.raises(ValueError):
        import_kv(src.kv_pool(), header, blob[:-1])                                    # truncated
    assert src.kv_pool().num_free_pages == free                                        # nothing leaked


def _prefill_server(kv_dtype, port_q):
    worker = PrefillWorker(_engine(kv_dtype))
    port_q.put(worker.port)
    worker.serve_forever()


def _generate(engine, prompt, n):
    async def go():
        await engine.start()
        try:
            return [t async for t in engine.stream_generate(prompt, max_new_tokens=n, temperature=0.0)]
        finally:
            await engine.stop()
    return asyncio.run(go())


@pytest.mark.parametrize("kv_dtype", ["native", "int8"])
def test_prefill_and_decode_in_two_processes_match_one_engine(kv_dtype):
    ctx = mp.get_context("spawn")
    port_q = ctx.Queue()
    proc = ctx.Process(target=_prefill_server, args=(kv_dtype, port_q), daemon=True)
    proc.start()
    try:
        addr = ("127.0.0.1", port_q.get(timeout=120))
        ref = _generate(_engine(kv_dtype), PROMPT, 12)

        decode = _engine(kv_dtype)

        async def go():
            await decode.start()
            try:
                outs = []
                for prompt in (PROMPT, PROMPT[:5]):  # two requests over the same worker
                    first, req = await add_remote_prefilled_request(decode, addr, prompt, max_new_tokens=12)
                    outs.append([first] + [t async for t in decode.stream_request(req)])
                return outs
            finally:
                await decode.stop()
        got, short = asyncio.run(go())
        assert got == ref and len(got) == 12
        assert short == _generate(_engine(kv_dtype), PROMPT[:5], 12)
        assert decode.kv_pool.used_pages == 0  # the imported pages were returned
    finally:
        proc.terminate()
        proc.join(timeout=30)


@pytest.mark.parametrize("kv_dtype", ["native", "int8"])
def test_engine_with_remote_prefill_streams_the_same_tokens(kv_dtype):
    """`turing serve --prefill-server`: the engine prefills on the worker, imports the pages, streams the worker's
    first token and then decodes; greedy output equals a single engine's, and no page leaks."""
    import threading
    worker = PrefillWorker(_engine(kv_dtype))
    threading.Thread(target=worker.serve_forever, daemon=True).start()
    ref = _generate(_engine(kv_dtype), PROMPT, 8)
    decode = _engine(kv_dtype)
    decode.remote_prefill = ("127.0.0.1", worker.port)
    assert _generate(decode, PROMPT, 8) == ref
    assert decode.kv_pool.used_pages == 0
