"""The libllama CPU backend (turing.turing_llama, csrc/cpu/turing_llama.hpp) against Hugging Face's models: the same
tiny random models and GGUF writer as tests/test_gguf_parity.py. Skips unless built with TURING_BUILD_LLAMA=1."""
import pytest
import torch

pytest.importorskip("gguf")
turing_llama = pytest.importorskip("turing.turing_llama", reason="libllama backend not built (TURING_BUILD_LLAMA=1)")

from tests.test_gguf_parity import DECODE, PROMPT, _hf_logits, _hf_model, _write_gguf  # noqa: E402

# dense, grouped-query, q/k-norm, and two MoE layouts (libllama's llama arch reads separate gate / up expert stacks only)
KINDS = ["llama", "qwen2", "qwen3", "qwen3_moe", "mixtral"]


def _llama(path, **kw):
    return turing_llama.Llama(path, n_ctx=64, n_threads=2, exact=True, **kw)


@pytest.mark.parametrize("kind", KINDS)
def test_libllama_logits_match_hf(tmp_path, kind):
    """A prefill of PROMPT, then one decode step per token of DECODE, as serving runs them: within 1e-3 of HF (F32
    weights, fp32 KV cache)."""
    hf = _hf_model(kind)
    path = _write_gguf(str(tmp_path / f"{kind}.gguf"), hf, merges=True)
    llm = _llama(path)
    rows = [torch.from_numpy(llm.eval(PROMPT, all_logits=True))]
    rows += [torch.from_numpy(llm.eval([t]))[None] for t in DECODE]
    ref = _hf_logits(hf, PROMPT + DECODE)
    assert ref.std() > 0.5  # far from uniform, so a wrong layout cannot pass at 1e-3
    torch.testing.assert_close(torch.cat(rows), ref, atol=1e-3, rtol=0)
    assert llm.n_past == len(PROMPT + DECODE)


def test_libllama_generate_is_greedy_and_reuses_the_cached_prefix(tmp_path):
    hf = _hf_model("qwen3")
    llm = _llama(_write_gguf(str(tmp_path / "q3.gguf"), hf, merges=True))
    seen = []
    out = llm.generate(PROMPT, 6, on_token=lambda t: seen.append(t))
    assert out == seen and 0 < len(out) <= 6
    ids = list(PROMPT)  # greedy against HF's argmax, token by token
    for t in out:
        assert t == int(_hf_logits(hf, ids)[-1].argmax())
        ids.append(t)
    # the same prompt again: the cached prefix is reused (only its last token is evaluated), same output
    assert llm.generate(PROMPT, 6) == out
    # a prompt that diverges after 5 tokens drops the cache beyond them
    other = PROMPT[:5] + [7, 8, 9]
    assert llm.generate(other, 1) == [int(_hf_logits(hf, other)[-1].argmax())]
    assert llm.n_past == len(other)
    assert llm.generate(PROMPT, 2, on_token=lambda t: False) == out[:1]  # on_token returning False stops


def test_libllama_truncate_and_errors(tmp_path):
    llm = _llama(_write_gguf(str(tmp_path / "l.gguf"), _hf_model("llama"), merges=True))
    first = llm.eval(PROMPT)
    llm.truncate(4)
    assert llm.n_past == 4
    torch.testing.assert_close(torch.from_numpy(llm.eval(PROMPT[4:])), torch.from_numpy(first), atol=1e-5, rtol=0)
    with pytest.raises(ValueError):
        llm.eval([])
    with pytest.raises(ValueError):
        llm.eval([1] * llm.n_ctx)  # past the context (libllama rounds the requested 64 up)
    llm.truncate(0)
    long = [i % 90 + 3 for i in range(llm.n_ctx - 1)]  # longer than one batch (n_batch is capped at n_ctx)
    assert llm.eval(long, all_logits=True).shape == (len(long), llm.n_vocab)
    with pytest.raises(RuntimeError):
        turing_llama.Llama(str(tmp_path / "missing.gguf"))


def _hf_greedy(hf, ids, n, eos=(), banned=None):
    out, ids = [], list(ids)
    for _ in range(n):
        logits = _hf_logits(hf, ids)[-1]
        if banned is not None:
            logits[banned] = float("-inf")
        t = int(logits.argmax())
        if t in eos:
            break
        out.append(t)
        ids.append(t)
    return out


def _engine(tmp_path, kind="qwen3"):
    from turing.models.gguf_loader import GGUFReader
    from turing.models.gguf_tokenizer import GGUFTokenizer
    from turing.serving.llama_engine import LlamaCppEngine
    hf = _hf_model(kind)
    path = _write_gguf(str(tmp_path / f"{kind}.gguf"), hf, merges=True)
    meta = GGUFReader(path, willneed=False).metadata
    tok = GGUFTokenizer(meta)
    return hf, tok, LlamaCppEngine(path, tok, "tiny", metadata=meta, n_ctx=64, n_threads=2, exact=True)


def test_libllama_engine_serves_the_http_api(tmp_path):
    """turing serve --backend libllama: the OpenAI completion endpoint returns HF's greedy tokens."""
    from fastapi.testclient import TestClient
    from turing.serving.server import create_app
    hf, tok, engine = _engine(tmp_path)
    want = _hf_greedy(hf, PROMPT, 5, eos=engine.eos_token_ids)
    with TestClient(create_app(engine, admission=False, lanes=False, warmup=True)) as client:
        r = client.post("/v1/completions", json={"model": "tiny", "prompt": PROMPT, "max_tokens": 5, "temperature": 0})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["choices"][0]["text"] == tok.decode(want, skip_special_tokens=True)
        assert body["usage"]["completion_tokens"] == len(want)
        assert client.get("/health").json()["device"] == "cpu"
        assert client.get("/metrics", params={"format": "json"}).json()["telemetry"]["total_tokens_generated"] >= len(want)
        too_long = client.post("/v1/completions", json={"model": "tiny", "prompt": PROMPT, "max_tokens": 1000})
        assert too_long.status_code == 400


def test_libllama_engine_logits_processor_path(tmp_path):
    """A request with a logits processor (grammar, tools, thinking budget) samples in Python each step: banning HF's
    first greedy token changes the output to HF's greedy output under the same ban."""
    import asyncio
    hf, _, engine = _engine(tmp_path)
    banned = _hf_greedy(hf, PROMPT, 1)[0]

    def ban(gen, logits):
        logits[banned] = float("-inf")
        return logits

    async def run():
        await engine.start()
        req = await engine.add_request(PROMPT, 4, temperature=0.0, logits_processor=ban)
        out = [t async for t in engine.stream_request(req)]
        await engine.stop()
        return out

    got = asyncio.run(run())
    assert banned not in got
    assert got == _hf_greedy(hf, PROMPT, 4, eos=engine.eos_token_ids, banned=banned)


@pytest.mark.parametrize("check_resident", [True, False])
def test_expert_readahead_is_exact_and_reads_the_chosen_experts(tmp_path, check_resident):
    """Wave 2: the read-ahead only reads the file, so the logits are bit-identical with it on; it runs once per MoE
    layer per evaluation, and (forced to read even resident pages) reads at least the chosen experts' bytes."""
    path = _write_gguf(str(tmp_path / "moe.gguf"), _hf_model("qwen3_moe"), merges=True)
    plain, ra = _llama(path, repack=False), _llama(path, repack=False, readahead=True, readahead_threads=4,
                                                    readahead_check_resident=check_resident)

    def run(llm):
        return torch.cat([torch.from_numpy(llm.eval(PROMPT, all_logits=True))]
                         + [torch.from_numpy(llm.eval([t]))[None] for t in DECODE])

    assert torch.equal(run(ra), run(plain))
    s = ra.readahead_stats()
    assert s["moe_layers"] == 2 and s["calls"] == 2 * (1 + len(DECODE))
    assert 0 < s["experts"] <= s["calls"] * 4 and s["bytes_wanted"] > 0
    if check_resident:
        assert s["bytes_read"] <= s["bytes_wanted"] + 2 * s["experts"] * 3 * 16384  # page rounding at most
    else:
        assert s["bytes_read"] >= s["bytes_wanted"]
    assert plain.readahead_stats()["calls"] == 0


def test_expert_readahead_needs_the_mapped_file(tmp_path):
    path = _write_gguf(str(tmp_path / "moe.gguf"), _hf_model("qwen3_moe"), merges=True)
    with pytest.raises(ValueError):
        _llama(path, readahead=True)  # repack on: the products would read a copy, not the file
