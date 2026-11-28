"""The libllama CPU backend (turing.turing_llama, csrc/cpu/turing_llama.hpp) against Hugging Face's models: the same
tiny random models and GGUF writer as tests/test_gguf_parity.py. Skips unless built with TURING_BUILD_LLAMA=1."""
import pytest
import torch

pytest.importorskip("gguf")
turing_llama = pytest.importorskip("turing.turing_llama", reason="libllama backend not built (TURING_BUILD_LLAMA=1)")

from tests.test_gguf_parity import DECODE, PROMPT, VOCAB, _hf_logits, _hf_model, _write_gguf  # noqa: E402

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


@pytest.mark.parametrize("check_resident,wait", [(True, True), (False, True), (True, False)])
def test_expert_readahead_is_exact_and_reads_the_chosen_experts(tmp_path, check_resident, wait):
    """Wave 2: the read-ahead only reads the file, so the logits are bit-identical with it on; it runs once per MoE
    layer per evaluation, and (forced to read even resident pages) reads at least the chosen experts' bytes."""
    path = _write_gguf(str(tmp_path / "moe.gguf"), _hf_model("qwen3_moe"), merges=True)
    plain, ra = _llama(path, repack=False), _llama(path, repack=False, readahead=True, readahead_threads=4,
                                                    readahead_check_resident=check_resident, readahead_wait=wait,
                                                    readahead_adaptive=False)

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


@pytest.mark.parametrize("kind", ["qwen3", "qwen3_moe"])
def test_prompt_lookup_speculation_is_exact(tmp_path, kind):
    """Wave 3: prompt-lookup drafts are kept only while each is the argmax, so in exact mode (F32 weights, fp32 KV) the
    output equals greedy decoding (with 4-bit weights and fp16 KV, batched and one-token kernels round apart); a
    prompt that repeats itself gives drafts that are accepted."""
    path = _write_gguf(str(tmp_path / f"{kind}.gguf"), _hf_model(kind), merges=True)
    plain, spec = _llama(path), _llama(path)
    prompt = PROMPT + PROMPT + PROMPT[:6]  # the continuation of a repeated pattern is in the prompt itself
    want = plain.generate(prompt, 24)
    seen = []
    got = spec.generate(prompt, 24, n_draft=4, on_token=lambda t: seen.append(t))
    assert got == want and seen == got
    s = spec.spec_stats()
    assert s["steps"] > 0 and s["drafted"] > 0 and 0 <= s["accepted"] <= s["drafted"]
    assert s["steps"] < len(got)  # fewer evaluations than tokens: some drafts were accepted
    assert spec.generate(prompt, 3, n_draft=4) == want[:3]  # stops at max_new mid-way through accepted drafts
    assert spec.n_past == len(prompt) + 2  # K/V of the prompt and the tokens fed, rejected drafts dropped


def test_libllama_engine_prompt_lookup_matches_plain(tmp_path):
    """The engine with draft tokens returns the same greedy tokens as without (exact mode), and reports its drafts."""
    import asyncio
    from turing.models.gguf_loader import GGUFReader
    from turing.models.gguf_tokenizer import GGUFTokenizer
    from turing.serving.llama_engine import LlamaCppEngine
    path = _write_gguf(str(tmp_path / "q3.gguf"), _hf_model("qwen3"), merges=True)
    meta = GGUFReader(path, willneed=False).metadata
    prompt = PROMPT + PROMPT + PROMPT[:6]

    async def run(drafts):
        engine = LlamaCppEngine(path, GGUFTokenizer(meta), "tiny", metadata=meta, n_ctx=64, n_threads=2, exact=True,
                                draft_tokens=drafts)
        await engine.start()
        req = await engine.add_request(prompt, 20, temperature=0.0)
        out = [t async for t in engine.stream_request(req)]
        await engine.stop()
        return out, engine.get_telemetry()["speculation"]

    plain, s0 = asyncio.run(run(0))
    spec, s4 = asyncio.run(run(4))
    assert spec == plain and s0["drafted_tokens"] == 0
    assert s4["drafted_tokens"] > 0 and 0 < s4["acceptance_rate"] <= 1


def test_expert_readahead_adapts_to_the_miss_share(tmp_path):
    """Adaptive (opt-in): it reads only after a window of router outputs in which at least 5% of the chosen experts'
    bytes missed the cache. Every page counted missing (check_resident off) switches it on after the first window; a
    small file that stays in the page cache never does. The default reads from the first router output."""
    path = _write_gguf(str(tmp_path / "moe.gguf"), _hf_model("qwen3_moe"), merges=True)
    always = _llama(path, repack=False, readahead=True, readahead_check_resident=False)
    always.eval([PROMPT[0]])
    assert always.readahead_stats()["active_calls"] == 2  # both MoE layers read on the first token
    cold = _llama(path, repack=False, readahead=True, readahead_check_resident=False, readahead_window=2,
                  readahead_adaptive=True)
    for t in PROMPT[:6]:
        cold.eval([t])
    s = cold.readahead_stats()
    assert s["calls"] == 12 and s["active_calls"] == 10  # the first window (2 router outputs) only looks
    assert s["bytes_read"] > 0 and s["bytes_missing"] >= s["bytes_read"]
    warm = _llama(path, repack=False, readahead=True, readahead_window=2, readahead_adaptive=True)
    warm.eval(PROMPT)  # a fresh file is in the page cache: nothing misses
    for t in DECODE:
        warm.eval([t])
    s = warm.readahead_stats()
    assert s["active_calls"] == 0 and s["bytes_read"] == 0 and s["bytes_missing"] <= s["bytes_wanted"] // 20


def test_step_decodes_several_sequences_like_one_each(tmp_path):
    """Wave 7: one batched step over two sequences gives each the logits it gets alone (exact mode)."""
    path = _write_gguf(str(tmp_path / "q3.gguf"), _hf_model("qwen3"), merges=True)
    a, b = PROMPT, [5, 9, 77, 31, 2, 64, 42]
    alone = [torch.from_numpy(_llama(path).eval(x)) for x in (a, b)]
    both = _llama(path, n_seq=2)
    rows = torch.from_numpy(both.step([0, 1], [a, b], [0, 0]))
    torch.testing.assert_close(rows[0], alone[0], atol=1e-4, rtol=0)
    torch.testing.assert_close(rows[1], alone[1], atol=1e-4, rtol=0)
    nxt = torch.from_numpy(both.step([1, 0], [[DECODE[0]], [DECODE[1]]], [len(b), len(a)]))  # order and positions per sequence
    torch.testing.assert_close(nxt[0], _hf_logits(_hf_model("qwen3"), b + [DECODE[0]])[-1], atol=1e-3, rtol=0)
    torch.testing.assert_close(nxt[1], _hf_logits(_hf_model("qwen3"), a + [DECODE[1]])[-1], atol=1e-3, rtol=0)
    both.seq_clear(0)
    torch.testing.assert_close(torch.from_numpy(both.step([0], [a], [0]))[0], alone[0], atol=1e-4, rtol=0)


def test_batched_engine_serves_concurrent_requests_exactly(tmp_path):
    """max_batch_size 3: three requests decode together, one with a logits processor; each gets HF's greedy tokens."""
    import asyncio
    from turing.models.gguf_loader import GGUFReader
    from turing.models.gguf_tokenizer import GGUFTokenizer
    from turing.serving.llama_engine import LlamaCppEngine
    hf = _hf_model("qwen3")
    path = _write_gguf(str(tmp_path / "q3.gguf"), hf, merges=True)
    meta = GGUFReader(path, willneed=False).metadata
    engine = LlamaCppEngine(path, GGUFTokenizer(meta), "tiny", metadata=meta, n_ctx=128, n_threads=2, exact=True,
                            max_batch_size=3)
    prompts = [PROMPT, [5, 9, 77, 31, 2, 64, 42], [8, 8, 30, 12]]
    banned = _hf_greedy(hf, prompts[2], 1)[0]

    def ban(gen, logits):
        logits[banned] = float("-inf")
        return logits

    async def run():
        await engine.start()
        await engine.warm_up()  # as turing serve does: its prefill must not leave tokens in sequence 0
        reqs = [await engine.add_request(p, 5, temperature=0.0, logits_processor=ban if i == 2 else None)
                for i, p in enumerate(prompts)]
        outs = await asyncio.gather(*[(lambda r: _collect(engine, r))(r) for r in reqs])
        util = engine.get_kv_cache_utilization()
        await engine.stop()
        return outs, util

    outs, util = asyncio.run(run())
    eos = engine.eos_token_ids
    assert outs[0] == _hf_greedy(hf, prompts[0], 5, eos=eos)
    assert outs[1] == _hf_greedy(hf, prompts[1], 5, eos=eos)
    assert outs[2] == _hf_greedy(hf, prompts[2], 5, eos=eos, banned=banned)
    assert util == 0.0  # every sequence freed


async def _collect(engine, req):
    return [t async for t in engine.stream_request(req)]


def _write_mamba_gguf(path, n_embd=64, n_layer=2, d_inner=128, d_state=16, d_conv=4, dt_rank=4, vocab=VOCAB):
    """A tiny Mamba with random weights, in the tensor layout llama.cpp's mamba model loads (src/models/mamba.cpp):
    recurrent memory, which keeps no per-token state to drop a suffix of."""
    import gguf
    import numpy as np
    rng = np.random.default_rng(0)

    def w(*shape):  # numpy order is the reverse of ggml's ne
        return (rng.standard_normal(shape) / np.sqrt(shape[-1])).astype(np.float32)

    g = gguf.GGUFWriter(path, "mamba")
    g.add_name("tiny-mamba")
    g.add_context_length(128)
    g.add_embedding_length(n_embd)
    g.add_block_count(n_layer)
    g.add_feed_forward_length(0)
    g.add_head_count(0)
    g.add_ssm_conv_kernel(d_conv)
    g.add_ssm_inner_size(d_inner)
    g.add_ssm_state_size(d_state)
    g.add_ssm_time_step_rank(dt_rank)
    g.add_layer_norm_rms_eps(1e-5)
    g.add_file_type(gguf.LlamaFileType.ALL_F32)
    g.add_tokenizer_model("gpt2")
    g.add_token_list([f"<tok_{i}>" for i in range(vocab)])
    g.add_token_types([1] * vocab)
    g.add_token_merges(["<tok_3> <tok_4>"])
    g.add_bos_token_id(1)
    g.add_eos_token_id(2)
    g.add_tensor("token_embd.weight", w(vocab, n_embd))
    g.add_tensor("output_norm.weight", np.ones(n_embd, np.float32))
    g.add_tensor("output.weight", w(vocab, n_embd))
    for i in range(n_layer):
        p = f"blk.{i}."
        g.add_tensor(p + "attn_norm.weight", np.ones(n_embd, np.float32))
        g.add_tensor(p + "ssm_in.weight", w(2 * d_inner, n_embd))
        g.add_tensor(p + "ssm_conv1d.weight", w(d_inner, d_conv))
        g.add_tensor(p + "ssm_conv1d.bias", np.zeros(d_inner, np.float32))
        g.add_tensor(p + "ssm_x.weight", w(dt_rank + 2 * d_state, d_inner))
        g.add_tensor(p + "ssm_dt.weight", w(d_inner, dt_rank))
        g.add_tensor(p + "ssm_dt.bias", np.full(d_inner, -2.0, np.float32))
        g.add_tensor(p + "ssm_a", -rng.uniform(0.5, 2.0, (d_inner, d_state)).astype(np.float32))  # A = -exp(A_log)
        g.add_tensor(p + "ssm_d", np.ones(d_inner, np.float32))
        g.add_tensor(p + "ssm_out.weight", w(n_embd, d_inner))
    g.write_header_to_file()
    g.write_kv_data_to_file()
    g.write_tensors_to_file()
    g.close()
    return path


def test_prompt_reuse_on_recurrent_memory(tmp_path):
    """Recurrent and hybrid memory (Mamba here; Inkling 2, Qwen3-Next, Falcon-H1) cannot drop a cached suffix, so a
    prompt that shares only a prefix with the cache, or repeats it, runs again from the start; speculation drafts
    nothing (it could not roll back rejected drafts). Every path gives a fresh context's tokens."""
    path = _write_mamba_gguf(str(tmp_path / "mamba.gguf"))

    def fresh(prompt, n):
        return _llama(path).generate(prompt, n)

    llm = _llama(path)
    first = llm.generate(PROMPT, 6)
    assert first == fresh(PROMPT, 6)
    other = PROMPT[:5] + [7, 8, 9]  # shares 5 tokens with the cache: the suffix cannot be dropped
    assert llm.generate(other, 4) == fresh(other, 4)
    assert llm.cached_prompt_tokens == 0
    assert llm.generate(PROMPT, 6) == first  # the whole prompt again: its last token cannot be re-evaluated in place
    longer = _llama(path)
    longer.eval(PROMPT[:5])
    longer.prefill(PROMPT)  # the cache is a prefix of the prompt: appending needs no truncation, so it is reused
    assert longer.cached_prompt_tokens == 5
    spec = _llama(path)
    rep = PROMPT + PROMPT + PROMPT[:6]
    assert spec.generate(rep, 12, n_draft=4) == fresh(rep, 12)
    assert spec.spec_stats()["drafted"] == 0


def test_libllama_tokenizer_matches_hf_on_a_byte_level_vocabulary(tmp_path):
    """turing serve --backend libllama tokenizes with libllama (LlamaTokenizer, tokenizer None): on SmolLM2's byte-level
    BPE it encodes, decodes and renders the chat template as Hugging Face does, and the chat endpoint serves HF's greedy
    text. GGUFTokenizer turned "a long story" into "alongstory" here."""
    import transformers as tf
    from fastapi.testclient import TestClient
    from turing.models.gguf_loader import GGUFReader
    from turing.serving.llama_engine import LlamaCppEngine
    from turing.serving.server import create_app
    name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    try:
        hf_tok = tf.AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"tokenizer {name} is not in the local Hugging Face cache: {e}")
    hf = _hf_model("qwen3", vocab_size=len(hf_tok))
    path = _write_gguf(str(tmp_path / "q3.gguf"), hf, tokenizer=hf_tok)
    engine = LlamaCppEngine(path, None, "tiny", metadata=GGUFReader(path, willneed=False).metadata, n_ctx=256,
                            n_threads=2, exact=True)
    tok = engine.tokenizer
    for text in ["Write a long, detailed story.", "  two  spaces\n\nand\ttabs", "café naïve 日本語 🦙", "x = f(1); # ok",
                 "<|im_start|>user\nhi<|im_end|>"]:
        ids = tok.encode(text, add_special_tokens=False)
        assert ids == hf_tok.encode(text, add_special_tokens=False), text
        assert tok.decode(ids) == hf_tok.decode(ids, skip_special_tokens=True), text
    msgs = [{"role": "user", "content": "Hello there"}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    assert text == hf_tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    assert tok.convert_tokens_to_ids("<|im_end|>") == hf_tok.convert_tokens_to_ids("<|im_end|>")
    assert hf_tok.eos_token_id in engine.eos_token_ids
    prompt = hf_tok.encode(text, add_special_tokens=False)
    want = _hf_greedy(hf, prompt, 4, eos=engine.eos_token_ids)
    with TestClient(create_app(engine, admission=False, lanes=False, warmup=True)) as client:
        r = client.post("/v1/chat/completions", json={"model": "tiny", "messages": msgs, "max_tokens": 4, "temperature": 0})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["usage"]["prompt_tokens"] == len(prompt)
        assert body["choices"][0]["message"]["content"] == hf_tok.decode(want, skip_special_tokens=True)
