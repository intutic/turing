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
