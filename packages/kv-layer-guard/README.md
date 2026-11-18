# kv-layer-guard

Find the attention layers a low-bit KV cache breaks, before serving with it.

Some models lose almost everything when their KV cache is stored in fp8, int8 or int4, and the server gives no sign of
it. On one NVIDIA L4 with vLLM 0.31, Qwen2.5-1.5B-Instruct (bf16) scored:

| KV cache | GSM8K (500, 8-shot) | Pass-key retrieval, 2K–32K |
| :--- | :---: | :---: |
| bf16 | 57.4% | 100% |
| fp8 (default) | 2.4% | 5% |
| fp8, layer 0 skipped | 56.8% | 100% |
| int4 per token-head | 0.6% | 0% |
| int4 per token-head, layer 0 skipped | 47.8% | 100% |

The cause is a few layers whose keys the format cannot represent (in Qwen2, key-projection biases put some channels of
layer 0 at 30–70× the rest). vLLM can keep chosen layers in full precision (`kv_cache_dtype_skip_layers`); this package
tells you which ones.

## How it works

For each layer, only that layer's keys (after RoPE, as the cache stores them) and values take the format's round
trip during a forward pass over random tokens, through a `transformers` cache that quantizes on write. The mean KL
divergence of the next-token distribution from the exact model is measured per layer; layers above 0.02 nats are the
ones to keep (at most a quarter of the layers). Random tokens are enough: the damage comes from the weights, not the
text. One pass costs (layers + 1) forwards of 4 × 256 tokens.

## Use

```bash
pip install ./packages/kv-layer-guard   # from a Turing Engine checkout; not on PyPI yet
python -m kv_layer_guard Qwen/Qwen2.5-1.5B-Instruct --kv-cache-dtype fp8
# {"kv_cache_dtype_skip_layers": ["0"], "layer_kl": [...]}
```

```python
from kv_layer_guard import vllm_skip_layers
from vllm import LLM

model = "Qwen/Qwen2.5-1.5B-Instruct"
llm = LLM(model, kv_cache_dtype="fp8", kv_cache_dtype_skip_layers=vllm_skip_layers(model, "fp8"))
```

Formats: `fp8` (e4m3, scale 1.0, saturating, as vLLM's default fp8 cache), `int8_per_token_head` and
`int4_per_token_head` (asymmetric, one scale and zero per token and head; an approximation of vLLM's kernels).

## Scope

Validated on six instruct models on an NVIDIA L4 (vLLM 0.31, fp8 KV cache, bf16 weights; GSM8K 8-shot on 500
problems unless noted; pass-key retrieval at 5 depths per length). The Turing columns are the same calibration inside
[Turing Engine](https://github.com/intutic/turing)'s int8/int4 KV pools, where it was first measured on Qwen2.5 and
SmolLM2.

<!-- table:validation -->
| Model | vLLM bf16 | vLLM fp8 (default) | Guard keeps in bf16 | vLLM fp8 + guard | Turing int8 KV | Turing int4 KV | Pass-key: fp8 / fp8 + guard / Turing int4 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-1.5B-Instruct | 57.4% | 2.2% | 0 | 56.8% | 58.2% | 52.8% | 5.0% / 100.0% / 100.0% |
| Llama-3.2-3B-Instruct | 67.2% | 65.8% | none | = fp8 | 67.0% | 64.4% | 100.0% / 100.0% / 100.0% |
| gemma-2-2b-it | 40.8% | 40.4% | none | = fp8 | 40.2% | 38.0% | 100.0% / 100.0% / 100.0% |
| Phi-3.5-mini-instruct | 71.8% | 66.6% | none | = fp8 | 79.8% | 77.8% | 100.0% / 100.0% / 100.0% |
| mistral-7b-instruct-v0.3 | 47.8% | 47.0% | none | = fp8 | — | 46.0% (n=200) | 100.0% / 100.0% / 100.0% |
| Qwen2.5-7B-Instruct | 74.6% | 2.4% | 0, 1, 27 | 70.2% | — | 71.5% (n=200) | 0.0% / 100.0% / 100.0% |
<!-- /table -->

- It keeps layers only where vLLM's default fp8 cache breaks the model: layer 0 of Qwen2.5-1.5B and layers 0, 1 and
  27 of Qwen2.5-7B. On Llama 3.2, Gemma 2, Phi-3.5 and Mistral it keeps none, and their fp8 runs need none.
- **Limit:** Phi-3.5-mini loses 5.2 GSM8K points under fp8 (71.8% to 66.6%; standard errors about 2 points each) and
  the guard does not flag it: its largest per-layer KL is 0.006, under the 0.02 threshold. A loss spread thinly over
  many layers is not what a per-layer check catches.

## License

Apache License 2.0 ([LICENSE](LICENSE)). The rest of the Turing Engine repository is under the Business Source
License 1.1; this directory is not.
