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

The same calibration runs inside [Turing Engine](https://github.com/intutic/turing)'s int8/int4 KV pools, where it was
measured on Qwen2.5 (0.5B, 1.5B, 1.5B-Instruct) and SmolLM2 (360M, 1.7B): it keeps layer 0 for the Qwen models and no
layer for SmolLM2. The vLLM rows above skip layer 0 by hand; this package's own selections for vLLM's formats, and
other model families, are being validated.

## License

Apache License 2.0 ([LICENSE](LICENSE)). The rest of the Turing Engine repository is under the Business Source
License 1.1; this directory is not.
