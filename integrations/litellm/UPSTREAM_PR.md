# LiteLLM upstream status

BerriAI/litellm#38181 adds `turing` to LiteLLM's JSON provider registry (`litellm/llms/openai_like/providers.json`):
`turing/<model>` routes chat, streaming, `text_completion` (the server's native `/v1/completions`) and embeddings to a
`turing serve` instance, with `TURING_API_BASE` (default `http://localhost:8000/v1`) and `TURING_API_KEY`. It was
rebuilt on 2026-11-20 (commit 7f084f2d, 7 unit tests, checked with the LiteLLM SDK against a live server, not through
the proxy) and is waiting for review.

The PR's first version added a `TuringConfig` class that LiteLLM never registered, so `turing/` routed nowhere, and
it exposed `sparsity_ratio` and `use_svd_kv`, which the server now rejects with HTTP 400.

Set `TURING_API_KEY` (any value unless the server runs with `--api-key`): with it unset and `OPENAI_API_KEY` set,
LiteLLM sends the OpenAI key to the Turing server, as it does for its other JSON providers.

Until it merges, use LiteLLM's `openai/` prefix with `api_base` ([docs/integrations.md §2](../../docs/integrations.md)).
`turing_engine.py` here is a small helper for that setup, not the upstream code.
