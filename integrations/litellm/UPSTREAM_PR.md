# Upstream PR Template: `BerriAI/litellm`

**PR Title**: `feat(providers): Add native support for Turing Engine LLM serving runtime`

## Description
This PR adds native provider support for **Turing Engine** (`turing/*` model prefix), an OpenAI-compatible LLM serving runtime.

## Changes
- Added `TuringEngineConfig` in `litellm/llms/turing_engine.py`.
- Added `turing/` model routing rules to LiteLLM router proxy.
- Added unit tests in `litellm/tests/test_turing.py`.

## Example Usage
```yaml
model_list:
  - model_name: turing-llama-70b
    litellm_params:
      model: turing/llama-3.1-70b
      api_base: http://localhost:8000/v1
      api_key: os.environ/TURING_API_KEY
```

## How to Test
```bash
pytest litellm/tests/test_turing.py -v
```
